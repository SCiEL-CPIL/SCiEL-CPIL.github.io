# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0,5,3), (1,7,2), (2,9,4), (3,6,1), (0,4,5),
        (2,8,2), (5,9,3), (1,4,2), (6,0,4), (3,8,1),
        (7,2,3), (4,9,2), (8,1,5), (0,6,1), (5,2,2)
    ]
    
    # Precompute edge masks for CW and CCW routing for each demand
    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        cw = np.zeros(N)
        if s <= t:
            edges = list(range(s, t))
        else:
            edges = list(range(s, N)) + list(range(0, t))
        for e in edges:
            cw[e] = 1
        cw_masks.append(cw)
        ccw = 1.0 - cw
        ccw_masks.append(ccw)
    
    cw_masks = np.array(cw_masks)  # shape (D, N)
    ccw_masks = np.array(ccw_masks)  # shape (D, N)
    weights = np.array([d[2] for d in demands])  # shape (D,)
    
    def evaluate(routing):
        """Compute max edge load given routing vector (0=CW, 1=CCW)."""
        delta_masks = ccw_masks - cw_masks
        base_load = (cw_masks * weights[:, None]).sum(axis=0)
        weighted_delta = delta_masks * weights[:, None]
        delta_load = (weighted_delta * routing[:, None]).sum(axis=0)
        total_load = base_load + delta_load
        return np.max(total_load)
    
    # Try MIP solver first for global optimum
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Variables: x[0..14] = routing decisions (0 or 1), x[15] = t (max load)
        # Minimize t
        # Constraints: for each edge e, sum_d w_d * (cw_d[e] + (ccw_d[e]-cw_d[e])*x_d) <= t
        
        c = np.zeros(D + 1)
        c[-1] = 1.0
        
        delta_masks = ccw_masks - cw_masks  # (D, N)
        base_load = (cw_masks * weights[:, None]).sum(axis=0)  # (N,)
        
        A = np.zeros((N, D + 1))
        A[:, :D] = (delta_masks * weights[:, None]).T
        A[:, D] = -1.0
        
        rhs = -base_load
        
        constraint = LinearConstraint(A, -np.inf, rhs)
        
        lb = np.zeros(D + 1)
        ub = np.ones(D + 1)
        ub[-1] = np.inf
        bounds = Bounds(lb, ub)
        
        integrality = np.zeros(D + 1)
        integrality[:D] = 1.0
        
        result = milp(c=c, constraints=constraint, bounds=bounds, integrality=integrality)
        
        if result.success and result.x is not None:
            routing_mip = result.x[:D].round()
            routing_mip = np.where(routing_mip >= 0.5, 1.0, 0.0)
            score_mip = evaluate(routing_mip)
            
            best_routing = routing_mip.copy()
            best_score = score_mip
        else:
            best_routing = np.zeros(D, dtype=np.float64)
            best_score = evaluate(best_routing)
    except Exception:
        best_routing = np.zeros(D, dtype=np.float64)
        best_score = evaluate(best_routing)
    
    start_time = time.time()
    deadline = start_time + 600
    
    # Simulated annealing polish from best found
    rng = np.random.default_rng(42)
    
    for restart in range(500):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            routing = best_routing.copy()
        else:
            routing = rng.random(D).round()
        
        current_score = evaluate(routing)
        if current_score < best_score:
            best_score = current_score
            best_routing = routing.copy()
        
        T = 5.0
        T_min = 0.01
        alpha = 0.9995
        steps_per_iter = 5000
        
        for step in range(steps_per_iter):
            if time.time() > deadline - 10:
                break
            
            d = rng.integers(0, D)
            old_val = routing[d]
            routing[d] = 1.0 - old_val
            
            new_score = evaluate(routing)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_routing = routing.copy()
            else:
                routing[d] = old_val
            
            T *= alpha
            if T < T_min:
                break
    
    # Single-flip polish
    for i in range(D):
        if time.time() > deadline - 5:
            break
        trial = best_routing.copy()
        trial[i] = 1.0 - trial[i]
        s = evaluate(trial)
        if s < best_score:
            best_score = s
            best_routing = trial
    
    # Pair-flip polish
    for i in range(D):
        if time.time() > deadline - 5:
            break
        for j in range(i+1, D):
            trial = best_routing.copy()
            trial[i] = 1.0 - trial[i]
            trial[j] = 1.0 - trial[j]
            s = evaluate(trial)
            if s < best_score:
                best_score = s
                best_routing = trial
    
    return best_routing

# EVOLVE-BLOCK-END
