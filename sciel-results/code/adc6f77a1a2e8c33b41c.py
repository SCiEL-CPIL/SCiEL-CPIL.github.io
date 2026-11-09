# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute CW edge masks as bitmask
    cw_masks = []
    for s, t, w in demands:
        mask = 0
        if s <= t:
            for e in range(s, t):
                mask |= (1 << e)
        else:
            for e in range(s, N):
                mask |= (1 << e)
            for e in range(0, t):
                mask |= (1 << e)
        cw_masks.append(mask)
    
    all_edges_mask = (1 << N) - 1
    ccw_masks = [all_edges_mask ^ m for m in cw_masks]
    
    weights = np.array([w for s, t, w in demands], dtype=np.float64)
    
    # Build edge-incidence matrix: A[d, e] = 1 if edge e is on CW path of demand d
    A = np.zeros((D, N), dtype=np.float64)
    for d in range(D):
        mask = cw_masks[d]
        for e in range(N):
            if mask & (1 << e):
                A[d, e] = 1.0
    
    # For routing x in [0,1]^D where x[d]=0 means CW, x[d]=1 means CCW:
    # Edge load on edge e = sum_d w_d * (A[d,e] * (1-x[d]) + (1-A[d,e]) * x[d])
    # = sum_d w_d * A[d,e] + sum_d w_d * (1-2*A[d,e]) * x[d]
    
    base_loads = A.T @ weights  # shape (N,)
    delta = weights * (1 - 2 * A)  # shape (D, N)
    
    def compute_max_load_exact(routing):
        """Compute max edge load given binary routing decisions."""
        x = (routing >= 0.5).astype(np.float64)
        loads = base_loads + delta.T @ x
        return np.max(loads)
    
    def soft_max_surrogate(x, beta=20.0):
        """Smooth approximation of max load using log-sum-exp."""
        x = np.clip(x, 0, 1)
        loads = base_loads + delta.T @ x
        scaled = beta * loads
        max_val = np.max(scaled)
        logsumexp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
        return logsumexp / beta
    
    def soft_max_grad(x, beta=20.0):
        """Gradient of the soft max surrogate."""
        x = np.clip(x, 0, 1)
        loads = base_loads + delta.T @ x
        scaled = beta * loads
        max_val = np.max(scaled)
        exps = np.exp(scaled - max_val)
        softmax_weights = exps / np.sum(exps)
        grad = delta @ softmax_weights
        return grad
    
    start_time = time.time()
    deadline = start_time + 620.0
    
    # Phase 1: Use smooth optimization from multiple starting points
    best_routing = np.zeros(D)
    best_load = compute_max_load_exact(best_routing)
    
    np.random.seed(42)
    
    # Try smooth optimization from several random starts
    for restart in range(30):
        if time.time() > deadline - 10.0:
            break
        
        x0 = np.random.rand(D)
        
        # Use L-BFGS-B with the smooth surrogate
        result = minimize(
            soft_max_surrogate,
            x0,
            jac=soft_max_grad,
            method='L-BFGS-B',
            bounds=[(0, 1)] * D,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        # Round to binary
        x_rounded = (result.x >= 0.5).astype(np.float64)
        load = compute_max_load_exact(x_rounded)
        
        if load < best_load - 1e-9:
            best_routing = x_rounded
            best_load = load
    
    # Phase 2: Local search polish (flip one at a time, then pairs)
    improved = True
    while improved:
        improved = False
        for i in range(D):
            if time.time() > deadline - 5.0:
                break
            trial = best_routing.copy()
            trial[i] = 1.0 - trial[i]
            load = compute_max_load_exact(trial)
            if load < best_load - 1e-9:
                best_routing = trial
                best_load = load
                improved = True
        if time.time() > deadline - 5.0:
            break
    
    # Phase 3: Pairwise flip search
    if time.time() < deadline - 5.0:
        pair_improved = True
        while pair_improved:
            pair_improved = False
            for i in range(D):
                for j in range(i + 1, D):
                    if time.time() > deadline - 5.0:
                        break
                    trial = best_routing.copy()
                    trial[i] = 1.0 - trial[i]
                    trial[j] = 1.0 - trial[j]
                    load = compute_max_load_exact(trial)
                    if load < best_load - 1e-9:
                        best_routing = trial
                        best_load = load
                        pair_improved = True
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
    
    # Phase 4: More smooth optimization from best solution and neighbors
    if time.time() < deadline - 5.0:
        # Try from the current best and its neighbors
        starts = [best_routing.copy()]
        for i in range(D):
            nb = best_routing.copy()
            nb[i] = 1.0 - nb[i]
            starts.append(nb)
        
        for x0 in starts:
            if time.time() > deadline - 5.0:
                break
            result = minimize(
                soft_max_surrogate,
                x0,
                jac=soft_max_grad,
                method='L-BFGS-B',
                bounds=[(0, 1)] * D,
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            x_rounded = (result.x >= 0.5).astype(np.float64)
            load = compute_max_load_exact(x_rounded)
            if load < best_load - 1e-9:
                best_routing = x_rounded
                best_load = load
                # Quick local polish
                for k in range(3):
                    if time.time() > deadline - 5.0:
                        break
                    local_improved = True
                    while local_improved:
                        local_improved = False
                        for i in range(D):
                            if time.time() > deadline - 5.0:
                                break
                            trial = best_routing.copy()
                            trial[i] = 1.0 - trial[i]
                            load = compute_max_load_exact(trial)
                            if load < best_load - 1e-9:
                                best_routing = trial
                                best_load = load
                                local_improved = True
                        if time.time() > deadline - 5.0:
                            break
    
    return best_routing

# EVOLVE-BLOCK-END
