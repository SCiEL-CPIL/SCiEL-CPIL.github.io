"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_DEADLINE = time.time() + 320.0  # safety margin before 330s kill


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the finite-horizon MDP by backward induction.
    
    State: (step n, inventory q)
    Action: (bid_depth, ask_depth) from a discrete grid
    Reward: depth_earned - (gamma/2)*sigma^2*q^2*dt
    
    The mid moves randomly but since the objective is pathwise and the
    mid shift doesn't affect the depth earned (only the absolute price,
    not the depth), the optimal depth depends only on (n, q, params).
    
    Actually, let me reconsider. The mid moves by sigma*sqrt(dt)*N(0,1).
    The depth is a distance from mid. The fill probability depends on depth.
    The reward from a fill is the depth itself. The mid movement doesn't
    change the depth-to-mid relationship for the NEXT step's quotes because
    we re-quote relative to the new mid each step.
    
    So the problem decomposes: at each step, given current inventory q and
    time remaining, choose bid_depth and ask_depth to maximize expected
    immediate reward + continuation value.
    
    Immediate reward if bid fills: bid_depth - (gamma/2)*sigma^2*q^2*dt
    (the q^2 term is charged regardless)
    
    Wait, re-reading: "depths earned minus sum over steps of (gamma/2) sigma^2 q^2 dt"
    So the inventory cost is charged every step regardless of fills.
    
    Let V(n, q) = max expected total reward from step n to end, given inventory q at start of step n.
    
    At step n, we choose bid_depth b >= 0, ask_depth a >= 0.
    p_bid = 1 - exp(-A * exp(-k*b) * dt)  [if q < Q]
    p_ask = 1 - exp(-A * exp(-k*a) * dt)  [if q > -Q]
    
    Immediate reward:
    - If bid fills: +b (earn depth), inventory becomes q-1
    - If ask fills: +a (earn depth), inventory becomes q+1
    - Inventory cost: -(gamma/2)*sigma^2*q^2*dt (always charged)
    
    V(n, q) = -cost(q) + max_{b,a} [ p_bid*(b + V(n+1, q-1)) + p_ask*(a + V(n+1, q+1)) 
              + (1-p_bid)*(1-p_ask)*V(n+1, q) 
              + p_bid*p_ask*(b+a + V(n+1, q)) ]
    
    Wait, if both fill: inventory goes q-1+1 = q, reward b+a.
    
    V(n, q) = -cost(q) + max_{b,a} [ 
        p_bid*(1-p_ask)*(b + V(n+1, q-1)) + 
        p_ask*(1-p_bid)*(a + V(n+1, q+1)) + 
        p_bid*p_ask*(b+a + V(n+1, q)) + 
        (1-p_bid)*(1-p_ask)*V(n+1, q) ]
    
    V(steps, q) = 0 (no terminal charge)
    
    This is a 2D optimization at each state. I'll use a grid for depths.
    """
    dt = T / steps
    n_states = 2 * Q + 1  # q from -Q to Q
    
    # Depth grid: from 0 to some max
    # Fill prob: 1 - exp(-A*exp(-k*d)*dt)
    # For large d, fill prob -> 0. For d=0, fill prob = 1-exp(-A*dt)
    # Optimal depth is where marginal benefit = marginal cost
    # Let's use a grid of ~50 points from 0 to 5/k
    max_depth = max(5.0 / k, 1.0)
    n_depth = 50
    depth_grid = np.linspace(0.0, max_depth, n_depth)
    
    # Precompute fill probabilities for each depth
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # V[n, q] for n in 0..steps, q in -Q..Q
    # V[steps, q] = 0
    V = np.zeros((steps + 1, n_states), dtype=np.float64)
    # Store optimal depths
    opt_bid = np.zeros((steps, n_states), dtype=np.float64)
    opt_ask = np.zeros((steps, n_states), dtype=np.float64)
    
    cost_coeff = 0.5 * gamma * sigma * sigma * dt
    
    for n in range(steps - 1, -1, -1):
        for qi in range(n_states):
            q = qi - Q
            cost = cost_coeff * q * q
            
            best_val = -np.inf
            best_b = depth_grid[0]
            best_a = depth_grid[0]
            
            # Bid fill only possible if q < Q
            # Ask fill only possible if q > -Q
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            if not can_bid and not can_ask:
                # No fills possible, just cost
                V[n, qi] = -cost + V[n + 1, qi]
                opt_bid[n, qi] = depth_grid[0]
                opt_ask[n, qi] = depth_grid[0]
                continue
            
            # Try all combinations of bid and ask depths
            # For speed, we can optimize over one side at a time
            # But let's do full grid search with early termination if too slow
            
            # To keep it fast, use a smaller grid or optimize analytically
            # For now, use the full grid but limit to reasonable depths
            
            # Optimize over bid depth
            bid_vals = []
            if can_bid:
                for bi in range(n_depth):
                    b = depth_grid[bi]
                    p_b = fill_probs[bi]
                    # For each ask depth, compute value
                    # V = p_b*(1-p_a)*(b + V[n+1, q-1]) + p_a*(1-p_b)*(a + V[n+1, q+1]) 
                    #     + p_b*p_a*(b+a + V[n+1, q]) + (1-p_b)*(1-p_a)*V[n+1, q]
                    # = p_b*(b + V[n+1,q-1]) - p_b*p_a*(b + V[n+1,q-1]) 
                    #   + p_a*(a + V[n+1,q+1]) - p_a*p_b*(a + V[n+1,q+1])
                    #   + p_b*p_a*(b+a + V[n+1,q]) + (1-p_b)*(1-p_a)*V[n+1,q]
                    # Simplify:
                    # = p_b*(b + V[n+1,q-1]) + p_a*(a + V[n+1,q+1]) 
                    #   + p_b*p_a*(V[n+1,q] - V[n+1,q-1] - V[n+1,q+1])
                    #   + (1-p_b)*(1-p_a)*V[n+1,q]
                    # Let me just compute directly for clarity
                    
                    v_next_q = V[n + 1, qi]
                    v_next_qm1 = V[n + 1, qi - 1]  # q-1
                    v_next_qp1 = V[n + 1, qi + 1]  # q+1
                    
                    for ai in range(n_depth):
                        a = depth_grid[ai]
                        p_a = fill_probs[ai] if can_ask else 0.0
                        
                        if can_bid and can_ask:
                            val = (p_b * (1 - p_a) * (b + v_next_qm1) +
                                   p_a * (1 - p_b) * (a + v_next_qp1) +
                                   p_b * p_a * (b + a + v_next_q) +
                                   (1 - p_b) * (1 - p_a) * v_next_q)
                        elif can_bid:
                            val = p_b * (b + v_next_qm1) + (1 - p_b) * v_next_q
                        else:
                            val = p_a * (a + v_next_qp1) + (1 - p_a) * v_next_q
                        
                        if val > best_val:
                            best_val = val
                            best_b = b
                            best_a = a
            else:
                # No bid, optimize ask only
                v_next_q = V[n + 1, qi]
                v_next_qp1 = V[n + 1, qi + 1]
                for ai in range(n_depth):
                    a = depth_grid[ai]
                    p_a = fill_probs[ai]
                    val = p_a * (a + v_next_qp1) + (1 - p_a) * v_next_q
                    if val > best_val:
                        best_val = val
                        best_b = depth_grid[0]
                        best_a = a
            
            V[n, qi] = -cost + best_val
            opt_bid[n, qi] = best_b
            opt_ask[n, qi] = best_a
        
        # Check deadline
        if time.time() > _DEADLINE - 10:
            break
    
    return V, opt_bid, opt_ask


def _get_policy(params):
    """Get or compute the optimal policy for the given regime."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _solve_regime(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Quote using precomputed optimal policy from MDP backward induction.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n_paths = len(inventory)
    inv = np.asarray(inventory, dtype=np.int32)
    
    # Check deadline
    if time.time() > _DEADLINE - 5:
        # Fallback: simple heuristic
        depth = np.full(n_paths, 1.0 / k)
        return depth, depth.copy()
    
    # Get or compute policy
    try:
        V, opt_bid, opt_ask = _get_policy(params)
    except Exception:
        depth = np.full(n_paths, 1.0 / k)
        return depth, depth.copy()
    
    # Determine step index
    dt = T / steps
    n = min(int(round(t / dt)), steps - 1)
    n = max(0, n)
    
    # Map inventory to state index: q -> q + Q
    state_idx = inv + Q
    state_idx = np.clip(state_idx, 0, 2 * Q)
    
    # Look up optimal depths
    bid_depths = opt_bid[n, state_idx].astype(np.float64)
    ask_depths = opt_ask[n, state_idx].astype(np.float64)
    
    # Ensure valid: no bid at q=Q, no ask at q=-Q
    # At q=Q, bid should be inf (no quote)
    # At q=-Q, ask should be inf (no quote)
    bid_depths[inv == Q] = np.inf
    ask_depths[inv == -Q] = np.inf
    
    # Validate
    min_allowed = -3.0 / k
    bid_depths = np.where(bid_depths < min_allowed, min_allowed + 1e-10, bid_depths)
    ask_depths = np.where(ask_depths < min_allowed, min_allowed + 1e-10, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
