"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 280.0  # Safety margin before 300s kill
_POLICY_CACHE = {}


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the exact optimal policy via dynamic programming.
    
    Returns:
        opt_bid: (steps, 2*Q+1) array of optimal bid depths
        opt_ask: (steps, 2*Q+1) array of optimal ask depths
        V: (steps+1, 2*Q+1) value function
    """
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    
    # V[t, q_idx] where q_idx = q + Q, q in [-Q, Q]
    # Terminal: V[N, q] = 0 (no terminal charge)
    V = np.zeros((N + 1, n_q))
    opt_bid = np.zeros((N, n_q))
    opt_ask = np.zeros((N, n_q))
    
    # Precompute probability function
    # P_fill(d) = 1 - exp(-A * exp(-k*d) * dt)
    # We need to optimize over d >= 0
    # For each step and inventory, the optimal depth solves:
    #   max_d  { P(d) * d + (1-P(d)) * E[V_{t+1}(q')] - P(d) * E[V_{t+1}(q'+delta)] }
    # where q' is new inventory after fill
    
    # Since mid moves as a random walk, and the problem is translation-invariant in mid,
    # the optimal depths don't depend on mid directly, only on t, q, and parameters.
    
    # The key insight: for each (t, q), we need to find optimal bid and ask depths.
    # The bid fill earns depth d_b and changes q -> q+1
    # The ask fill earns depth d_a and changes q -> q-1
    # Both fills happen independently with probabilities P_b and P_a
    
    # Expected value if we quote bid depth d_b and ask depth d_a at state (t, q):
    # E = P_b * (d_b + E[V_{t+1}(q+1)]) + P_a * (d_a + E[V_{t+1}(q-1)]) 
    #     + (1-P_b)(1-P_a) * E[V_{t+1}(q)]
    #     - (gamma/2) * sigma^2 * q^2 * dt
    #
    # Where P_b = 1-exp(-A*exp(-k*d_b)*dt), P_a = 1-exp(-A*exp(-k*d_a)*dt)
    #
    # But we also need to account for the mid movement. Since the mid moves by 
    # sigma*sqrt(dt)*N(0,1), and the value function is defined in terms of the
    # current mid being the reference, the expected value of V_{t+1} given the
    # current state involves integrating over the mid movement.
    #
    # However, since the problem is translation-invariant (the reward depends only
    # on depths, not absolute mid), V_{t+1}(q) is independent of mid. So:
    # E[V_{t+1}(q)] = V_{t+1}(q) (no expectation needed for the mid term)
    
    # Boundary conditions:
    # At q = Q: no bid fill possible
    # At q = -Q: no ask fill possible
    
    # For the optimization, we can use a grid search over depths.
    # The optimal depth is likely in [0, 5/k] for reasonable parameters.
    
    max_depth = 5.0 / k
    n_grid = 80
    depths_grid = np.linspace(0, max_depth, n_grid)
    
    # Precompute P for the grid
    P_grid = 1.0 - np.exp(-A * np.exp(-k * depths_grid) * dt)
    
    # For each time step from N-1 down to 0
    for t in range(N - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 15:
            # Fallback: use 1/k for all
            for qi in range(n_q):
                q = qi - Q
                opt_bid[t, qi] = 1.0 / k if q < Q else np.inf
                opt_ask[t, qi] = 1.0 / k if q > -Q else np.inf
            return opt_bid, opt_ask, V
        
        # Inventory holding cost for this step
        inv_cost = (gamma / 2.0) * sigma**2 * (np.arange(-Q, Q + 1)**2) * dt
        
        for qi in range(n_q):
            q = qi - Q  # actual inventory
            
            # Get next step values
            V_next_q = V[t + 1, qi]
            V_next_q_plus = V[t + 1, min(qi + 1, n_q - 1)] if q < Q else 0.0
            V_next_q_minus = V[t + 1, max(qi - 1, 0)] if q > -Q else 0.0
            
            # Can we quote bid? Only if q < Q
            can_bid = (q < Q)
            # Can we quote ask? Only if q > -Q
            can_ask = (q > -Q)
            
            # Optimize bid depth (if allowed)
            if can_bid:
                # Expected value for each bid depth
                # P_b = P_grid, reward = P_b * (d + V_next_q_plus) 
                # The (1-P_b) term for no bid fill is common to all bid choices
                # so we can ignore it for optimization
                # EV_bid(d) = P(d) * (d + V_next_q_plus)
                ev_bid = P_grid * (depths_grid + V_next_q_plus)
                best_idx = np.argmax(ev_bid)
                opt_bid[t, qi] = depths_grid[best_idx]
            else:
                opt_bid[t, qi] = np.inf
            
            # Optimize ask depth (if allowed)
            if can_ask:
                ev_ask = P_grid * (depths_grid + V_next_q_minus)
                best_idx = np.argmax(ev_ask)
                opt_ask[t, qi] = depths_grid[best_idx]
            else:
                opt_ask[t, qi] = np.inf
            
            # Compute V[t, q] = max over (d_b, d_a) of the expected value
            # Since bid and ask are optimized independently (separable),
            # V[t,q] = max_d_b [P_b(d_b)*(d_b + V_next_q_plus) + (1-P_b(d_b))*V_next_q]
            #         + max_d_a [P_a(d_a)*(d_a + V_next_q_minus) + (1-P_a(d_a))*V_next_q]
            #         - inv_cost[q]
            #         + (1-P_b)(1-P_a) term is included in both
        
            if can_bid:
                best_bid_idx = np.argmax(P_grid * (depths_grid + V_next_q_plus) + (1 - P_grid) * V_next_q)
                bid_contrib = P_grid[best_bid_idx] * (depths_grid[best_bid_idx] + V_next_q_plus) + \
                              (1 - P_grid[best_bid_idx]) * V_next_q
            else:
                bid_contrib = V_next_q
            
            if can_ask:
                best_ask_idx = np.argmax(P_grid * (depths_grid + V_next_q_minus) + (1 - P_grid) * V_next_q)
                ask_contrib = P_grid[best_ask_idx] * (depths_grid[best_ask_idx] + V_next_q_minus) + \
                              (1 - P_grid[best_ask_idx]) * V_next_q
            else:
                ask_contrib = V_next_q
            
            # But wait, the above double-counts V_next_q. Let me redo:
            # EV = P_b * (d_b + V_{q+1}) + P_a * (d_a + V_{q-1}) + (1-P_b)(1-P_a) * V_q - inv_cost
            #    = P_b*d_b + P_b*V_{q+1} + P_a*d_a + P_a*V_{q-1} + (1-P_b)(1-P_a)*V_q - inv_cost
            # 
            # This is not separable in P_b and P_a due to (1-P_b)(1-P_a)
            # Let me use the full 2D grid for correctness
            
            # Actually, let's just do a 2D grid search for accuracy
            if can_bid and can_ask:
                # EV[i,j] = P[i]*(d[i]+V_{q+1}) + P[j]*(d[j]+V_{q-1}) + (1-P[i])(1-P[j])*V_q
                ev_matrix = np.outer(P_grid, depths_grid + V_next_q_plus) + \
                           np.outer(P_grid, depths_grid + V_next_q_minus).T + \
                           np.outer(1 - P_grid, 1 - P_grid) * V_next_q
                
                # Wait, that's wrong. Let me be more careful:
                # EV(i,j) = P_i * (d_i + V_{q+1}) + P_j * (d_j + V_{q-1}) + (1-P_i)(1-P_j)*V_q
                # where i indexes bid, j indexes ask
                
                ev_matrix = np.zeros((n_grid, n_grid))
                for i in range(n_grid):
                    Pi = P_grid[i]
                    di = depths_grid[i]
                    row = Pi * (di + V_next_q_plus) + (1 - Pi) * V_next_q
                    for j in range(n_grid):
                        Pj = P_grid[j]
                        dj = depths_grid[j]
                        ev_matrix[i, j] = Pi * (di + V_next_q_plus) + Pj * (dj + V_next_q_minus) + \
                                         (1 - Pi) * (1 - Pj) * V_next_q
                
                best = np.unravel_index(np.argmax(ev_matrix), ev_matrix.shape)
                V[t, qi] = ev_matrix[best] - inv_cost[qi]
                opt_bid[t, qi] = depths_grid[best[0]]
                opt_ask[t, qi] = depths_grid[best[1]]
            
            elif can_bid:
                # Only bid
                ev_bid = P_grid * (depths_grid + V_next_q_plus) + (1 - P_grid) * V_next_q
                best_idx = np.argmax(ev_bid)
                V[t, qi] = ev_bid[best_idx] - inv_cost[qi]
                opt_bid[t, qi] = depths_grid[best_idx]
                opt_ask[t, qi] = np.inf
            
            elif can_ask:
                # Only ask
                ev_ask = P_grid * (depths_grid + V_next_q_minus) + (1 - P_grid) * V_next_q
                best_idx = np.argmax(ev_ask)
                V[t, qi] = ev_ask[best_idx] - inv_cost[qi]
                opt_bid[t, qi] = np.inf
                opt_ask[t, qi] = depths_grid[best_idx]
            else:
                V[t, qi] = -inv_cost[qi]
                opt_bid[t, qi] = np.inf
                opt_ask[t, qi] = np.inf
    
    return opt_bid, opt_ask, V


def _get_policy(params):
    """Get or compute the optimal policy for the given regime."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        if time.time() > _DEADLINE - 20:
            return None
        opt_bid, opt_ask, V = _compute_optimal_policy(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
        _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """Quote using precomputed optimal policy."""
    n = len(inventory)
    k = params['k']
    Q = params['Q']
    T = params['T']
    steps = params['steps']
    dt = T / steps
    
    # Try to get the optimal policy
    policy = _get_policy(params)
    
    if policy is None:
        # Fallback: seed strategy
        depth = np.full(n, 1.0 / k)
        return depth, depth.copy()
    
    opt_bid, opt_ask = policy
    
    # Determine step index
    step_idx = int(round(t / dt))
    step_idx = np.clip(step_idx, 0, steps - 1)
    
    # Map inventory to index: q in [-Q, Q] -> idx in [0, 2Q]
    q_idx = np.array(inventory, dtype=int) + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Get depths for this step and inventory
    bid_depths = opt_bid[step_idx, q_idx].copy()
    ask_depths = opt_ask[step_idx, q_idx].copy()
    
    # Enforce boundaries
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    # Ensure valid values: no values below -3/k, no NaN, no -inf
    min_valid = -3.0 / k
    bid_depths = np.where((bid_depths < min_valid) | np.isnan(bid_depths) | (bid_depths == -np.inf), np.inf, bid_depths)
    ask_depths = np.where((ask_depths < min_valid) | np.isnan(ask_depths) | (ask_depths == -np.inf), np.inf, ask_depths)
    
    # Ensure shape is correct
    assert bid_depths.shape == (n,), f"bid_depths shape {bid_depths.shape} != ({n},)"
    assert ask_depths.shape == (n,), f"ask_depths shape {ask_depths.shape} != ({n},)"
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
