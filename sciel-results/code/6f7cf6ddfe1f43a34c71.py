"""Optimal market making with inventory risk using vectorized HJB PDE solver."""
# EVOLVE-BLOCK-START
import numpy as np

# Global cache for pre-computed policies
_POLICY_CACHE = {}

def _get_depth_grid(k, n_depths=1000):
    """
    Generate a log-spaced depth grid.
    We need to cover small depths (high volume) and large depths (low volume).
    d_min should be small enough to capture the optimum if C is large.
    d_max should be large enough to cover the "no quote" equivalent or very low probability.
    """
    d_min = 1e-4
    d_max = 5.0 / k
    # Log space is good for exponential decay
    return np.geomspace(d_min, d_max, n_depths)

def _solve_policy(params, n_time_steps=1000):
    """
    Solve for the optimal bid and ask depth policy for all t and q.
    Returns:
        bid_policy: array of shape (n_time_steps, 2*Q+1)
        ask_policy: array of shape (n_time_steps, 2*Q+1)
        V: array of shape (n_time_steps+1, 2*Q+1)
    """
    T = params['T']
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    
    dt = T / n_time_steps
    n_q = 2 * Q + 1
    q_grid = np.arange(-Q, Q + 1)
    
    # Depth grid
    depths = _get_depth_grid(k, n_depths=1000)
    
    # Pre-compute hit probabilities and rewards for all depths
    # p(d) = 1 - exp(-A * exp(-k * d) * dt)
    # lambda(d) = A * exp(-k * d)
    lambdas = A * np.exp(-k * depths)
    p_hit = 1.0 - np.exp(-lambdas * dt)
    
    # For bid: term is p(d) * (d + C_b)
    # For ask: term is p(d) * (-d + C_a)
    # We need to maximize these.
    
    # V[t, q]
    V = np.zeros((n_time_steps + 1, n_q))
    bid_policy = np.zeros((n_time_steps, n_q))
    ask_policy = np.zeros((n_time_steps, n_q))
    
    # Backward induction
    for t_idx in range(n_time_steps - 1, -1, -1):
        V_next = V[t_idx + 1, :]
        
        # Compute differences C_b and C_a for all q
        # C_b[q] = V_next[q+1] - V_next[q]
        # C_a[q] = V_next[q-1] - V_next[q]
        
        # Initialize C_b, C_a with 0 to handle boundaries
        C_b = np.zeros(n_q)
        C_a = np.zeros(n_q)
        
        # For q < Q, bid is possible. q index 0 to n_q-2 (q=-Q to Q-1)
        # q_idx corresponds to q = q_idx - Q
        # If q < Q, i.e., q_idx < n_q - 1
        C_b[:n_q-1] = V_next[1:n_q] - V_next[0:n_q-1]
        
        # If q > -Q, ask is possible. q index 1 to n_q-1 (q=-Q+1 to Q)
        C_a[1:n_q] = V_next[0:n_q-1] - V_next[1:n_q]
        
        # Now, for each q, find optimal d_b and d_a
        # We can vectorize this by creating a matrix of values: (n_q, n_depths)
        
        # Bid values: (n_q, n_depths)
        # term_b[i, j] = p_hit[j] * (depths[j] + C_b[i])
        # Note: C_b is (n_q,), depths is (n_depths,)
        # Broadcasting: C_b[:, np.newaxis] + depths[np.newaxis, :]
        bid_values = p_hit[np.newaxis, :] * (depths[np.newaxis, :] + C_b[:, np.newaxis])
        
        # Ask values: (n_q, n_depths)
        # term_a[i, j] = p_hit[j] * (-depths[j] + C_a[i])
        ask_values = p_hit[np.newaxis, :] * (-depths[np.newaxis, :] + C_a[:, np.newaxis])
        
        # Find argmax for each q
        # For q=Q (last index), bid is not allowed. We set C_b to 0, but we should mask it.
        # Actually, if q=Q, we don't quote bid. The value contribution is 0.
        # Our C_b for q=Q is 0, so bid_values will be p(d)*d. This is positive, so it would pick a depth.
        # We need to enforce that for q=Q, bid_depth is inf (or effectively no quote).
        # Similarly for q=-Q, ask_depth is inf.
        
        # Let's compute the optimal depths first
        optimal_bid_idx = np.argmax(bid_values, axis=1)
        optimal_ask_idx = np.argmax(ask_values, axis=1)
        
        optimal_bid_depths = depths[optimal_bid_idx]
        optimal_ask_depths = depths[optimal_ask_idx]
        
        # Mask boundaries
        # q = Q is index n_q - 1
        optimal_bid_depths[n_q - 1] = np.inf
        # q = -Q is index 0
        optimal_ask_depths[0] = np.inf
        
        # Store policies
        bid_policy[t_idx, :] = optimal_bid_depths
        ask_policy[t_idx, :] = optimal_ask_depths
        
        # Update V
        # V[t, q] = V_next[q] + dt * ( ... ) ? 
        # No, the Bellman equation is:
        # V[t, q] = max ( E[R] )
        # E[R] = p_b * (d_b + V_next[q+1]) + p_a * (-d_a + V_next[q-1]) + (1-p_b-p_a+p_b p_a) * V_next[q] - (gamma/2) sigma^2 q^2 dt
        # Note: The probabilities are not mutually exclusive in the exact sense if we allow both fills?
        # The problem statement says "bid hit with prob ..., ask likewise, independently".
        # So 4 outcomes:
        # 1. Bid only: p_b (1-p_a) -> Reward d_b, Inv q+1
        # 2. Ask only: (1-p_b) p_a -> Reward -d_a, Inv q-1
        # 3. Both: p_b p_a -> Reward d_b - d_a, Inv q (net 0 change? No, +1-1=0)
        # 4. Neither: (1-p_b)(1-p_a) -> Reward 0, Inv q
        
        # Wait, the problem says "A fill earns its depth and moves inventory one unit".
        # If both fill, inventory moves +1 and -1, so net 0. Reward d_b - d_a.
        
        # Let's compute the expected value properly for the optimal depths found.
        
        p_b_opt = p_hit[optimal_bid_idx]
        p_a_opt = p_hit[optimal_ask_idx]
        
        # Handle inf depths (p=0)
        p_b_opt[~np.isfinite(optimal_bid_depths)] = 0.0
        p_a_opt[~np.isfinite(optimal_ask_depths)] = 0.0
        
        # d_b_opt, d_a_opt
        d_b_opt = optimal_bid_depths
        d_a_opt = optimal_ask_depths
        d_b_opt[~np.isfinite(d_b_opt)] = 0.0
        d_a_opt[~np.isfinite(d_a_opt)] = 0.0
        
        # Expected Reward
        # R_bid_only = p_b (1-p_a) * d_b
        # R_ask_only = (1-p_b) p_a * (-d_a)
        # R_both = p_b p_a * (d_b - d_a)
        # R_neither = 0
        
        # Expected Next Value
        # V_next[q+1] * p_b (1-p_a)  (if q < Q)
        # V_next[q-1] * (1-p_b) p_a  (if q > -Q)
        # V_next[q] * (p_b p_a + (1-p_b)(1-p_a))
        
        # Let's build vectors for V_next[q+1] and V_next[q-1]
        V_next_q_plus = np.zeros(n_q)
        V_next_q_minus = np.zeros(n_q)
        
        V_next_q_plus[:n_q-1] = V_next[1:n_q]
        V_next_q_plus[n_q-1] = V_next[n_q-1] # Dummy, will be masked
        
        V_next_q_minus[1:n_q] = V_next[0:n_q-1]
        V_next_q_minus[0] = V_next[0] # Dummy, will be masked
        
        # Probabilities
        P_bid_only = p_b_opt * (1.0 - p_a_opt)
        P_ask_only = (1.0 - p_b_opt) * p_a_opt
        P_both = p_b_opt * p_a_opt
        P_neither = (1.0 - p_b_opt) * (1.0 - p_a_opt)
        
        # Mask for valid bid (q < Q)
        valid_bid = np.arange(n_q) < n_q - 1
        # Mask for valid ask (q > -Q)
        valid_ask = np.arange(n_q) > 0
        
        # If not valid bid, the bid fill doesn't happen.
        # So if q=Q, P_bid_only=0, P_both=0.
        P_bid_only[~valid_bid] = 0.0
        P_both[~valid_bid] = 0.0
        
        # If not valid ask, the ask fill doesn't happen.
        P_ask_only[~valid_ask] = 0.0
        P_both[~valid_ask] = 0.0
        
        # Recalculate P_neither to sum to 1?
        # P_neither = 1 - (P_bid_only + P_ask_only + P_both)
        P_neither = 1.0 - (P_bid_only + P_ask_only + P_both)
        
        # Expected Value
        EV = (P_bid_only * (d_b_opt + V_next_q_plus) + 
              P_ask_only * (-d_a_opt + V_next_q_minus) + 
              P_both * (d_b_opt - d_a_opt + V_next) + # Inventory returns to q
              P_neither * V_next)
        
        # Subtract holding cost
        cost = (gamma / 2.0) * sigma * sigma * q_grid * q_grid * dt
        V[t_idx, :] = EV - cost
    
    return bid_policy, ask_policy, V

def _precompute_policies():
    global _POLICY_CACHE
    regimes = [
        {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.05, 'T': 1.0, 'steps': 1000, 'Q': 30},
        {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.5, 'T': 1.0, 'steps': 1000, 'Q': 30},
        {'sigma': 1.5, 'A': 50.0, 'k': 0.75, 'gamma': 0.2, 'T': 1.0, 'steps': 1000, 'Q': 30},
    ]
    for reg in regimes:
        key = (reg['sigma'], reg['A'], reg['k'], reg['gamma'])
        if key not in _POLICY_CACHE:
            bid_p, ask_p, V = _solve_policy(reg, n_time_steps=1000)
            _POLICY_CACHE[key] = (bid_p, ask_p, V)

def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths using the pre-computed policy.
    """
    global _POLICY_CACHE
    
    if not _POLICY_CACHE:
        _precompute_policies()
        
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma)
    
    if key not in _POLICY_CACHE:
        bid_p, ask_p, V = _solve_policy(params, n_time_steps=1000)
        _POLICY_CACHE[key] = (bid_p, ask_p, V)
    else:
        bid_p, ask_p, V = _POLICY_CACHE[key]
        
    # Determine time index
    # t is in [0, T]. steps is 1000.
    # The policy is computed for n_time_steps=1000.
    # t_idx = int(round(t / (T / 1000)))
    dt = T / 1000
    t_idx = int(round(t / dt))
    t_idx = max(0, min(t_idx, 999))
    
    # Get policies for this time step
    bid_depths_t = bid_p[t_idx, :]
    ask_depths_t = ask_p[t_idx, :]
    
    # Map inventory to index
    inventory_int = np.round(inventory).astype(int)
    inventory_int = np.clip(inventory_int, -Q, Q)
    indices = inventory_int + Q
    
    bid_depths = bid_depths_t[indices]
    ask_depths = ask_depths_t[indices]
    
    # Ensure valid outputs
    # Values below -3/k, -inf, NaN are invalid.
    # Our depths are >= 0.
    # If depth is inf, it means no quote.
    
    # Clip small negative values if any (shouldn't happen)
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    # Replace NaN with inf
    bid_depths[~np.isfinite(bid_depths)] = np.inf
    ask_depths[~np.isfinite(ask_depths)] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
