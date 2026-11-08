"""Optimal market making with inventory risk using vectorized HJB PDE solver."""
# EVOLVE-BLOCK-START
import numpy as np

# Global cache for pre-computed policies
_POLICY_CACHE = {}

def _get_depth_grid(k, n_depths=5000):
    """
    Generate a log-spaced depth grid with high resolution.
    We need to cover small depths (high volume) and large depths (low volume).
    d_min should be small enough to capture the optimum if C is large.
    d_max should be large enough to cover the "no quote" equivalent or very low probability.
    """
    d_min = 1e-5
    d_max = 8.0 / k
    # Log space is good for exponential decay
    return np.geomspace(d_min, d_max, n_depths)

def _solve_policy(params, n_time_steps=2000):
    """
    Solve for the optimal bid and ask depth policy for all t and q using
    a high-resolution depth grid and fine time discretization.
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
    
    # High-resolution depth grid
    depths = _get_depth_grid(k, n_depths=5000)
    n_depths = len(depths)
    
    # Pre-compute hit probabilities for all depths
    # p(d) = 1 - exp(-A * exp(-k * d) * dt)
    # lambda(d) = A * exp(-k * d)
    lambdas = A * np.exp(-k * depths)
    p_hit = 1.0 - np.exp(-lambdas * dt)
    
    # V[t, q]
    V = np.zeros((n_time_steps + 1, n_q))
    bid_policy = np.zeros((n_time_steps, n_q))
    ask_policy = np.zeros((n_time_steps, n_q))
    
    # Pre-compute valid masks
    valid_bid = np.arange(n_q) < n_q - 1  # q < Q
    valid_ask = np.arange(n_q) > 0         # q > -Q
    
    # Backward induction
    for t_idx in range(n_time_steps - 1, -1, -1):
        V_next = V[t_idx + 1, :]
        
        # Compute marginal values C_b and C_a for all q
        # C_b[q] = V_next[q+1] - V_next[q]  (value of one more unit of inventory)
        # C_a[q] = V_next[q-1] - V_next[q]  (value of one less unit of inventory)
        
        C_b = np.zeros(n_q)
        C_a = np.zeros(n_q)
        
        # For q < Q (indices 0 to n_q-2)
        C_b[:n_q-1] = V_next[1:n_q] - V_next[0:n_q-1]
        
        # For q > -Q (indices 1 to n_q-1)
        C_a[1:n_q] = V_next[0:n_q-1] - V_next[1:n_q]
        
        # Vectorized computation of optimal bid depths
        # bid_values[i, j] = p_hit[j] * (depths[j] + C_b[i])
        # Shape: (n_q, n_depths)
        bid_values = p_hit[np.newaxis, :] * (depths[np.newaxis, :] + C_b[:, np.newaxis])
        
        # Vectorized computation of optimal ask depths
        # ask_values[i, j] = p_hit[j] * (-depths[j] + C_a[i])
        # Shape: (n_q, n_depths)
        ask_values = p_hit[np.newaxis, :] * (-depths[np.newaxis, :] + C_a[:, np.newaxis])
        
        # Find argmax for each inventory level
        optimal_bid_idx = np.argmax(bid_values, axis=1)
        optimal_ask_idx = np.argmax(ask_values, axis=1)
        
        optimal_bid_depths = depths[optimal_bid_idx]
        optimal_ask_depths = depths[optimal_ask_idx]
        
        # Enforce boundary conditions: no bid at q=Q, no ask at q=-Q
        optimal_bid_depths[~valid_bid] = np.inf
        optimal_ask_depths[~valid_ask] = np.inf
        
        # Store policies
        bid_policy[t_idx, :] = optimal_bid_depths
        ask_policy[t_idx, :] = optimal_ask_depths
        
        # Compute probabilities for optimal depths
        p_b_opt = p_hit[optimal_bid_idx]
        p_a_opt = p_hit[optimal_ask_idx]
        
        # Handle inf depths (no quote, p=0)
        p_b_opt[~valid_bid] = 0.0
        p_a_opt[~valid_ask] = 0.0
        
        # Optimal depths (0 where no quote)
        d_b_opt = optimal_bid_depths.copy()
        d_a_opt = optimal_ask_depths.copy()
        d_b_opt[~valid_bid] = 0.0
        d_a_opt[~valid_ask] = 0.0
        
        # Build V_next[q+1] and V_next[q-1] vectors
        V_next_q_plus = np.zeros(n_q)
        V_next_q_minus = np.zeros(n_q)
        
        V_next_q_plus[:n_q-1] = V_next[1:n_q]
        V_next_q_plus[n_q-1] = V_next[n_q-1]
        
        V_next_q_minus[1:n_q] = V_next[0:n_q-1]
        V_next_q_minus[0] = V_next[0]
        
        # Compute joint probabilities (bid and ask fills are independent)
        P_bid_only = p_b_opt * (1.0 - p_a_opt)
        P_ask_only = (1.0 - p_b_opt) * p_a_opt
        P_both = p_b_opt * p_a_opt
        
        # Mask invalid transitions
        P_bid_only[~valid_bid] = 0.0
        P_both[~valid_bid] = 0.0
        P_ask_only[~valid_ask] = 0.0
        P_both[~valid_ask] = 0.0
        
        P_neither = 1.0 - (P_bid_only + P_ask_only + P_both)
        
        # Expected value of next state
        EV = (P_bid_only * (d_b_opt + V_next_q_plus) + 
              P_ask_only * (-d_a_opt + V_next_q_minus) + 
              P_both * (d_b_opt - d_a_opt + V_next) +
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
            bid_p, ask_p, V = _solve_policy(reg, n_time_steps=2000)
            _POLICY_CACHE[key] = (bid_p, ask_p, V)

def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths using the pre-computed policy.
    Uses interpolation between time steps for smoother policy.
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
        bid_p, ask_p, V = _solve_policy(params, n_time_steps=2000)
        _POLICY_CACHE[key] = (bid_p, ask_p, V)
    else:
        bid_p, ask_p, V = _POLICY_CACHE[key]
        
    n_time_steps = bid_p.shape[0]
    
    # Determine time index with interpolation
    # t is in [0, T]. The policy is computed for n_time_steps=2000.
    # Map t to the solver's time grid
    solver_dt = T / n_time_steps
    t_cont = t / solver_dt
    t_idx = int(np.floor(t_cont))
    t_idx_next = min(t_idx + 1, n_time_steps - 1)
    t_frac = t_cont - t_idx
    
    # Get policies for this time step (interpolate)
    bid_depths_t = bid_p[t_idx, :] * (1 - t_frac) + bid_p[t_idx_next, :] * t_frac
    ask_depths_t = ask_p[t_idx, :] * (1 - t_frac) + ask_p[t_idx_next, :] * t_frac
    
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
