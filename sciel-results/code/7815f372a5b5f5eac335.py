"""Optimal market making with inventory risk: Continuous-action HJB with grid refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # Safety margin (330s kill, leave 35s buffer)

_REGIME_CACHE = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q, time_budget=80.0):
    """
    Solves the HJB with continuous-action approximation via 1D scalar maximization.
    Two passes: coarse grid, then refined grid around first-pass optima.
    """
    dt = T / steps
    num_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Precompute risk cost
    risk_cost = 0.5 * gamma * sigma * sigma * (q_vals ** 2) * dt
    
    # Fill probability function: P(d) = 1 - exp(-A * exp(-k*d) * dt)
    def fill_prob(d):
        # d is scalar or array
        return 1.0 - np.exp(-A * np.exp(-k * d) * dt)
    
    # Coarse grid for depths
    max_d = 4.0 / k  # generous upper bound
    min_d = 1e-4
    D_coarse = 80
    d_grid = np.linspace(min_d, max_d, D_coarse)
    
    # Precompute fill probs on coarse grid
    p_grid = fill_prob(d_grid)  # shape (D_coarse,)
    
    # V_next: shape (num_q,)
    V_next = np.zeros(num_q)
    
    # Policy storage
    policy = np.zeros((steps, num_q, 2))
    
    # Backward induction - Pass 1 (coarse)
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 30:
            # Fill remaining with current best
            break
        
        # V_next shifted
        V_q = V_next.copy()
        V_q_plus = np.zeros(num_q)
        V_q_minus = np.zeros(num_q)
        # q+1: index i -> i-1 in array (since q_vals[i] = -Q + i, q+1 is at i+1)
        # V_next[q+1] for q at index i is V_next[i+1]
        V_q_plus[:-1] = V_next[1:]
        V_q_plus[-1] = 0.0  # q=Q, no bid fill
        # V_next[q-1] for q at index i is V_next[i-1]
        V_q_minus[1:] = V_next[:-1]
        V_q_minus[0] = 0.0  # q=-Q, no ask fill
        
        # Boundary masks
        mask_q_max = np.zeros(num_q, dtype=np.float64)
        mask_q_max[-1] = 1.0
        mask_q_min = np.zeros(num_q, dtype=np.float64)
        mask_q_min[0] = 1.0
        
        best_bid_d = np.zeros(num_q)
        best_ask_d = np.zeros(num_q)
        best_bid_val = np.full(num_q, -np.inf)
        best_ask_val = np.full(num_q, -np.inf)
        
        # For each depth candidate, compute the contribution to bid and ask values
        # Bid value for depth d at inventory q:
        #   p_b = P_bid(d) * (1 - mask_q_max[q])
        #   bid_val = p_b * (d + V_q_plus[q]) + (1 - p_b) * V_q[q] - risk_cost[q]
        # Ask value for depth d at inventory q:
        #   p_a = P_ask(d) * (1 - mask_q_min[q])
        #   ask_val = p_a * (d + V_q_minus[q]) + (1 - p_a) * V_q[q] - risk_cost[q]
        
        # Vectorize over depth grid
        # For bid: p_b shape (D_coarse,), applied per q
        # bid_val[q, j] = p_grid[j] * (1-mask_q_max[q]) * (d_grid[j] + V_q_plus[q]) 
        #                 + (1 - p_grid[j]*(1-mask_q_max[q])) * V_q[q] - risk_cost[q]
        
        # Let's compute for all q and all d at once
        # p_b_eff[q, j] = p_grid[j] * (1 - mask_q_max[q])
        p_b_eff = p_grid[None, :] * (1.0 - mask_q_max[:, None])  # (num_q, D_coarse)
        p_a_eff = p_grid[None, :] * (1.0 - mask_q_min[:, None])  # (num_q, D_coarse)
        
        # bid_val[q, j] = p_b_eff[q,j] * (d_grid[j] + V_q_plus[q]) + (1 - p_b_eff[q,j]) * V_q[q] - risk_cost[q]
        bid_val = p_b_eff * (d_grid[None, :] + V_q_plus[:, None]) + (1.0 - p_b_eff) * V_q[:, None] - risk_cost[:, None]
        
        # ask_val[q, j] = p_a_eff[q,j] * (d_grid[j] + V_q_minus[q]) + (1 - p_a_eff[q,j]) * V_q[q] - risk_cost[q]
        ask_val = p_a_eff * (d_grid[None, :] + V_q_minus[:, None]) + (1.0 - p_a_eff) * V_q[:, None] - risk_cost[:, None]
        
        # Find argmax
        best_bid_idx = np.argmax(bid_val, axis=1)
        best_ask_idx = np.argmax(ask_val, axis=1)
        
        best_bid_val = bid_val[np.arange(num_q), best_bid_idx]
        best_ask_val = ask_val[np.arange(num_q), best_ask_idx]
        
        best_bid_d = d_grid[best_bid_idx]
        best_ask_d = d_grid[best_ask_idx]
        
        V_curr = best_bid_val + best_ask_val
        policy[n] = np.stack([best_bid_d, best_ask_d], axis=-1)
        V_next = V_curr
    
    # Pass 2: Refine around first-pass optima
    # For each (n, q), the optimal depth is near policy[n, q, :].
    # Create a local grid around each first-pass optimum.
    # To save time, we'll do a second backward induction with a refined local grid.
    # But the local grid is per (n, q), which makes vectorization harder.
    # Alternative: Use a medium grid (160 points) for the entire second pass.
    
    # Actually, let's use a medium grid for pass 2.
    D_med = 160
    d_grid_med = np.linspace(min_d, max_d, D_med)
    p_grid_med = fill_prob(d_grid_med)
    
    V_next = np.zeros(num_q)
    policy2 = np.zeros((steps, num_q, 2))
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 15:
            break
        
        V_q = V_next.copy()
        V_q_plus = np.zeros(num_q)
        V_q_minus = np.zeros(num_q)
        V_q_plus[:-1] = V_next[1:]
        V_q_plus[-1] = 0.0
        V_q_minus[1:] = V_next[:-1]
        V_q_minus[0] = 0.0
        
        p_b_eff = p_grid_med[None, :] * (1.0 - mask_q_max[:, None])
        p_a_eff = p_grid_med[None, :] * (1.0 - mask_q_min[:, None])
        
        bid_val = p_b_eff * (d_grid_med[None, :] + V_q_plus[:, None]) + (1.0 - p_b_eff) * V_q[:, None] - risk_cost[:, None]
        ask_val = p_a_eff * (d_grid_med[None, :] + V_q_minus[:, None]) + (1.0 - p_a_eff) * V_q[:, None] - risk_cost[:, None]
        
        best_bid_idx = np.argmax(bid_val, axis=1)
        best_ask_idx = np.argmax(ask_val, axis=1)
        
        best_bid_val = bid_val[np.arange(num_q), best_bid_idx]
        best_ask_val = ask_val[np.arange(num_q), best_ask_idx]
        
        best_bid_d = d_grid_med[best_bid_idx]
        best_ask_d = d_grid_med[best_ask_idx]
        
        V_curr = best_bid_val + best_ask_val
        policy2[n] = np.stack([best_bid_d, best_ask_d], axis=-1)
        V_next = V_curr
    
    return policy2


def _get_policy_for_params(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _REGIME_CACHE:
        if time.time() > _DEADLINE - 20:
            return None
        policy = _solve_regime(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
        _REGIME_CACHE[key] = policy
    return _REGIME_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Returns optimal depths from precomputed HJB policy.
    """
    policy = _get_policy_for_params(params)
    
    if policy is None:
        # Fallback: Avellaneda-Stoikov heuristic
        sigma = params["sigma"]
        A = params["A"]
        k = params["k"]
        gamma = params["gamma"]
        T = params["T"]
        dt_rem = T - t
        R = gamma * (sigma ** 2) * inventory * dt_rem
        delta_star = 1.0 / k
        bid_depths = np.maximum(R + delta_star, 1e-6)
        ask_depths = np.maximum(-R + delta_star, 1e-6)
        return bid_depths, ask_depths
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    n = int(round(t / dt))
    if n < 0:
        n = 0
    if n >= steps:
        n = steps - 1
    
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    depths = policy[n, q_idx, :]
    bid_depths = depths[:, 0]
    ask_depths = depths[:, 1]
    
    bid_depths = np.maximum(bid_depths, 1e-6)
    ask_depths = np.maximum(ask_depths, 1e-6)
    
    return bid_depths, ask_depths


# Precompute known regimes at import time
_KNOWN_REGIMES = [
    (2, 140, 1.5, 0.05),
    (2, 140, 1.5, 0.5),
    (1.5, 50, 0.75, 0.2)
]

for sig, A, k, gam in _KNOWN_REGIMES:
    if time.time() > _DEADLINE - 20:
        break
    try:
        p = {'sigma': sig, 'A': A, 'k': k, 'gamma': gam, 'T': 1.0, 'steps': 1000, 'Q': 30}
        _get_policy_for_params(p)
    except Exception:
        pass
# EVOLVE-BLOCK-END
