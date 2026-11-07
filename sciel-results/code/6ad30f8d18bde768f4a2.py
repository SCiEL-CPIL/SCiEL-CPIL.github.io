"""Optimal market making with inventory risk using HJB PDE solver."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys

# Pre-computed HJB solution
_HJB_SOLUTION = None
_HJB_PARAMS = None
_HJB_GRID_T = None
_HJB_GRID_Q = None
_HJB_GRID_DEPTH = None

def _solve_hjb(params, n_time_steps=50, n_inventory=61, n_depths=200):
    """
    Solve the HJB PDE for optimal market making.
    
    The state is (t, q) where t is time and q is inventory.
    The action is the quote depth (same for bid and ask for simplicity in HJB,
    but we can extend to asymmetric quotes).
    
    Actually, for this problem, the optimal policy may quote different depths
    for bid and ask depending on inventory. Let's think more carefully.
    
    In the continuous-time limit, the HJB equation is:
    V_t + sup_a [ r(t,q,a) - (gamma/2) sigma^2 q^2 ] + (sigma^2/2) V_qq = 0
    
    where r(t,q,a) is the expected instantaneous reward from quoting with action a.
    
    For a symmetric quote with depth d on both sides:
    - Bid hit rate: A * exp(-k*d)
    - Ask hit rate: A * exp(-k*d)
    - Bid fill earns +d, inventory increases by 1
    - Ask fill earns -d, inventory decreases by 1
    
    So the drift of inventory is: A*exp(-k*d) * 1 - A*exp(-k*d) * 1 = 0 (symmetric)
    But the reward is: A*exp(-k*d)*d - A*exp(-k*d)*d = 0
    
    Wait, that's not right. Let me reconsider.
    
    If we quote bid at depth d_b below mid and ask at depth d_a above mid:
    - Bid hit: prob ~ A*exp(-k*d_b)*dt, earn +d_b, inventory q -> q+1
    - Ask hit: prob ~ A*exp(-k*d_a)*dt, earn -d_a, inventory q -> q-1
    
    So the expected instantaneous reward is:
    A*exp(-k*d_b)*d_b - A*exp(-k*d_a)*d_a
    
    And the inventory drift is:
    A*exp(-k*d_b) - A*exp(-k*d_a)
    
    The HJB in continuous time:
    -V_t + sup_{d_b, d_a} [ A*exp(-k*d_b)*d_b - A*exp(-k*d_a)*d_a - (gamma/2)*sigma^2*q^2 
                              + (A*exp(-k*d_b) - A*exp(-k*d_a))*V_q
                              + (sigma^2/2)*V_qq ] = 0
    
    This is a 2D optimization over (d_b, d_a) at each (t, q).
    
    For simplicity and to match the discrete model better, let's use a semi-discrete approach:
    We discretize time backwards and inventory, and at each step we optimize over a grid of depths.
    """
    T = params['T']
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    
    # Time grid (backward from T to 0)
    dt = T / n_time_steps
    t_grid = np.linspace(0, T, n_time_steps + 1)
    
    # Inventory grid
    q_grid = np.arange(-Q, Q + 1)
    n_q = len(q_grid)
    
    # Depth grid (log-spaced for better resolution near 0)
    d_min = 1e-4
    d_max = 5.0 / k  # reasonable max depth
    depth_grid = np.geomspace(d_min, d_max, n_depths)
    
    # V[t_idx, q_idx] = value function at time t_grid[t_idx], inventory q_grid[q_idx]
    # We solve backwards from t = T to t = 0
    V = np.zeros((n_time_steps + 1, n_q))
    
    # Pre-compute some constants
    sigma2 = sigma * sigma
    gamma_half_sigma2 = (gamma / 2.0) * sigma2
    
    # For each time step backwards
    for t_idx in range(n_time_steps - 1, -1, -1):
        t = t_grid[t_idx]
        t_next = t_grid[t_idx + 1]
        
        # V_next[q_idx] = V at t_next for each q
        V_next = V[t_idx + 1, :]
        
        # For each inventory level, optimize over (d_b, d_a)
        for q_idx, q in enumerate(q_grid):
            # Inventory after bid fill: q+1 (if q < Q)
            # Inventory after ask fill: q-1 (if q > -Q)
            
            # We need V at (t_next, q+1) and (t_next, q-1)
            # Use linear interpolation or just index if within bounds
            
            if q < Q:
                q_bid = q + 1
                q_bid_idx = q_bid_idx = q_idx + 1
                V_bid = V_next[q_bid_idx]
            else:
                V_bid = 0.0  # no bid fill at q = Q
            
            if q > -Q:
                q_ask = q - 1
                q_ask_idx = q_idx - 1
                V_ask = V_next[q_ask_idx]
            else:
                V_ask = 0.0  # no ask fill at q = -Q
            
            # Optimize over d_b and d_a
            # Expected instantaneous reward + change in value
            # f(d_b, d_a) = A*exp(-k*d_b)*d_b - A*exp(-k*d_a)*d_a 
            #               - gamma_half_sigma2 * q^2
            #               + A*exp(-k*d_b) * (V_bid - V_next[q_idx])  # bid fill changes V
            #               + A*exp(-k*d_a) * (V_ask - V_next[q_idx])  # ask fill changes V
            #               + (sigma2/2) * V_qq  # diffusion term (approximate)
            
            # For simplicity, ignore V_qq in the optimization (it doesn't depend on action)
            # The action-dependent part is:
            # g(d_b, d_a) = A*exp(-k*d_b)*(d_b + V_bid - V_next[q_idx]) 
            #              - A*exp(-k*d_a)*(d_a - V_ask + V_next[q_idx])
            
            # Wait, let me redo this more carefully.
            # 
            # The HJB (discretized in time):
            # V(t, q) = sup_a [ dt * (reward_rate + drift * V_q) + (1-dt) * V(t+dt, q) 
            #                   + dt * (sigma2/2) * V_qq ]
            # 
            # Actually, for the discrete model, it's easier to work directly with the discrete dynamics.
            
            # Let's use the discrete-time Bellman equation:
            # V(t, q) = sup_{d_b, d_a} [ 
            #     p_b * (d_b + V(t+dt, q+1)) + 
            #     p_a * (-d_a + V(t+dt, q-1)) + 
            #     (1 - p_b - p_a) * V(t+dt, q) 
            #     - (gamma/2) * sigma2 * q^2 * dt
            # ]
            # where p_b = 1 - exp(-A*exp(-k*d_b)*dt), p_a = 1 - exp(-A*exp(-k*d_a)*dt)
            
            # For small dt, p_b ≈ A*exp(-k*d_b)*dt, p_a ≈ A*exp(-k*d_a)*dt
            
            # So: V(t,q) ≈ sup [ 
            #     A*exp(-k*d_b)*dt * (d_b + V(t+dt, q+1) - V(t+dt, q)) +
            #     A*exp(-k*d_a)*dt * (-d_a + V(t+dt, q-1) - V(t+dt, q))
            #     + V(t+dt, q)
            #     - (gamma/2)*sigma2*q^2*dt
            # ]
            
            # = V(t+dt, q) + dt * sup [
            #     A*exp(-k*d_b)*(d_b + V_bid - V_mid) -
            #     A*exp(-k*d_a)*(d_a - V_ask + V_mid)
            #     - (gamma/2)*sigma2*q^2
            # ]
            # where V_mid = V(t+dt, q)
            
            V_mid = V_next[q_idx]
            
            # Define the objective to maximize (without the V_mid + dt*constant part)
            # h(d_b, d_a) = A*exp(-k*d_b)*(d_b + V_bid - V_mid) 
            #             - A*exp(-k*d_a)*(d_a - V_ask + V_mid)
            
            # This is separable! We can optimize d_b and d_a independently.
            
            # For d_b: maximize A*exp(-k*d_b)*(d_b + C_b) where C_b = V_bid - V_mid
            # Let f_b(d_b) = A*exp(-k*d_b)*(d_b + C_b)
            # f_b'(d_b) = A*exp(-k*d_b)*(-k*(d_b + C_b) + 1) = 0
            # => 1 - k*(d_b + C_b) = 0
            # => d_b* = 1/k - C_b = 1/k - (V_bid - V_mid)
            # 
            # But d_b must be >= 0, so d_b* = max(0, 1/k - (V_bid - V_mid))
            
            # For d_a: maximize -A*exp(-k*d_a)*(d_a - C_a) where C_a = V_ask - V_mid
            # Let f_a(d_a) = -A*exp(-k*d_a)*(d_a - C_a)
            # f_a'(d_a) = -A*exp(-k*d_a)*(-k*(d_a - C_a) + 1) = 0
            # => 1 - k*(d_a - C_a) = 0  (with the negative sign absorbed)
            # Wait: f_a'(d_a) = -A*exp(-k*d_a)*[1 - k*(d_a - C_a)]
            # Set to 0: 1 - k*(d_a - C_a) = 0 => d_a* = 1/k + C_a = 1/k + (V_ask - V_mid)
            # 
            # But we need f_a to be maximized. Let's check the second derivative or just evaluate.
            # Actually, let's just compute the optimal d_b and d_a analytically.
            
            C_b = V_bid - V_mid
            C_a = V_ask - V_mid
            
            d_b_opt = 1.0 / k - C_b
            d_a_opt = 1.0 / k + C_a
            
            # Clamp to valid range
            d_b_opt = max(d_b_opt, 0.0)
            d_a_opt = max(d_a_opt, 0.0)
            
            # Compute the optimal value
            if q < Q:
                term_b = A * np.exp(-k * d_b_opt) * (d_b_opt + C_b)
            else:
                term_b = 0.0
            
            if q > -Q:
                term_a = -A * np.exp(-k * d_a_opt) * (d_a_opt - C_a)
            else:
                term_a = 0.0
            
            V[t_idx, q_idx] = V_mid + dt * (term_b + term_a - gamma_half_sigma2 * q * q)
    
    return V, t_grid, q_grid


def _solve_hjb_full(params, n_time_steps=200, Q=None):
    """
    Solve the HJB PDE more carefully using the discrete-time Bellman equation
    with the exact hit probabilities.
    """
    T = params['T']
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    if Q is None:
        Q = params['Q']
    
    dt = T / n_time_steps
    t_grid = np.linspace(0, T, n_time_steps + 1)
    q_grid = np.arange(-Q, Q + 1)
    n_q = len(q_grid)
    
    # V[t_idx, q_idx]
    V = np.zeros((n_time_steps + 1, n_q))
    
    gamma_half_sigma2 = (gamma / 2.0) * sigma * sigma
    
    for t_idx in range(n_time_steps - 1, -1, -1):
        V_next = V[t_idx + 1, :]
        
        for q_idx, q in enumerate(q_grid):
            V_mid = V_next[q_idx]
            
            # Bid fill
            if q < Q:
                V_bid = V_next[q_idx + 1]
                C_b = V_bid - V_mid
                # Maximize A*exp(-k*d_b)*dt*(d_b + C_b) over d_b >= 0
                # For small dt, the optimal d_b is approximately 1/k - C_b (from first-order condition)
                # But let's be more precise: maximize exp(-k*d_b)*(d_b + C_b)
                # f(d) = exp(-k*d)*(d + C)
                # f'(d) = exp(-k*d)*(1 - k*(d + C))
                # f'(d) = 0 => d = 1/k - C
                d_b_opt = 1.0 / k - C_b
                d_b_opt = max(d_b_opt, 0.0)
                term_b = A * np.exp(-k * d_b_opt) * dt * (d_b_opt + C_b)
            else:
                term_b = 0.0
            
            # Ask fill
            if q > -Q:
                V_ask = V_next[q_idx - 1]
                C_a = V_ask - V_mid
                # Maximize -A*exp(-k*d_a)*dt*(d_a - C_a) over d_a >= 0
                # i.e., minimize A*exp(-k*d_a)*(d_a - C_a)
                # g(d) = exp(-k*d)*(d - C_a)
                # g'(d) = exp(-k*d)*(1 - k*(d - C_a))
                # g'(d) = 0 => d = 1/k + C_a
                d_a_opt = 1.0 / k + C_a
                d_a_opt = max(d_a_opt, 0.0)
                term_a = -A * np.exp(-k * d_a_opt) * dt * (d_a_opt - C_a)
            else:
                term_a = 0.0
            
            V[t_idx, q_idx] = V_mid + term_b + term_a - gamma_half_sigma2 * q * q * dt
    
    return V, t_grid, q_grid


def _precompute_hjb():
    """Pre-compute HJB solutions for all three regimes."""
    global _HJB_SOLUTION, _HJB_PARAMS
    
    regimes = [
        {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.05, 'T': 1.0, 'steps': 1000, 'Q': 30},
        {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.5, 'T': 1.0, 'steps': 1000, 'Q': 30},
        {'sigma': 1.5, 'A': 50.0, 'k': 0.75, 'gamma': 0.2, 'T': 1.0, 'steps': 1000, 'Q': 30},
    ]
    
    solutions = {}
    n_time_steps = 200
    
    for reg in regimes:
        key = (reg['sigma'], reg['A'], reg['k'], reg['gamma'])
        V, t_grid, q_grid = _solve_hjb_full(reg, n_time_steps=n_time_steps)
        solutions[key] = (V, t_grid, q_grid)
    
    _HJB_SOLUTION = solutions
    _HJB_PARAMS = regimes


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths using the pre-computed HJB solution.
    
    Returns (bid_depths, ask_depths) as arrays of shape (4000,).
    """
    global _HJB_SOLUTION
    
    if _HJB_SOLUTION is None:
        _precompute_hjb()
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma)
    
    if key not in _HJB_SOLUTION:
        # Fallback: solve on the fly
        V, t_grid, q_grid = _solve_hjb_full(params, n_time_steps=200)
        _HJB_SOLUTION[key] = (V, t_grid, q_grid)
    else:
        V, t_grid, q_grid = _HJB_SOLUTION[key]
    
    n_time_steps = len(t_grid) - 1
    dt = T / n_time_steps
    
    # Determine current time index
    t_idx = int(round(t / dt))
    t_idx = max(0, min(t_idx, n_time_steps - 1))
    
    # V at current time and next time
    V_curr = V[t_idx, :]
    V_next = V[min(t_idx + 1, n_time_steps), :]
    
    # q_grid is from -Q to Q
    # inventory is in [-Q, Q]
    # Map inventory to q_grid index
    # q_grid = np.arange(-Q, Q+1), so index = inventory + Q
    
    inventory_int = np.round(inventory).astype(int)
    inventory_int = np.clip(inventory_int, -Q, Q)
    
    q_indices = inventory_int + Q
    
    bid_depths = np.full(len(inventory), np.inf)
    ask_depths = np.full(len(inventory), np.inf)
    
    for i in range(len(inventory)):
        q = inventory_int[i]
        q_idx = q_indices[i]
        
        V_mid = V_next[q_idx]
        
        # Bid
        if q < Q:
            V_bid = V_next[q_idx + 1]
            C_b = V_bid - V_mid
            d_b_opt = 1.0 / k - C_b
            d_b_opt = max(d_b_opt, 0.0)
            bid_depths[i] = d_b_opt
        # else: no bid quote, leave as inf
        
        # Ask
        if q > -Q:
            V_ask = V_next[q_idx - 1]
            C_a = V_ask - V_mid
            d_a_opt = 1.0 / k + C_a
            d_a_opt = max(d_a_opt, 0.0)
            ask_depths[i] = d_a_opt
        # else: no ask quote, leave as inf
    
    # Ensure depths are non-negative and finite
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    # Replace any NaN or negative values with inf (no quote)
    bid_depths[~np.isfinite(bid_depths)] = np.inf
    ask_depths[~np.isfinite(ask_depths)] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
