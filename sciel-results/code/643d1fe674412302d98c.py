# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0
_POLICY_CACHE = {}

def _solve_optimal_depth(alpha, k, C):
    """
    Solve for optimal depth x >= 0 that maximizes f(x) = (1 - exp(-alpha*exp(-k*x))) * (x + C)
    Using Newton-Raphson on the first-order condition.
    """
    if alpha <= 0:
        return 0.0
    
    # f'(x) = alpha*k*exp(-k*x)*exp(-alpha*exp(-k*x))*(x + C) + (1 - exp(-alpha*exp(-k*x)))
    # Let u = exp(-k*x), so x = -ln(u)/k, u in (0, 1]
    # f'(x) = 0 => alpha*k*u*exp(-alpha*u)*(x + C) + 1 - exp(-alpha*u) = 0
    
    def f_prime(x):
        if x < 0:
            return 1e10
        u = np.exp(-k * x)
        p = 1.0 - np.exp(-alpha * u)
        return alpha * k * u * np.exp(-alpha * u) * (x + C) + p
    
    def f_prime2(x):
        if x < 0:
            return 1e10
        u = np.exp(-k * x)
        au = alpha * u
        p = 1.0 - np.exp(-au)
        # f''(x) is complex, use finite difference
        h = 1e-6
        return (f_prime(x + h) - f_prime(x - h)) / (2 * h)
    
    # Initial guess
    x = 0.5 / k
    
    for _ in range(20):
        fp = f_prime(x)
        if abs(fp) < 1e-8:
            break
        fp2 = f_prime2(x)
        if abs(fp2) < 1e-12:
            break
        dx = -fp / fp2
        x = x + dx
        if x < 0:
            x = 0.0
            break
    
    return max(0.0, x)

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal policy via dynamic programming with Newton-Raphson depth solver.
    Returns a 2D array policy[step, inv_idx] = (bid_depth, ask_depth)
    """
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1
    
    V = np.zeros((n_steps + 1, n_inv))
    policy = np.full((n_steps, n_inv, 2), np.inf)
    
    alpha = A * dt
    
    for s in range(n_steps - 1, -1, -1):
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            V_s_q = -0.5 * gamma * sigma**2 * q**2 * dt
            
            bid_depth = np.inf
            ask_depth = np.inf
            
            if q < Q:
                C_bid = V[s + 1, q + 1 + Q] - V[s + 1, q + Q]
                bid_depth = _solve_optimal_depth(alpha, k, C_bid)
                
                p_b = 1.0 - np.exp(-alpha * np.exp(-k * bid_depth))
                V_s_q += p_b * (bid_depth + C_bid)
            
            if q > -Q:
                C_ask = V[s + 1, q - 1 + Q] - V[s + 1, q + Q]
                ask_depth = _solve_optimal_depth(alpha, k, C_ask)
                
                p_a = 1.0 - np.exp(-alpha * np.exp(-k * ask_depth))
                V_s_q += p_a * (ask_depth + C_ask)
            
            V[s, q_idx] = V_s_q
            policy[s, q_idx, 0] = bid_depth
            policy[s, q_idx, 1] = ask_depth
    
    return policy

def _get_policy(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        if time.time() > _DEADLINE - 15:
            return None
        _POLICY_CACHE[key] = _compute_policy(*key)
    return _POLICY_CACHE[key]

def quote(t, inventory, mid, params):
    if time.time() > _DEADLINE - 5:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    policy = _get_policy(params)
    if policy is None:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    n = min(n, params['steps'] - 1)
    n = max(n, 0)
    
    Q = params['Q']
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    bid_depths = policy[n, inv_idx, 0].copy()
    ask_depths = policy[n, inv_idx, 1].copy()
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
