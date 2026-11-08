"""Market-making quote strategy using parametric policy with simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

_policy_cache = {}
_DEADLINE = 290.0
_start_time = None


def _simulate_objective(params, policy_params, sigma, A, k, gamma, T, steps, Q, n_paths, seed=42):
    """
    Simulate n_paths and compute the mean objective.
    policy_params: array of shape (6,) = [c0, c1, c2, c3, c4, c5]
    bid_depth = c0 + c1*inv + c2*inv^2 + c3*mid_norm + c4*mid_norm^2 + c5*mid_norm*inv
    ask_depth = c0 - c1*inv - c2*inv^2 + c3*mid_norm - c4*mid_norm^2 - c5*mid_norm*inv
    where mid_norm = (mid - 100) / sigma
    """
    rng = np.random.default_rng(seed)
    dt = T / steps
    n_q = 2 * Q + 1
    
    c0, c1, c2, c3, c4, c5 = policy_params
    
    total_obj = 0.0
    for path in range(n_paths):
        inv = 0
        mid = 100.0
        obj = 0.0
        for n in range(steps):
            # Compute depths
            mid_norm = (mid - 100.0) / sigma
            bid_d = c0 + c1*inv + c2*inv*inv + c3*mid_norm + c4*mid_norm*mid_norm + c5*mid_norm*inv
            ask_d = c0 - c1*inv - c2*inv*inv + c3*mid_norm - c4*mid_norm*mid_norm - c5*mid_norm*inv
            
            min_depth = -3.0 / k
            bid_d = max(bid_d, min_depth)
            ask_d = max(ask_d, min_depth)
            
            # Inventory cost
            obj += -0.5 * gamma * sigma**2 * inv * inv * dt
            
            # Fills
            if inv < Q:
                lam_b = A * np.exp(-k * bid_d)
                p_b = 1.0 - np.exp(-lam_b * dt)
                if rng.random() < p_b:
                    obj += bid_d
                    inv += 1
            if inv > -Q:
                lam_a = A * np.exp(-k * ask_d)
                p_a = 1.0 - np.exp(-lam_a * dt)
                if rng.random() < p_a:
                    obj += ask_d
                    inv -= 1
            
            # Mid moves
            mid += sigma * np.sqrt(dt) * rng.standard_normal()
        
        total_obj += obj
    
    return total_obj / n_paths


def _optimize_policy(sigma, A, k, gamma, T, steps, Q, deadline):
    """
    Optimize policy parameters using simulated annealing.
    Returns the best policy_params found.
    """
    # Initialize with reasonable defaults
    # At inv=0, mid=100: bid=ask=c0. A good starting depth is ~1/k
    best_params = np.array([1.0/k, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_obj = _simulate_objective(None, best_params, sigma, A, k, gamma, T, steps, Q, 50, seed=42)
    
    # Simulated annealing
    current_params = best_params.copy()
    current_obj = best_obj
    T_start = 0.5
    T_end = 0.01
    n_iter = 5000
    cooling = (T_end / T_start) ** (1.0 / max(n_iter - 1, 1))
    
    rng = np.random.default_rng(123)
    
    for it in range(n_iter):
        if time.time() - _start_time > deadline:
            break
        
        temp = T_start * (cooling ** it)
        
        # Propose new parameters
        step_size = 0.1 * (1.0 + it / n_iter)
        new_params = current_params + rng.standard_normal(6) * step_size
        
        # Evaluate
        new_obj = _simulate_objective(None, new_params, sigma, A, k, gamma, T, steps, Q, 50, seed=42)
        
        # Accept/reject
        if new_obj > current_obj or rng.random() < np.exp((new_obj - current_obj) / (temp + 1e-10)):
            current_params = new_params
            current_obj = new_obj
            if new_obj > best_obj:
                best_obj = new_obj
                best_params = new_params.copy()
    
    return best_params


def _compute_parametric_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the parametric policy.
    Returns policy_params array.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _policy_cache:
        return _policy_cache[key]
    
    deadline = _DEADLINE - 20  # safety margin
    
    policy_params = _optimize_policy(sigma, A, k, gamma, T, steps, Q, deadline)
    
    _policy_cache[key] = policy_params
    return policy_params


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths using the parametric policy.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    try:
        policy_params = _compute_parametric_policy(sigma, A, k, gamma, T, steps, Q)
    except Exception:
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    c0, c1, c2, c3, c4, c5 = policy_params
    
    inv = inventory.astype(float)
    mid_norm = (mid - 100.0) / sigma
    
    bid_depths = c0 + c1*inv + c2*inv*inv + c3*mid_norm + c4*mid_norm*mid_norm + c5*mid_norm*inv
    ask_depths = c0 - c1*inv - c2*inv*inv + c3*mid_norm - c4*mid_norm*mid_norm - c5*mid_norm*inv
    
    min_depth = -3.0 / k
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Handle boundary conditions: no bid at q=Q, no ask at q=-Q
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)
    
    # Clean up any NaN or invalid values
    bid_depths = np.where(np.isnan(bid_depths), 1.0/k, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), 1.0/k, ask_depths)
    bid_depths = np.where(np.isinf(bid_depths) & (inventory < Q), 1.0/k, bid_depths)
    ask_depths = np.where(np.isinf(ask_depths) & (inventory > -Q), 1.0/k, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
