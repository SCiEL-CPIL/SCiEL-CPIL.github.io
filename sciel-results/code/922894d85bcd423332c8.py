"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """Compute R_k for k=1..N-1. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _max_overlap(h):
    """Compute the maximum overlap (worst case)."""
    return np.max(_compute_overlaps(h))

def _project_to_simplex(h, target_sum):
    """Project h to satisfy sum(h) = target_sum and 0<=h<=1."""
    h = np.clip(h, 0, 1).copy()
    for _ in range(30):
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-13:
            break
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        if np.sum(mask) == 0:
            # If all at bounds, try relaxing slightly
            mask = (h > 0) & (h < 1)
            if np.sum(mask) == 0:
                break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0, 1)
    # Final exact adjustment
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-10:
        # Distribute to elements with most room
        rooms = np.where(h > 0.5, 1 - h, h)
        total_room = np.sum(rooms)
        if total_room > 1e-10:
            h += diff * rooms / total_room
            h = np.clip(h, 0, 1)
    return h

def _soft_min_obj_and_grad(params, beta):
    """
    Objective and gradient for symmetric h.
    params: length N/2, represents h[0:N/2]
    h is constructed symmetrically: h[i] = params[i] for i < N/2, h[N-1-i] = params[i]
    """
    n_half = len(params)
    n = 2 * n_half
    h = np.zeros(n)
    h[:n_half] = params
    h[n_half:] = params[::-1]
    
    corrs = _compute_overlaps(h)
    
    # Soft min of corrs
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    # Gradient
    exp_neg_beta = np.exp(-beta * corrs)
    sum_exp_neg = np.sum(exp_neg_beta)
    weights = exp_neg_beta / sum_exp_neg
    
    grad_h = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w < 1e-15:
            continue
        grad_h[:n-k] += w * h[k:]
        grad_h[k:] += w * h[:n-k]
    
    grad_h = -grad_h
    
    # Map grad_h back to params (symmetric)
    grad_params = grad_h[:n_half] + grad_h[n-1:n_half-1:-1]
    
    return obj, grad_params

def _construct_symmetric(params):
    n_half = len(params)
    n = 2 * n_half
    h = np.zeros(n)
    h[:n_half] = params
    h[n_half:] = params[::-1]
    return h

def _bang_bang_initialization(n_half, seed):
    """Create bang-bang (0/1) initialization with random pattern."""
    rng = np.random.RandomState(seed)
    params = rng.randint(0, 2, n_half).astype(float)
    # Ensure sum is close to n/4
    target = n_half / 2.0
    current_sum = np.sum(params)
    if current_sum < target:
        # Flip some 0s to 1s
        zeros = np.where(params == 0)[0]
        need = int(target - current_sum)
        if len(zeros) >= need:
            idx = rng.choice(zeros, size=need, replace=False)
            params[idx] = 1.0
    elif current_sum > target:
        ones = np.where(params == 1)[0]
        need = int(current_sum - target)
        if len(ones) >= need:
            idx = rng.choice(ones, size=need, replace=False)
            params[idx] = 0.0
    return params

def _refine_bang_bang(params, max_iters=500):
    """Local search: flip individual elements if it improves max overlap."""
    n_half = len(params)
    target = n_half / 2.0
    best_params = params.copy()
    best_obj = _max_overlap(_construct_symmetric(best_params))
    
    for it in range(max_iters):
        improved = False
        for i in range(n_half):
            old_val = best_params[i]
            new_val = 1.0 - old_val
            # Check if flip keeps sum constraint
            new_sum = np.sum(best_params) - old_val + new_val
            if abs(new_sum - target) > 0.51:
                continue
            test_params = best_params.copy()
            test_params[i] = new_val
            test_h = _construct_symmetric(test_params)
            test_obj = _max_overlap(test_h)
            if test_obj < best_obj - 1e-10:
                best_params = test_params
                best_obj = test_obj
                improved = True
        if not improved:
            break
    return best_params, best_obj

def construct():
    n = N
    n_half = n // 2
    target_sum = n / 2.0
    target_param_sum = n / 4.0
    
    best_params = np.full(n_half, 0.5)
    best_obj = float('inf')
    
    # Initial evaluation
    h_init = _construct_symmetric(best_params)
    best_obj = _max_overlap(h_init)
    
    np.random.seed(42)
    
    # Generate diverse initializations
    restarts = []
    
    # Uniform
    restarts.append(np.full(n_half, 0.5))
    
    # Linear gradients
    restarts.append(np.linspace(0, 1, n_half))
    restarts.append(np.linspace(1, 0, n_half))
    
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n_half)
    for freq in [1, 2, 3, 4, 5, 8]:
        restarts.append(0.5 + 0.45*np.sin(freq*x))
        restarts.append(0.5 + 0.45*np.cos(freq*x))
    
    # Random
    for _ in range(30):
        restarts.append(np.random.uniform(0, 1, n_half))
    
    # Bang-bang patterns
    for seed in range(20):
        restarts.append(_bang_bang_initialization(n_half, seed))
    
    # Block patterns
    for block_size in [2, 4, 5, 10]:
        n_blocks = n_half // block_size
        for _ in range(3):
            pattern = np.random.uniform(0, 1, n_blocks)
            block_params = np.repeat(pattern, block_size)
            if len(block_params) < n_half:
                block_params = np.concatenate([block_params, np.random.uniform(0,1, n_half - len(block_params))])
            restarts.append(block_params[:n_half])
    
    # Optimize each restart
    for x0 in restarts:
        if _time_remaining() < 3.0:
            break
        
        # Project to feasible region
        params = _project_to_simplex(x0, target_param_sum)
        
        # Quick local search for bang-bang-like solutions
        if np.all((params < 0.1) | (params > 0.9)):
            refined_params, refined_obj = _refine_bang_bang(params, max_iters=100)
            if refined_obj < best_obj:
                best_obj = refined_obj
                best_params = refined_params
            continue
        
        # Multi-stage beta optimization
        for beta in [30.0, 80.0, 150.0, 300.0]:
            if _time_remaining() < 2.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    params,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n_half,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                if res.fun < best_obj:
                    params_new = _project_to_simplex(res.x, target_param_sum)
                    h_new = _construct_symmetric(params_new)
                    obj_new = _max_overlap(h_new)
                    if obj_new < best_obj:
                        best_obj = obj_new
                        best_params = params_new
                        params = params_new
            except Exception:
                pass
    
    # Final polish with full vector optimization if time permits
    if _time_remaining() > 15.0:
        h_full = _construct_symmetric(best_params)
        
        def full_obj_and_grad(h):
            corrs = _compute_overlaps(h)
            min_val = np.min(corrs)
            shifted = corrs - min_val
            beta = 500.0
            log_sum = np.log(np.sum(np.exp(-beta * shifted)))
            soft_min = min_val - log_sum / beta
            obj = 1.0 - soft_min
            
            exp_neg_beta = np.exp(-beta * corrs)
            sum_exp_neg = np.sum(exp_neg_beta)
            weights = exp_neg_beta / sum_exp_neg
            
            grad_h = np.zeros(n)
            for k in range(1, n):
                w = weights[k-1] * (2.0 / n)
                if w < 1e-15:
                    continue
                grad_h[:n-k] += w * h[k:]
                grad_h[k:] += w * h[:n-k]
            
            return obj, -grad_h

        try:
            res = minimize(
                full_obj_and_grad,
                h_full,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            h_new = np.clip(res.x, 0, 1)
            h_new = _project_to_simplex(h_new, target_sum)
            
            corrs_new = _compute_overlaps(h_new)
            obj_new = np.max(corrs_new)
            if obj_new < best_obj:
                best_obj = obj_new
                final_h = h_new
            else:
                final_h = _construct_symmetric(best_params)
        except Exception:
            final_h = _construct_symmetric(best_params)
    else:
        final_h = _construct_symmetric(best_params)
    
    # Final adjustment for exact sum
    final_h = _project_to_simplex(final_h, target_sum)
    
    return final_h
# EVOLVE-BLOCK-END
