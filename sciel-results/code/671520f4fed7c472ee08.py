"""
Combined approach: Spectral-flatness optimization via continuous relaxation
with Dirichlet-kernel parametric construction and L-BFGS-B polish.

The task: minimize max_k |c_k(x)| / (sum x_i)^2 for signed step functions of length N.
The constant sequence gives ratio = 1/N (since c_0 = N, sum^2 = N^2).

Key insights:
1. From current program: use soft-max surrogate for off-diagonal max, gradient descent.
2. From inspiration 1: spectral flatness (minimize sum |DFT|^4) correlates with low sidelobes.
3. From strategy: represent signal as sum of shifted/scaled/signed Dirichlet kernels
   (block pulses) with continuous positions/widths, optimize geometric params via DE,
   then polish with L-BFGS-B on the full continuous relaxation.

Strategy:
1. Phase 1: Parametric search - construct signals from K block pulses with continuous
   positions, widths, and signs. Use differential evolution to find good geometry.
2. Phase 2: For each good parametric solution, run L-BFGS-B polish on the full N-dim
   continuous relaxation using soft-max surrogate (two-phase annealing).
3. Phase 3: Final local search (single flips) on the best sign-thresholded candidate.
4. Time budget: ~140s wall clock with safety margin.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400
DEADLINE = 140.0  # seconds (150 - 10 safety)
SAFETY_MARGIN = 5.0

def _autoconv(x):
    """Compute full autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N
    X = np.fft.fft(x, n=n)
    C = np.fft.ifft(X * X).real[:2*N-1]
    return C

def _eval_ratio(x):
    """Compute the raw value: max|c_k| / (sum x)^2."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(c)) / (s * s)

def _soft_max_abs(vals, temperature):
    """Soft approximation of max of |vals| using log-sum-exp. Returns (value, gradient)."""
    abs_vals = np.abs(vals)
    m = np.max(abs_vals)
    if m < 1e-15:
        return m, np.zeros_like(vals)
    exp_vals = np.exp((abs_vals - m) / temperature)
    sum_exp = np.sum(exp_vals)
    soft_max_val = m + temperature * np.log(sum_exp)
    grad = np.sign(vals) * exp_vals / sum_exp
    return soft_max_val, grad

def _dirichlet_kernel_block(center, width, sign, N):
    """Create a single block pulse (Dirichlet-like) centered at 'center' with 'width'."""
    x = np.zeros(N)
    half = max(1, int(round(width / 2.0)))
    start = int(round(center - half))
    end = int(round(center + half))
    # Clip to valid range
    start = max(0, start)
    end = min(N, end)
    x[start:end] = sign
    return x

def _parametric_signal(params, N):
    """
    Construct signal from K block pulses.
    params: array of shape (K, 3) where each row is [center, width, sign]
    sign is in [-1, 1] (continuous, will be thresholded later)
    """
    x = np.zeros(N)
    K = len(params)
    for k in range(K):
        center = params[k, 0]
        width = params[k, 1]
        sign = params[k, 2]
        # Ensure width is positive
        width = max(2.0, abs(width))
        x += _dirichlet_kernel_block(center, width, sign, N)
    return x

def _parametric_objective(params, N):
    """Objective for parametric optimization: eval_ratio of the constructed signal."""
    x = _parametric_signal(params, N)
    s = np.sum(x)
    if abs(s) < 1e-6:
        return 1e6
    return _eval_ratio(x)

def _lbfgs_polish(x0, deadline, temp_init=1.0, temp_min=0.1, max_iter=2000):
    """
    L-BFGS-B polish on continuous relaxation using soft-max surrogate.
    Two-phase: anneal temperature from temp_init to temp_min.
    """
    x = x0.copy().astype(np.float64)
    # Normalize to sum=1
    s = np.sum(x)
    if abs(s) < 1e-12:
        x = np.ones(N) / N
    else:
        x = x / s
    
    best_obj = np.inf
    best_x = x.copy()
    
    # Phase 1: coarse with higher temperature
    # Phase 2: fine with lower temperature
    
    for phase, temp in [(0, temp_init), (1, temp_min)]:
        if time.time() > deadline:
            break
        
        max_it_phase = max_iter // 2
        
        def objective_and_grad(x_flat):
            if time.time() > deadline:
                return np.inf, np.zeros(N)
            x = x_flat
            c = _autoconv(x)
            c0 = c[N-1]
            off_diag = np.concatenate([c[:N-1], c[N:]])
            sm_val, sm_grad = _soft_max_abs(off_diag, temp)
            obj = c0 + sm_val
            
            # Gradient
            grad = 2.0 * x  # d(c0)/dx
            
            # Gradient from off-diagonal soft-max
            w_pos = sm_grad[N-1:]  # length N-1
            w_neg = sm_grad[:N-1]  # length N-1
            w = 0.5 * (w_pos + w_neg[::-1])
            
            grad_off = np.zeros(N)
            for k in range(1, N):
                wk = 2.0 * w[k-1]
                if abs(wk) < 1e-15:
                    continue
                grad_off[:N-k] += wk * x[k:]
                grad_off[k:] += wk * x[:N-k]
            
            grad += grad_off
            grad -= np.mean(grad)  # Project to maintain sum=1
            
            return obj, grad
        
        # Use L-BFGS-B
        res = minimize(
            objective_and_grad,
            x,
            method='L-BFGS-B',
            options={'maxiter': max_it_phase, 'gtol': 1e-4, 'maxfun': max_it_phase}
        )
        
        if res.success or res.fun < best_obj:
            x = res.x
            # Re-normalize
            s = np.sum(x)
            if abs(s) > 1e-12:
                x = x / s
            obj_final = objective_and_grad(x)[0]
            if obj_final < best_obj:
                best_obj = obj_final
                best_x = x.copy()
    
    return best_x, best_obj

def construct():
    """
    Return signed step heights that minimize max|c_k| / (sum x_i)^2.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Baseline: constant sequence
    best_result = np.ones(N)
    best_val = _eval_ratio(best_result)  # = 1/N
    
    # Phase 1: Parametric search with differential evolution
    # Use K block pulses
    K = 4  # number of blocks
    # Bounds: center in [0, N), width in [2, N), sign in [-1, 1]
    bounds = [(0, N), (2, N), (-1, 1)] * K
    
    np.random.seed(42)
    
    # Run DE with limited population and generations
    t_remaining = deadline - time.time()
    if t_remaining > 20:  # Only if we have enough time
        try:
            # Limit DE runtime
            de_start = time.time()
            de_result = differential_evolution(
                _parametric_objective,
                bounds,
                args=(N,),
                maxiter=30,
                popsize=8,
                tol=0.01,
                seed=42,
                polish=False,
                update_interval=1
            )
            # Evaluate the best parametric solution
            x_param = _parametric_signal(de_result.x, N)
            v_param = _eval_ratio(x_param)
            if v_param < best_val:
                best_val = v_param
                best_result = x_param.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B polish on best candidate
    # Also try a few diverse initializations
    candidates = [best_result.copy()]
    
    # Add some diverse candidates
    rng = np.random.default_rng(123)
    for _ in range(3):
        # Random signs
        x_rand = rng.choice([-1.0, 1.0], size=N)
        candidates.append(x_rand.astype(float))
    
    # Alternating
    candidates.append(np.tile([1.0, -1.0], N // 2).astype(float))
    
    for x0 in candidates:
        if time.time() > deadline:
            break
        
        # Normalize
        s = np.sum(x0)
        if abs(s) < 1e-12:
            continue
        x0_norm = x0 / s
        
        # L-BFGS-B polish
        x_polished, obj = _lbfgs_polish(x0_norm, deadline, temp_init=1.0, temp_min=0.1, max_iter=1500)
        
        # Evaluate
        v = _eval_ratio(x_polished)
        if v < best_val:
            best_val = v
            best_result = x_polished.copy()
    
    # Phase 3: Threshold to signs and local search
    # Try sign-thresholding the best continuous solution
    sign_result = np.sign(best_result)
    sign_result[sign_result == 0] = 1.0
    v_sign = _eval_ratio(sign_result)
    if v_sign < best_val:
        best_val = v_sign
        best_result = sign_result.copy()
    
    # Local search: single flips
    s = best_result.copy()
    v = _eval_ratio(s)
    for _round in range(4):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            s[i] = -s[i]
            v2 = _eval_ratio(s)
            if v2 < v - 1e-12:
                v = v2
                improved = True
            else:
                s[i] = -s[i]  # revert
        if not improved:
            break
    
    if v < best_val:
        best_val = v
        best_result = s.copy()
    
    # Final safety check
    s_final = np.sum(best_result)
    if abs(s_final) < 1e-12:
        return np.ones(N)
    
    return best_result
