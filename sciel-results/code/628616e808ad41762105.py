"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses a hybrid approach: Fast Projected Gradient Descent on a smooth-max 
objective for broad search, followed by SLSQP polishing for precision.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 135  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    Uses a simple iterative clipping and redistribution.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    for _ in range(10): # A few iterations to ensure convergence of projection
        diff = target - np.sum(h)
        if abs(diff) < 1e-15:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        
    # Final exact adjustment if needed
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        # Add to the element with most room
        if diff > 0:
            idx = np.argmax(1.0 - h)
            add = min(diff, 1.0 - h[idx])
            h[idx] += add
            diff -= add
            # If still diff, distribute to others
            if abs(diff) > 1e-12:
                mask = h < 1.0
                if np.any(mask):
                    h[mask] += diff / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
        else:
            idx = np.argmin(h)
            sub = min(-diff, h[idx])
            h[idx] -= sub
            diff += sub
            if abs(diff) > 1e-12:
                mask = h > 0.0
                if np.any(mask):
                    h[mask] += diff / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
                    
    return h


def _smooth_max_value_grad(h, beta):
    """
    Returns value and gradient of the smooth max of C_k.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights
    
    val = m + np.log(sum_exp) / beta

    # Gradient calculation using FFT
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)

    # term1_j = sum_{k=0}^{j} w[k] * h[j-k]
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]

    # term2_j = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    h_flip = h[::-1]
    h_flip_pad = np.zeros(n_fft)
    h_flip_pad[:n] = h_flip
    HF = np.fft.rfft(h_flip_pad, n=n_fft)
    conv2_full = np.fft.irfft(W * HF, n=n_fft)
    term2 = conv2_full[n - 1 - np.arange(n)]

    grad = -(2.0 / n) * (conv1 + term2)

    return val, grad


def _projected_gradient_descent(h0, beta, max_iter=2000, lr=None, tol=1e-9):
    """
    Projected Gradient Descent on the smooth max objective.
    Projection handles sum constraint and box constraints.
    """
    h = np.array(h0, dtype=np.float64)
    h = _fix_sum(h)
    
    if lr is None:
        lr = 1.0 / (np.max(np.abs(_smooth_max_value_grad(h, beta)[1])) + 1e-6)
        # Heuristic step size, might need scaling
        lr = min(lr, 0.5)
    
    prev_val = float('inf')
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
            
        val, grad = _smooth_max_value_grad(h, beta)
        
        if abs(prev_val - val) < tol:
            break
        prev_val = val
        
        # Gradient step
        h_new = h - lr * grad
        
        # Project to [0, 1] and sum constraint
        h_new = _fix_sum(h_new)
        
        # Simple line search / backtracking if it gets worse (optional, usually PGT handles this)
        # For now, fixed step size based on initial gradient norm
        
        h = h_new
        
    return _fix_sum(h)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)
    starts = []
    
    # 1. Analytical Starts
    starts.append(_fix_sum(0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)))
    starts.append(_fix_sum(0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)))
    starts.append(_fix_sum(0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)))
    starts.append(_fix_sum(0.5 + 0.5 * (1.0 - 2.0 * np.abs((2.0 * i / N) - 1.0))))
    starts.append(np.full(N, 0.5))
    starts.append(_fix_sum(0.5 + 0.5 * (2.0 * (i / N) - 1.0)))
    
    # 2. Random Starts (more diverse)
    for _ in range(10):
        if time.time() - t0 > _DEADLINE - 60:
            break
        h_rand = np.random.uniform(0, 1, N)
        starts.append(_fix_sum(h_rand))
        
        # Binary-like starts
        if _ % 2 == 0:
            h_bin = (np.random.rand(N) > 0.5).astype(float)
            starts.append(_fix_sum(h_bin))

    # Phase 1: Gradient Descent from all starts
    for h_start in starts:
        if time.time() - t0 > _DEADLINE - 30:
            break
        _update_best(h_start)
        
        # Run PGT with moderate beta
        h_pgd = _projected_gradient_descent(h_start, beta=50.0, max_iter=500)
        _update_best(h_pgd)

    # Phase 2: Polishing with SLSQP on the best result
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        def _run_slsqp(h0, beta, maxiter, ftol):
            nonlocal best_h, best_score
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N

            def obj_func(x):
                val, _ = _smooth_max_value_grad(x, beta)
                return val

            def jac_func(x):
                _, grad = _smooth_max_value_grad(x, beta)
                return grad

            try:
                res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

        # Polish with high beta
        _run_slsqp(best_h, beta=100.0, maxiter=200, ftol=1e-9)
        
    # Phase 3: Final very fine polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        def _run_slsqp_high(h0, beta, maxiter, ftol):
            nonlocal best_h, best_score
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N

            def obj_func(x):
                val, _ = _smooth_max_value_grad(x, beta)
                return val

            def jac_func(x):
                _, grad = _smooth_max_value_grad(x, beta)
                return grad

            try:
                res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

        _run_slsqp_high(best_h, beta=200.0, maxiter=100, ftol=1e-12)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
