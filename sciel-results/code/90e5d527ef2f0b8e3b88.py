import numpy as np
import time
from scipy.optimize import minimize

N = 100
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """Compute R_k for k=1..N-1. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _max_overlap_value(h):
    """Return the maximum overlap value."""
    corrs = _compute_overlaps(h)
    return np.max(corrs)

def _project_to_feasible(h, target_sum):
    """Project h to satisfy sum(h) = target_sum and 0<=h<=1."""
    h = np.clip(h, 0, 1)
    diff = target_sum - np.sum(h)
    for _ in range(30):
        if abs(diff) < 1e-14:
            break
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        n_active = np.sum(mask)
        if n_active == 0:
            break
        step = diff / n_active
        new_h = h.copy()
        new_h[mask] += step
        if np.any(new_h < -1e-12) or np.any(new_h > 1 + 1e-12):
            lo, hi = -1.0, 1.0
            for _ in range(50):
                mid = (lo + hi) / 2
                test_h = np.clip(h + mid, 0, 1)
                test_sum = np.sum(test_h)
                if test_sum > target_sum:
                    hi = mid
                else:
                    lo = mid
            h = np.clip(h + (lo + hi) / 2, 0, 1)
        else:
            h = new_h
        diff = target_sum - np.sum(h)
    
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-8) & (h < 1 - 1e-8)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    return h

def _spectral_init():
    """Deterministic circulant spectral construction as initial guess."""
    EPS = 1e-4
    A = max(0.25 * N - (N - 3) * EPS, 0.0)
    mags = np.full(N, EPS)
    mags[0] = 0.5 * N
    mags[1] = A
    mags[N - 1] = A
    H = mags * np.exp(1j * 2 * np.pi * np.arange(N) / N)
    h = np.real(np.fft.ifft(H))
    return _project_to_feasible(h, N / 2.0)

def _soft_min_polish(h, temps=[0.1, 0.05, 0.02, 0.01]):
    """Polish using soft-min (log-sum-exp) approximation."""
    n = len(h)
    
    def _obj_and_grad(h_vec, temp):
        h_vec = np.clip(h_vec, 0, 1)
        corrs = _compute_overlaps(h_vec)
        shifted = corrs / temp
        max_sh = np.max(shifted)
        logsumexp = max_sh + np.log(np.sum(np.exp(shifted - max_sh)))
        obj = temp * logsumexp
        
        weights = np.exp(shifted - max_sh)
        weights /= weights.sum()
        
        grad = np.zeros(n)
        for k in range(1, n):
            w = weights[k-1]
            if w < 1e-10:
                continue
            for j in range(n - k):
                grad[j] += w * h_vec[j + k]
                grad[j + k] += w * h_vec[j]
            grad *= (2.0 / n)
        
        return obj, grad
    
    best_h = h.copy()
    best_val = _max_overlap_value(h)
    
    for temp in temps:
        if _time_remaining() < 10.0:
            break
        try:
            res = minimize(
                _obj_and_grad,
                h,
                args=(temp,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0.0, 1.0)] * n,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8,
                         'maxls': 20, 'maxfun': 500}
            )
            h_new = _project_to_feasible(res.x, n / 2.0)
            v_new = _max_overlap_value(h_new)
            if v_new < best_val:
                best_val = v_new
                best_h = h_new.copy()
            h = h_new
        except Exception:
            pass
    
    return best_h, best_val

def _fourier_optimize(h, max_iters=2000, lr_init=0.1):
    """Optimize in Fourier space using subgradient method."""
    n = len(h)
    target_sum = n / 2.0
    
    h_centered = h - np.mean(h)
    coeffs = np.fft.fft(h_centered) / n
    
    best_h = h.copy()
    best_val = _max_overlap_value(h)
    
    lr = lr_init
    patience = 0
    best_iter_val = best_val
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        h_cur = np.real(np.fft.ifft(coeffs * n)) + 0.5
        h_cur = _project_to_feasible(h_cur, target_sum)
        
        corrs = _compute_overlaps(h_cur)
        max_val = np.max(corrs)
        
        if max_val < best_iter_val - 1e-10:
            best_iter_val = max_val
            patience = 0
        else:
            patience += 1
        
        if patience > 100:
            lr *= 0.9
            patience = 0
            if lr < 1e-6:
                break
        
        threshold = max_val - 1e-6
        active = corrs >= threshold
        
        subgrad = np.zeros(n)
        for k in range(1, n):
            if not active[k-1]:
                continue
            w = corrs[k-1] / max_val if max_val > 1e-15 else 1.0
            w *= (2.0 / n)
            for j in range(n - k):
                subgrad[j] += w * h_cur[j + k]
                subgrad[j + k] += w * h_cur[j]
        
        subgrad_centered = subgrad - np.mean(subgrad)
        subgrad_coeffs = np.fft.fft(subgrad_centered) / n
        
        coeffs -= lr * subgrad_coeffs
        
        h_test = np.real(np.fft.ifft(coeffs * n)) + 0.5
        h_test = _project_to_feasible(h_test, target_sum)
        val_test = _max_overlap_value(h_test)
        if val_test < best_val:
            best_val = val_test
            best_h = h_test.copy()
    
    return best_h, best_val

def _lp_minimize(h_init):
    """
    Structural change: Formulate as a sequence of linear programs.
    We minimize T such that h_i + h_j - h_i h_j <= T for all relevant pairs, 
    but since h_i h_j is non-linear, we use the fact that for fixed h, 
    the overlap is linear in h. 
    Actually, the overlap R_k = (2/N) sum h[i]h[i+k] is bilinear.
    However, if we fix the 'shifted' part, it's linear.
    A better LP approach: 
    Minimize T
    Subject to:
    sum h_i = N/2
    0 <= h_i <= 1
    For each k, sum_{i=0}^{N-k-1} h_i h_{i+k} <= T * N / 2
    
    This is still non-linear due to h_i h_{i+k}.
    However, we can use a linearization trick if we assume h_i are binary, but they are continuous.
    
    Alternative: Use the 'Soft Min' approach but with a much tighter approximation 
    and multiple restarts from diverse initial points, leveraging the 
    symmetric structure if possible, or just robust multi-start.
    
    Let's try a different structural change: 
    Instead of just polishing the spectral init, we perform a multi-start 
    optimization with diverse initial points (symmetric, random, sinusoidal) 
    using the soft-min objective, which is a smooth approximation of the 
    max-overlap objective. This is a different search strategy (multi-start 
    local optimization) compared to the single-path Fourier subgradient.
    """
    n = N
    target_sum = n / 2.0
    best_h = h_init.copy()
    best_val = _max_overlap_value(h_init)
    
    # Generate diverse initial points
    np.random.seed(42)
    inits = [h_init]
    
    # Symmetric sinusoids
    for freq in [1, 2, 3, 4]:
        x = np.linspace(0, 2*np.pi, n)
        h_sym = 0.5 + 0.4 * np.sin(freq * x)
        h_sym = _project_to_feasible(h_sym, target_sum)
        inits.append(h_sym)
        
    # Random with sum constraint
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, n)
        h_rand = _project_to_feasible(h_rand, target_sum)
        inits.append(h_rand)
        
    # Block patterns
    for block_size in [2, 4, 5, 10]:
        h_block = np.zeros(n)
        for i in range(n):
            h_block[i] = 0.0 if (i // block_size) % 2 == 0 else 1.0
        h_block = _project_to_feasible(h_block, target_sum)
        inits.append(h_block)
        
    # Optimize each
    for h0 in inits:
        if _time_remaining() < 15.0:
            break
        h_opt, v_opt = _soft_min_polish(h0, temps=[0.2, 0.1, 0.05, 0.02, 0.01])
        if v_opt < best_val:
            best_val = v_opt
            best_h = h_opt.copy()
            
    return best_h, best_val

def construct():
    t0 = time.time()
    target_sum = N / 2.0
    
    # Start with spectral initialization
    h = _spectral_init()
    best_h = h.copy()
    best_val = _max_overlap_value(h)
    
    # Phase 1: Fourier-space subgradient optimization (refinement machinery)
    if _time_remaining() > 20.0:
        h1, val1 = _fourier_optimize(h, max_iters=3000, lr_init=0.05)
        if val1 < best_val:
            best_val = val1
            best_h = h1.copy()
    
    # Phase 2: Multi-start local optimization with soft-min (structural change)
    if _time_remaining() > 30.0:
        h2, val2 = _lp_minimize(best_h)
        if val2 < best_val:
            best_val = val2
            best_h = h2.copy()
    
    # Final projection to ensure feasibility
    best_h = _project_to_feasible(best_h, target_sum)
    
    return best_h
