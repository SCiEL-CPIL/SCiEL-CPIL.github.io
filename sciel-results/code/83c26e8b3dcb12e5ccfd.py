"""Erdos minimum overlap - structural rewrite using FFT-based correlation and epigraph formulation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
_START_TIME = time.time()
_TIME_LIMIT = 138.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_all_corrs(h):
    """Compute all non-circular cross-correlations R(k) for k=1..n-1.
    R(k) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    """
    n = len(h)
    # Use zero-padded FFT for linear cross-correlation
    # autocorr via FFT: corr_full = ifft(fft(h, 2n) * conj(fft(h, 2n)))
    # This gives circular autocorr of the zero-padded signal, which equals linear autocorr
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    H = np.fft.rfft(h, nfft)
    corr_full = np.fft.irfft(H * np.conj(H), nfft)
    # corr_full[k] = sum_{i} h[i] * h[i+k] for k=0..n-1 (linear, with zero padding)
    # Actually for zero-padded autocorr: corr_full[k] = sum_{i=0}^{n-1-k} h[i]*h[i+k] for k=0..n-1
    # And corr_full[n-k] = sum h[i]*h[i-k] which is the same by symmetry
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k - 1] = corr_full[k] * (2.0 / n)
    return corrs

def _max_overlap(h):
    """Maximum overlap = 1 - min_k R(k)."""
    corrs = _compute_all_corrs(h)
    return 1.0 - np.min(corrs)

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    # Project onto simplex {x >= 0, sum = target_sum} first
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    w = np.clip(w, 0, 1)
    
    # Iterate to fix sum constraint with box constraints
    for _ in range(20):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-13:
            break
        if diff > 0:
            mask = w < 1.0 - 1e-12
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            mask = w > 1e-12
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    
    # Final exact adjustment
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-15:
        # Distribute to elements not at boundary
        mask = (w > 1e-10) & (w < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            w[mask] += diff / np.sum(mask)
            w = np.clip(w, 0, 1)
    return w

def _epigraph_obj_grad(x, t, beta=200.0):
    """Epigraph formulation: minimize t subject to t >= 1 - R_k for all k.
    We use a smooth approximation: minimize t + (1/beta)*sum_k max(0, 1 - R_k - t)^2
    
    Actually, let's use: minimize t + (1/beta) * sum_k softplus(beta*(1 - R_k - t))
    
    Simpler: use the soft-min approach directly.
    Objective: 1 - softmin_k(R_k)
    """
    n = len(x)
    corrs = _compute_all_corrs(x)
    
    # Soft min of corrs
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min = min_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    # Gradient
    weights = exp_terms / sum_exp  # softmax weights for the min
    
    # d(soft_min)/d(h[j]) = sum_k weights_k * d(R_k)/d(h[j])
    # R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    # d(R_k)/d(h[j]) = (2/n)*h[j+k] if j < n-k, plus (2/n)*h[j-k] if j >= k
    grad_soft_min = np.zeros(n)
    for k in range(1, n):
        w = weights[k - 1]
        if w > 1e-15:
            coeff = w * (2.0 / n)
            grad_soft_min[:n - k] += coeff * x[k:]
            grad_soft_min[k:] += coeff * x[:n - k]
    
    grad_obj = -grad_soft_min
    return obj, grad_obj

def _subgradient_polish(h, n_iter=3000, lr_init=0.005):
    """Subgradient ascent on min correlation with projection."""
    n = len(h)
    target_sum = n / 2.0
    best_h = h.copy()
    best_obj = _max_overlap(h)
    lr = lr_init
    
    for it in range(n_iter):
        if _time_remaining() < 3.0:
            break
        
        corrs = _compute_all_corrs(h)
        min_idx = np.argmin(corrs)
        
        # Subgradient of min correlation: gradient of R at the argmin k
        k = min_idx + 1
        grad = np.zeros(n)
        grad[:n - k] += (2.0 / n) * h[k:]
        grad[k:] += (2.0 / n) * h[:n - k]
        
        # Add contributions from nearby k values (active set approximation)
        for dk in range(-3, 4):
            if dk == 0:
                continue
            k2 = min_idx + dk
            if 0 <= k2 < n - 1:
                gap = corrs[min_idx] - corrs[k2]
                if gap < 0.05:
                    w = 0.3 * np.exp(-gap * 20)
                    grad[:n - k2] += w * (2.0 / n) * h[k2:]
                    grad[k2:] += w * (2.0 / n) * h[:n - k2]
        
        # Gradient ascent step
        h_new = h + lr * grad
        h_new = _project_to_simplex(h_new, target_sum)
        
        cur_obj = _max_overlap(h_new)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_h = h_new.copy()
        
        h = h_new
        # Decay learning rate
        lr = lr_init * (1.0 / (1.0 + it * 0.01))
    
    return best_h, best_obj

def _coordinate_descent(h, max_iter=5000):
    """Coordinate descent: for each coordinate, find optimal value given others.
    The objective is 1 - min_k R_k. We want to maximize min_k R_k.
    For a fixed coordinate j, R_k is linear in h[j] for each k.
    """
    n = len(h)
    target_sum = n / 2.0
    best_obj = _max_overlap(h)
    
    for it in range(max_iter):
        if _time_remaining() < 3.0:
            break
        
        improved = False
        for j in range(n):
            # Try a few candidate values for h[j]
            # The optimal h[j] maximizes min_k R_k where R_k depends linearly on h[j]
            # For each k, R_k = a_k * h[j] + b_k where:
            #   a_k = (2/n) * (h[j+k] if j < n-k else 0) + (2/n) * (h[j-k] if j >= k else 0)
            #   b_k = R_k - a_k * h[j]
            # We want to maximize min_k (a_k * x + b_k) over x in [0,1]
            # This is a 1D piecewise-linear maximization.
            
            corrs_current = _compute_all_corrs(h)
            min_corr = np.min(corrs_current)
            
            # Compute coefficients for each k
            a_coeffs = np.zeros(n - 1)
            for k in range(1, n):
                coeff = 0.0
                if j < n - k:
                    coeff += (2.0 / n) * h[j + k]
                if j >= k:
                    coeff += (2.0 / n) * h[j - k]
                a_coeffs[k - 1] = coeff
            
            # b_k = R_k - a_k * h[j]
            b_coeffs = corrs_current - a_coeffs * h[j]
            
            # Maximize min_k (a_k * x + b_k) over x in [0, 1]
            # Sample x at critical points and endpoints
            candidates = [0.0, 1.0, h[j]]
            # Add intersection points of pairs (expensive for large n, skip for now)
            # Just use a few quantiles
            for frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
                candidates.append(frac)
            
            best_x = h[j]
            best_min_corr = min_corr
            
            for x_cand in candidates:
                x_cand = np.clip(x_cand, 0.0, 1.0)
                if abs(x_cand - h[j]) < 1e-10:
                    continue
                new_corrs = a_coeffs * x_cand + b_coeffs
                new_min = np.min(new_corrs)
                if new_min > best_min_corr + 1e-10:
                    best_min_corr = new_min
                    best_x = x_cand
            
            if abs(best_x - h[j]) > 1e-10:
                h[j] = best_x
                improved = True
        
        # Project to maintain constraints
        h = _project_to_simplex(h, target_sum)
        cur_obj = _max_overlap(h)
        if cur_obj < best_obj:
            best_obj = cur_obj
        
        if not improved:
            break
    
    return h, best_obj

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = _max_overlap(best_x)
    
    # Generate diverse initializations
    np.random.seed(42)
    
    restarts = []
    restarts.append(np.full(n, 0.5))
    
    # Block structures
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        k = int(n * frac)
        if 0 < k < n:
            val1 = 2.0 * target_sum / k
            val2 = 2.0 * target_sum / (n - k)
            if val1 <= 1.0:
                restarts.append(np.concatenate([np.full(k, val1), np.full(n - k, 0.0)]))
            if val2 <= 1.0:
                restarts.append(np.concatenate([np.full(k, 0.0), np.full(n - k, val2)]))
    
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    
    # Random
    for _ in range(15):
        restarts.append(np.random.uniform(0, 1, n))
    
    # Smooth functions
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    x_grid = np.linspace(0, 2 * np.pi, n)
    for freq in [1, 2, 3]:
        for amp in [0.3, 0.4, 0.5]:
            restarts.append(np.clip(0.5 + amp * np.sin(freq * x_grid), 0, 1))
            restarts.append(np.clip(0.5 + amp * np.cos(freq * x_grid), 0, 1))
    
    # Sawtooth
    saw = np.array([i / n if i < n/2 else (n-i)/n for i in range(n)])
    restarts.append(saw)
    
    # Triangular
    tri = np.array([1.0 - 2.0*abs(i/n - 0.5) for i in range(n)])
    restarts.append(tri)
    
    # Optimize each restart
    for idx, x0 in enumerate(restarts):
        if _time_remaining() < 8.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        # L-BFGS-B with soft-min
        try:
            res = minimize(
                _epigraph_obj_grad,
                x,
                args=(200.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 150, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            res_x = _project_to_simplex(res.x, target_sum)
            res_obj = _max_overlap(res_x)
            if res_obj < best_obj:
                best_obj = res_obj
                best_x = res_x
        except Exception:
            pass
        
        # Polish with subgradient
        if _time_remaining() > 10.0:
            try:
                pg_x, pg_obj = _subgradient_polish(best_x, n_iter=2000, lr_init=0.003)
                if pg_obj < best_obj:
                    best_obj = pg_obj
                    best_x = pg_x
            except Exception:
                pass
    
    # Higher beta polish
    if _time_remaining() > 15.0:
        for beta in [300.0, 500.0, 800.0]:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(
                    _epigraph_obj_grad,
                    best_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = _max_overlap(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
            except Exception:
                pass
    
    # Final subgradient polish
    if _time_remaining() > 5.0:
        try:
            pg_x, pg_obj = _subgradient_polish(best_x, n_iter=5000, lr_init=0.002)
            if pg_obj < best_obj:
                best_obj = pg_obj
                best_x = pg_x
        except Exception:
            pass
    
    # Coordinate descent polish
    if _time_remaining() > 5.0:
        try:
            cd_x, cd_obj = _coordinate_descent(best_x, max_iter=3000)
            if cd_obj < best_obj:
                best_obj = cd_obj
                best_x = cd_x
        except Exception:
            pass
    
    # Final subgradient
    if _time_remaining() > 3.0:
        try:
            pg_x, pg_obj = _subgradient_polish(best_x, n_iter=3000, lr_init=0.001)
            if pg_obj < best_obj:
                best_obj = pg_obj
                best_x = pg_x
        except Exception:
            pass
    
    # Ensure exact sum
    best_x = _project_to_simplex(best_x, target_sum)
    
    return best_x
