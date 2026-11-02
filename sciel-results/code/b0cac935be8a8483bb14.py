"""Spectral-domain optimization: PAPR minimization via alternating phase/amplitude + L-BFGS polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _fft_len(n):
    L = 1
    while L < n:
        L *= 2
    return L

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = _fft_len(L)
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _compute_objective(x):
    """Compute the raw objective: max|ac| / (sum x)^2"""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _topk_peak_grad_descent(x_init, iters=500, top_k=5, step_size=0.1):
    """
    Directly minimize the top-k peaks of the autoconvolution using sub-gradient descent.
    """
    x = x_init.copy()
    best_x = x.copy()
    best_val = _compute_objective(x)
    
    # Normalize sum to 1 initially
    s = np.sum(x)
    if abs(s) > 1e-10:
        x = x / s
        best_x = x.copy()
        best_val = _compute_objective(x)

    # Pre-compute indices for speed? No, just vectorize the gradient calc.
    j_indices = np.arange(N)
    
    for i in range(iters):
        if _check_deadline():
            break
            
        c = _autoconv_fft(x)
        abs_c = np.abs(c)
        
        # Get top-k peaks
        top_indices = np.argsort(abs_c)[::-1][:top_k]
        
        # Compute sub-gradient
        grad_x = np.zeros(N)
        
        for k in top_indices:
            sign_c = np.sign(c[k])
            if sign_c == 0:
                continue
            
            # term1: x[k-j] for valid j
            idx1 = k - j_indices
            mask1 = (idx1 >= 0) & (idx1 < N)
            if np.any(mask1):
                grad_x[mask1] += sign_c * x[idx1[mask1]]
                
            # term2: x[j-k] for valid j
            idx2 = j_indices - k
            mask2 = (idx2 >= 0) & (idx2 < N)
            if np.any(mask2):
                grad_x[mask2] += sign_c * x[idx2[mask2]]
                
        # Normalize gradient to have unit norm? Or fixed step?
        # Fixed step is simpler.
        # Adaptive step: reduce if stuck
        x_new = x - step_size * grad_x
        
        # Project to sum=1
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-10:
            x_new = x_new / s_new
        else:
            continue
            
        val = _compute_objective(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
            x = x_new.copy()
            # Increase step slightly if improved?
            # step_size = min(0.5, step_size * 1.01)
        else:
            # Decrease step if no improvement
            step_size = max(0.01, step_size * 0.95)
            # Revert x to best_x? Or keep x_new for exploration?
            # Keep x_new to allow random walk, but track best_x separately.
            # Actually, reverting to best_x is safer for convergence.
            # But gradient descent usually moves in a direction.
            # Let's keep x_new.
            
    return best_x

def _spectral_init(rng, mode=0):
    """Generate diverse spectral initializations."""
    L = 2 * N - 1
    L_fft = _fft_len(L)
    
    if mode == 0:
        # Flat spectrum
        mag = np.ones(L_fft // 2 + 1)
        freqs = np.fft.rfftfreq(L_fft)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        # Raised cosine
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        # Low-pass
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-5.0 * (freqs * L_fft / 2) ** 2)
    elif mode == 3:
        # Bandpass
        freqs = np.fft.rfftfreq(L_fft)
        f = np.abs(freqs * L_fft / 2)
        mag = np.exp(-2.0 * (f - 0.3) ** 2)
    else:
        # Random smooth spectrum
        mag = 1.0 + 0.5 * rng.normal(0, 1, size=L_fft // 2 + 1)
        # Smooth it
        from scipy.ndimage import uniform_filter1d
        mag = uniform_filter1d(mag, size=10)
        mag = np.maximum(0.1, mag)
    
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0.0
    if L_fft % 2 == 0:
        phase[L_fft // 2] = 0.0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x = np.fft.irfft(spectrum, n=L_fft)[:N]
    s = np.sum(x)
    if abs(s) > 1e-10:
        x = x / s
    return x

def _lbfgs_polish(x_init, beta, maxiter=200):
    """L-BFGS polish with given beta."""
    try:
        def obj_and_grad(x, _beta=beta):
            return _objective_and_grad(x, _beta)
        
        result = minimize(obj_and_grad, x_init, method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12})
        return result.x, result.fun
    except Exception:
        return x_init, _compute_objective(x_init)

def _coordinate_polish(x, max_iter=5000):
    """Coordinate descent polish with adaptive step size."""
    best_x = x.copy()
    best_val = _compute_objective(x)
    step_size = 0.02
    no_improve = 0
    rng = np.random.default_rng(123)
    
    for _ in range(max_iter):
        if _check_deadline():
            break
        if no_improve > 2000:
            break
        
        # Perturb a small random subset
        n_perturb = rng.integers(1, 6)
        indices = rng.choice(N, size=n_perturb, replace=False)
        perturbation = np.zeros(N)
        perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
        x_trial = best_x + perturbation
        
        val = _compute_objective(x_trial)
        if val < best_val - 1e-12:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.002, step_size * 0.97)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve % 100 == 0:
                step_size = min(0.05, step_size * 1.03)
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _compute_objective(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initial_points = [np.ones(N)]
    
    # Spectral initializations with various spectra
    for mode in range(5):
        for _ in range(3):
            if _check_deadline():
                break
            initial_points.append(_spectral_init(rng, mode))
    
    # Some time-domain structured initializations
    t = np.linspace(0, 1, N)
    initial_points.extend([
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        np.exp(1.0 * t),
        np.exp(-1.0 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5)**2,
        1.0 + 0.5 * np.sin(6 * np.pi * t) * np.cos(2 * np.pi * t),
    ])
    
    # Random initializations
    for _ in range(5):
        if _check_deadline():
            break
        initial_points.append(rng.normal(1.0, 0.2, size=N) * rng.uniform(0.5, 1.5))
    
    candidates = []
    
    # Phase 1: Top-K Peak Gradient Descent from each initial point
    for x0 in initial_points:
        if _check_deadline():
            break
        
        try:
            x_opt = _topk_peak_grad_descent(x0, iters=200, top_k=5, step_size=0.1)
            val = _compute_objective(x_opt)
            candidates.append((val, x_opt))
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Refine top candidates with more iterations and different top_k
    candidates.sort(key=lambda item: item[0])
    top_candidates = [x for _, x in candidates[:10]]
    
    for x_init in top_candidates:
        if _check_deadline():
            break
        
        # Try different top_k values
        for top_k in [3, 10, 20]:
            if _check_deadline():
                break
            try:
                x_opt = _topk_peak_grad_descent(x_init, iters=300, top_k=top_k, step_size=0.05)
                val = _compute_objective(x_opt)
                candidates.append((val, x_opt))
                if val < best_val:
                    best_val = val
                    best_x = x_opt.copy()
            except Exception:
                pass
    
    # Phase 3: Coordinate descent polish on the best
    if not _check_deadline():
        x_cd, val_cd = _coordinate_polish(best_x, max_iter=4000)
        if val_cd < best_val:
            best_val = val_cd
            best_x = x_cd.copy()
            
    # Phase 4: Final Top-K polish
    if not _check_deadline():
        try:
            x_final = _topk_peak_grad_descent(best_x, iters=500, top_k=5, step_size=0.01)
            val_final = _compute_objective(x_final)
            if val_final < best_val:
                best_val = val_final
                best_x = x_final.copy()
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
