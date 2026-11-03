import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv_lin(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    conv = np.fft.irfft(X * X.conj(), size)
    return conv[:2 * n - 1]

def _metric_offdiag(x):
    """Max |c_k| for k != 0, divided by (sum x)^2."""
    c = _autoconv_lin(x)
    n = len(x)
    off1 = c[:n-1]
    off2 = c[n:]
    off = np.concatenate([off1, off2])
    max_abs = np.max(np.abs(off)) if len(off) > 0 else 0.0
    s = np.sum(x)
    if abs(s) < 1e-12:
        return max_abs
    return max_abs / (s * s)

def _soft_max_abs(v, alpha=10.0):
    """Smooth approximation of max(|v|) using log-sum-exp."""
    abs_v = np.abs(v)
    if len(abs_v) == 0:
        return 0.0
    max_val = np.max(abs_v)
    if max_val < 1e-15:
        return 0.0
    # Shift for numerical stability
    shifted = abs_v - max_val
    log_sum = np.log(np.sum(np.exp(alpha * shifted))) / alpha
    return max_val + log_sum

def _spectral_objective(params, alpha=10.0):
    """
    Objective function in spectral domain.
    params: real vector of length N/2+1 (rfft output magnitudes)
    We reconstruct x from magnitudes with alternating signs to maximize |sum x|.
    """
    # params are the magnitudes of the rfft
    mags = np.abs(params)
    n = len(mags)
    # Full FFT size for reconstruction
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    
    # Reconstruct X from magnitudes
    # Use a phase pattern that maximizes the DC component (sum of x)
    # Simple approach: all positive phases (real, non-negative x)
    X = np.zeros(fft_size // 2 + 1, dtype=complex)
    X[:n] = mags
    
    # Inverse FFT to get x
    x = np.fft.irfft(X, fft_size)
    x = x[:N]
    
    # Compute autoconvolution
    c = _autoconv_lin(x)
    off1 = c[:N-1]
    off2 = c[N:]
    off = np.concatenate([off1, off2])
    
    # Soft max of |off|
    soft_max = _soft_max_abs(off, alpha)
    
    # Sum of x
    s = np.sum(x)
    if abs(s) < 1e-12:
        return soft_max + 1e6  # Penalty for zero sum
    
    return soft_max / (s * s)

def _spectral_design_optimize(deadline, seed=42):
    """
    Optimize the spectral magnitude profile to minimize the metric.
    Returns the best x found.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize with random magnitudes
    n_mags = N // 2 + 1
    x0 = rng.uniform(0.5, 1.5, size=n_mags)
    
    # L-BFGS-B optimization
    best_x = None
    best_score = np.inf
    
    # Multiple random restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            x0 = rng.uniform(0.5, 1.5, size=n_mags)
        else:
            # Perturb previous best
            if best_x is not None:
                # Get current best magnitudes
                X_best = np.fft.rfft(best_x, 2*N-1)
                mags_best = np.abs(X_best[:n_mags])
                x0 = mags_best * rng.uniform(0.8, 1.2, size=n_mags)
            else:
                x0 = rng.uniform(0.5, 1.5, size=n_mags)
        
        # Run optimization
        try:
            result = minimize(
                _spectral_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500}
            )
            
            if result.success and result.fun < best_score:
                # Reconstruct x from optimal magnitudes
                mags_opt = np.abs(result.x)
                X_opt = np.zeros(N // 2 + 1, dtype=complex)
                X_opt[:] = mags_opt
                
                fft_size = 1
                while fft_size < 2 * N - 1:
                    fft_size *= 2
                
                x_opt = np.fft.irfft(X_opt, fft_size)
                x_opt = x_opt[:N]
                
                # Normalize to have reasonable magnitude
                s = np.sum(x_opt)
                if abs(s) > 1e-12:
                    x_opt = x_opt / np.abs(s) * np.sign(s)
                
                score = _metric_offdiag(x_opt)
                if score < best_score:
                    best_score = score
                    best_x = x_opt.copy()
                    
        except Exception:
            continue
    
    return best_x, best_score

def _try_local_search(x, deadline):
    """Local search to polish the solution."""
    best_x = x.copy()
    best_score = _metric_offdiag(best_x)
    cur_x = x.copy()
    cur_score = best_score
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
            cur_x[i] *= -1
            new_score = _metric_offdiag(cur_x)
            if new_score < cur_score:
                cur_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_x = cur_x.copy()
            else:
                cur_x[i] *= -1
    
    return best_x, best_score

def construct():
    deadline = time.time() + 110.0  # Safety margin
    
    # Phase 1: Spectral design optimization
    best_x, best_score = _spectral_design_optimize(deadline - 20.0, seed=42)
    
    if best_x is None:
        # Fallback to random
        rng = np.random.default_rng(42)
        best_x = rng.choice([-1, 1], size=N)
        best_score = _metric_offdiag(best_x)
    
    # Phase 2: Local search polish
    remaining = deadline - time.time()
    if remaining > 5.0:
        best_x, best_score = _try_local_search(best_x, deadline - 5.0)
    
    # Phase 3: More spectral restarts if time remains
    remaining = deadline - time.time()
    if remaining > 15.0:
        # Try different alpha values for soft max
        for alpha in [5.0, 20.0, 50.0]:
            if time.time() > deadline - 10.0:
                break
            
            # Modify objective to use this alpha
            def obj_with_alpha(params):
                return _spectral_objective(params, alpha=alpha)
            
            n_mags = N // 2 + 1
            rng = np.random.default_rng(123 + int(alpha))
            x0 = rng.uniform(0.5, 1.5, size=n_mags)
            
            try:
                result = minimize(
                    obj_with_alpha,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 300}
                )
                
                if result.success:
                    mags_opt = np.abs(result.x)
                    X_opt = np.zeros(N // 2 + 1, dtype=complex)
                    X_opt[:] = mags_opt
                    
                    fft_size = 1
                    while fft_size < 2 * N - 1:
                        fft_size *= 2
                    
                    x_opt = np.fft.irfft(X_opt, fft_size)
                    x_opt = x_opt[:N]
                    
                    s = np.sum(x_opt)
                    if abs(s) > 1e-12:
                        x_opt = x_opt / np.abs(s) * np.sign(s)
                    
                    score = _metric_offdiag(x_opt)
                    if score < best_score:
                        best_score = score
                        best_x = x_opt.copy()
                        
            except Exception:
                continue
        
        # Final local polish
        remaining = deadline - time.time()
        if remaining > 3.0:
            best_x, best_score = _try_local_search(best_x, deadline - 1.0)
    
    return best_x.astype(float)
