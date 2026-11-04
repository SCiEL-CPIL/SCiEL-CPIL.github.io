"""Convex L1 spectral minimization via projected gradient descent."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]

def _l1_autoconv(power):
    """Compute L1 norm of autoconvolution from power spectrum."""
    conv = np.fft.irfft(power, _FFT_SIZE)
    return np.sum(np.abs(conv[:2 * N - 1]))

def _project_power_spectrum(P, S):
    """
    Project power spectrum P onto the set of power spectra with sum S.
    """
    total = np.sum(P)
    if total < 1e-12:
        return np.ones_like(P) * (S / len(P))
    P_scaled = P * (S / total)
    P_scaled = np.maximum(P_scaled, 0)
    total2 = np.sum(P_scaled)
    if total2 < 1e-12:
        return np.ones_like(P) * (S / len(P))
    P_scaled = P_scaled * (S / total2)
    return P_scaled

def _solve_convex_l1(deadline):
    """
    Minimize L1(autoconv) subject to sum(power) = S.
    Using projected gradient descent on the power spectrum.
    """
    K = _FFT_SIZE // 2 + 1
    S = 1.0  # Fixed DC component (sum of x)
    
    # Initialize with flat spectrum
    P = np.ones(K) * (S / K)
    
    # Step size
    alpha = 0.01
    best_P = P.copy()
    best_val = _l1_autoconv(P)
    
    for i in range(5000):
        if time.time() > deadline:
            break
        
        conv = np.fft.irfft(P, _FFT_SIZE)
        conv = conv[:2 * N - 1]
        
        # Subgradient of L1 norm
        subgrad = np.sign(conv)
        
        # Gradient in frequency domain
        grad_P = np.fft.rfft(subgrad, _FFT_SIZE)[:K]
        
        # Update
        P = P - alpha * grad_P
        
        # Project
        P = _project_power_spectrum(P, S)
        
        # Check improvement
        val = _l1_autoconv(P)
        if val < best_val:
            best_val = val
            best_P = P.copy()
            
        # Adaptive step size
        if val > best_val * 1.1:
            alpha *= 0.9
        else:
            alpha = min(alpha * 1.01, 0.05)
            
    # Convert best_P to x
    x = np.fft.irfft(np.sqrt(best_P), _FFT_SIZE)[:N]
    
    # Normalize x to have sum S
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x * (S / s)
        
    return x, _metric(x)

def _try_local_search_fast(x, deadline, rng, max_rounds=100000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.2
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 150:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        k = int(rng.integers(1, 5))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(2):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val

def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(30):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.05, 0.05
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(10):
                if time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x1 = best_x.copy()
                x1[idx] += m1
                v1 = _metric(x1)
                
                x2 = best_x.copy()
                x2[idx] += m2
                v2 = _metric(x2)
                
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
                
                if v1 < best_v:
                    best_v = v1
                    best_delta = m1
                if v2 < best_v:
                    best_v = v2
                    best_delta = m2
            
            if best_delta != 0.0 and best_v < best_val:
                best_x[idx] += best_delta
                best_val = best_v
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0
    
    rng = np.random.default_rng(2024)
    
    # Phase 1: Convex L1 Spectral Optimization
    x_spec, val_spec = _solve_convex_l1(deadline - 10)
    
    if x_spec is not None and val_spec < float('inf'):
        best_x = x_spec
        best_val = val_spec
    else:
        # Fallback to ones
        best_x = np.ones(N)
        best_val = _metric(best_x)
    
    # Phase 2: Local Search
    if time.time() < deadline - 15:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 15, rng, max_rounds=50000)
    
    # Phase 3: Final polish
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
# EVOLVE-BLOCK-END
