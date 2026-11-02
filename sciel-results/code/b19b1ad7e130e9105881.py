"""Signed step function optimization: minimize max|autoconv| / (sum)^2 via spectral methods."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400
L = 2 * N - 1  # autoconvolution length
L_fft = 1
while L_fft < L:
    L_fft *= 2

START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute autoconvolution via FFT."""
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max|autoconv| / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_grad(abs_ac, beta=50.0):
    """Compute soft max and its gradient w.r.t. abs_ac."""
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        smooth_max = max_val
        weights = np.ones_like(abs_ac) / len(abs_ac)
    else:
        shifted = abs_ac - max_val
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        smooth_max = (max_val + np.log(sum_exp)) / beta
        weights = exp_shifted / sum_exp
    return smooth_max, weights

def _time_domain_obj_grad(x, beta=50.0):
    """Objective and gradient in time domain for continuous x."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    smooth_max, weights = _soft_max_grad(abs_ac, beta)
    val = smooth_max / (s * s)
    
    # Gradient: d(smooth_max)/dx = sum_j weights_j * sign(ac[j]) * d(ac[j])/dx
    # ac[j] = sum_i x[i] x[i+j-N+1] (for valid indices)
    # d(ac[j])/dx[k] = x[k-j+N-1] + x[k+j-N+1] (with appropriate boundary handling)
    
    # Using FFT approach for gradient
    c = weights * np.sign(ac)
    c_pad = np.zeros(L_fft)
    c_pad[:L] = c
    
    # grad_smooth = ifft(conj(fft(x)) * fft(c)) + ifft(fft(x) * fft(c))
    # Actually: d(ac)/dx = x * c (circular) + x * c (shifted)
    # More precisely using FFT:
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.rfft(c_pad)
    
    # ac = ifft(X^2), d(ac[j])/dx[k] involves convolution
    # grad[j] = sum_k (d(ac[k])/dx[j]) * weights_k * sign(ac[k])
    # = sum_k c[k] * (x[k-j] + x[j-k]) (circular)
    # In FFT: grad = ifft(conj(X) * C) + ifft(X * C) ... let me think more carefully
    
    # ac = x * x (linear convolution), padded
    # d(ac)/dx = x * (delta) + (delta) * x where delta is the perturbation
    # For gradient: grad[k] = sum_j c[j] * d(ac[j])/dx[k]
    # d(ac[j])/dx[k] = x[j-k] + x[k-j] (with linear conv indexing)
    
    # Using FFT: if we define c as the "gradient signal" in autoconv domain
    # grad = ifft(conj(FFT(x)) * FFT(c)) + ifft(FFT(x) * FFT(c))
    # But this isn't quite right for linear convolution.
    
    # Let me use a direct approach with FFT:
    # ac[j] = sum_i x[i]*x[i+(j-N+1)] for j in [0, L-1]
    # Let's define the correlation properly.
    
    # Actually, let's use the property:
    # If ac = ifft(|X|^2) where X = fft(x, L_fft), then
    # d(ac)/dx in circular sense: grad_circ = ifft(conj(X) * fft(c))
    # For linear convolution, we need to be careful.
    
    # Simpler: use the fact that for linear convolution ac = x * x (padded),
    # the gradient is: grad = x * c + flip(x) * c (where * is linear conv)
    # In FFT domain with sufficient padding:
    L_conv = L
    L_fft2 = 1
    while L_fft2 < 2 * L_conv:
        L_fft2 *= 2
    
    x_pad = np.zeros(L_fft2)
    x_pad[:N] = x
    c_pad2 = np.zeros(L_fft2)
    c_pad2[:L] = c
    
    X2 = np.fft.rfft(x_pad)
    C2 = np.fft.rfft(c_pad2)
    
    # x * c (linear conv)
    conv1 = np.fft.irfft(X2 * C2, n=L_fft2)[:N]
    # flip(x) * c
    x_flip = x[::-1]
    x_flip_pad = np.zeros(L_fft2)
    x_flip_pad[:N] = x_flip
    X2f = np.fft.rfft(x_flip_pad)
    conv2 = np.fft.irfft(X2f * C2, n=L_fft2)[:N]
    
    grad_smooth = conv1 + conv2
    
    # d(val)/dx = grad_smooth / s^2 - 2 * smooth_max * (sum x) / s^3
    # = grad_smooth / s^2 - 2 * smooth_max / s^2
    grad = grad_smooth / (s * s) - 2.0 * smooth_max / (s * s * s) * np.ones(N)
    
    return val, grad

def _quantize_to_signed(x, n_levels=2):
    """Quantize continuous signal to signed step function with n_levels levels."""
    if n_levels == 2:
        # Binary: ±1
        threshold = np.median(x)
        q = np.sign(x - threshold)
        q[q == 0] = 1
        return q
    elif n_levels == 3:
        # Three levels: -1, 0, 1
        q1 = np.percentile(x, 33)
        q2 = np.percentile(x, 67)
        q = np.zeros(N)
        q[x < q1] = -1
        q[x > q2] = 1
        return q
    else:
        # General: map to [-1, 1] with n_levels discrete values
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < 1e-10:
            return np.ones(N)
        x_norm = (x - x_min) / (x_max - x_min)
        levels = np.linspace(-1, 1, n_levels)
        q = np.zeros(N)
        for i in range(n_levels):
            lo = (i - 0.5) / n_levels if i > 0 else 0
            hi = (i + 0.5) / n_levels if i < n_levels - 1 else 1
            q[(x_norm >= lo) & (x_norm < hi)] = levels[i]
        q[x_norm >= 1.0] = levels[-1]
        return q

def _local_search_signed(x, max_iters=50000, beta=50.0):
    """Local search on ±1 sign patterns using single-bit and pair-bit flips."""
    best_x = x.copy()
    best_val = _objective(best_x)
    rng = np.random.default_rng(42)
    
    # Precompute autoconv for current x
    ac = _autoconv_fft(best_x)
    abs_ac = np.abs(ac)
    s = np.sum(best_x)
    max_ac = np.max(abs_ac)
    
    stall_count = 0
    for it in range(max_iters):
        if _check_deadline():
            break
        
        if stall_count > 2000:
            break
        
        # Try single-bit flip
        idx = rng.integers(0, N)
        x_trial = best_x.copy()
        x_trial[idx] *= -1
        
        # Update autoconv incrementally
        # ac_new[j] = ac[j] - 2*x[idx]*x[j-idx+N-1] + 2*x[idx]*x[j-idx+N-1] ... 
        # Actually flipping x[idx] from +1 to -1:
        # ac_new = ac - 2*x[idx]*x * e_idx - 2*e_idx*x*x[idx] + 2*e_idx*x*e_idx*x[idx]
        # This is complex; let's just recompute for simplicity with N=400
        val_trial = _objective(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
            s = np.sum(best_x)
            max_ac = np.max(abs_ac)
            stall_count = 0
        else:
            stall_count += 1
    
    return best_x, best_val

def _pair_flip_search(x, max_iters=30000):
    """Local search with pair-bit flips for ±1 signals."""
    best_x = x.copy()
    best_val = _objective(best_x)
    rng = np.random.default_rng(123)
    
    stall_count = 0
    for it in range(max_iters):
        if _check_deadline():
            break
        
        if stall_count > 3000:
            break
        
        i, j = rng.integers(0, N, size=2)
        if i == j:
            continue
        
        x_trial = best_x.copy()
        x_trial[i] *= -1
        x_trial[j] *= -1
        
        val_trial = _objective(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial
            stall_count = 0
        else:
            stall_count += 1
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Optimize continuous signal in time domain
    rng = np.random.default_rng(42)
    
    # Multiple random initializations
    inits = [
        np.ones(N),
        np.sin(2 * np.pi * np.arange(N) / N * 3),  # 3 cycles
        np.sin(2 * np.pi * np.arange(N) / N * 5),  # 5 cycles
        np.cos(2 * np.pi * np.arange(N) / N * 2),  # 2 cycles
        rng.normal(0, 1, N),
        rng.normal(0, 1, N),
    ]
    
    for x0 in inits:
        if _check_deadline():
            break
        
        for beta in [30.0, 50.0, 100.0]:
            if _check_deadline():
                break
            
            def _obj_grad(x, b=beta):
                return _time_domain_obj_grad(x, b)
            
            try:
                result = minimize(_obj_grad, x0, method='L-BFGS-B',
                                 jac=True, options={'maxiter': 200, 'gtol': 1e-5})
                if result.fun < best_val:
                    val_exact = _objective(result.x)
                    if val_exact < best_val:
                        best_val = val_exact
                        best_x = result.x.copy()
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Quantize to signed step functions and try local search
    quantized_candidates = []
    
    for n_levels in [2, 3]:
        x_q = _quantize_to_signed(best_x, n_levels)
        val_q = _objective(x_q)
        quantized_candidates.append((val_q, x_q))
    
    # Also try different threshold-based quantizations
    for pct in [25, 33, 40, 50, 60, 75]:
        threshold = np.percentile(best_x, pct)
        x_q = np.sign(best_x - threshold)
        x_q[x_q == 0] = 1
        val_q = _objective(x_q)
        quantized_candidates.append((val_q, x_q))
    
    # Select best quantized
    if quantized_candidates:
        quantized_candidates.sort(key=lambda t: t[0])
        best_q_val, best_q_x = quantized_candidates[0]
        if best_q_val < best_val:
            best_val = best_q_val
            best_x = best_q_x
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Local search on the best quantized signal
    if _check_deadline():
        return best_x
    
    # Single-bit flip search
    x_ls, val_ls = _local_search_signed(best_x, max_iters=50000)
    if val_ls < best_val:
        best_val = val_ls
        best_x = x_ls
    
    if _check_deadline():
        return best_x
    
    # Pair-bit flip search
    x_ls2, val_ls2 = _pair_flip_search(best_x, max_iters=30000)
    if val_ls2 < best_val:
        best_val = val_ls2
        best_x = x_ls2
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Try a few more random restarts with quantization
    for restart in range(5):
        if _check_deadline():
            break
        
        x_rand = rng.normal(0, 1, N)
        # Quick optimization
        try:
            result = minimize(lambda x: _time_domain_obj_grad(x, 50.0), x_rand,
                             method='L-BFGS-B', jac=True,
                             options={'maxiter': 100, 'gtol': 1e-4})
            x_q = _quantize_to_signed(result.x, 2)
            val_q = _objective(x_q)
            
            if val_q < best_val:
                # Refine with local search
                x_ls, val_ls = _local_search_signed(x_q, max_iters=20000)
                if val_ls < best_val:
                    best_val = val_ls
                    best_x = x_ls
        except Exception:
            pass
    
    return best_x
