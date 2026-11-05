"""Structural rewrite: smooth-max surrogate + projected gradient on DCT basis + FFT autoconv.

Key structural changes from parent:
1. Use FFT-based autoconvolution (O(N log N)) instead of np.convolve (O(N^2)) for speed.
2. Use a smooth-max surrogate (log-sum-exp) to get differentiable objective for gradient methods.
3. Parameterize h as: h = base + B @ coeffs, where B is DCT basis matrix (first K modes).
4. Use projected gradient descent with adaptive step size on the smooth surrogate.
5. Multiple random restarts in DCT coefficient space.
6. Final polish with L-BFGS-B on the smooth objective, then evaluate exact value.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct, idct

N = 400
_t0 = time.time()
_DEADLINE = 135.0  # safety margin under 150s

def _value_exact(h):
    """Compute the exact task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_autoconv(h):
    """Compute full autoconvolution via FFT (faster than np.convolve for large N)."""
    n = len(h)
    size = 2 * n - 1
    # Next power of 2 for efficient FFT
    fsize = 1
    while fsize < size:
        fsize *= 2
    F = np.fft.rfft(h, n=fsize)
    conv = np.fft.irfft(F * F, n=fsize)[:size]
    return conv

def _smooth_max_value(h, temperature=0.1):
    """Smooth surrogate for the task value using log-sum-exp for max."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if abs(s) < 1e-15 or not np.all(np.isfinite(h)):
        return 1e10
    conv = _fft_autoconv(h)
    # Smooth max via log-sum-exp
    max_c = np.max(conv)
    shifted = (conv - max_c) / temperature
    log_sum_exp = max_c + temperature * np.log(np.sum(np.exp(shifted)))
    val = abs(2.0 * N * log_sum_exp / (s * s))
    return val

def _smooth_max_value_grad(h, temperature=0.1):
    """Compute smooth max value and its gradient w.r.t. h."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if abs(s) < 1e-15 or not np.all(np.isfinite(h)):
        return 1e10, np.zeros_like(h)
    
    conv = _fft_autoconv(h)
    max_c = np.max(conv)
    shifted = (conv - max_c) / temperature
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum_exp = max_c + temperature * np.log(sum_exp)
    
    val = abs(2.0 * N * log_sum_exp / (s * s))
    
    # Gradient of log_sum_exp w.r.t. conv elements
    # d(log_sum_exp)/d(conv[j]) = exp_shifted[j] / sum_exp
    weights = exp_shifted / sum_exp
    
    # Gradient of log_sum_exp w.r.t. h
    # conv[j] = sum_i h[i] * h[j-i] (for valid indices)
    # d(conv[j])/d(h[k]) = sum_i [delta_{k,i} * h[j-i] + h[i] * delta_{k,j-i}]
    # = h[j-k] + h[k-j] (with appropriate boundary handling)
    # This is the cross-correlation of h with itself at lag k, which is conv itself shifted.
    # Actually: d(conv[j])/d(h[k]) = h[j-k] if 0<=j-k<N, plus h[k-j] if 0<=k-j<N
    # This equals (h * h)[k] in the correlation sense.
    
    # More carefully: conv = full convolution of h with h.
    # conv[j] = sum_{i} h[i] * h[j-i] where h is zero-padded outside [0,N-1]
    # d(conv[j])/d(h[k]) = h[j-k] + h[k-j] (both terms contribute)
    # = 2*h[j-k] if j != k, h[j-k] if j==k... no.
    # Actually for full convolution: conv[j] = sum_i h[i]*h[j-i]
    # d/dh[k]: term where i=k gives h[j-k]; term where j-i=k i.e. i=j-k gives h[j-k]
    # So d(conv[j])/d(h[k]) = h[j-k] + h[j-k] = 2*h[j-k] when both i=k and i=j-k are valid
    # But if k == j-k (i.e., j=2k), then it's just h[j-k] (same term counted once)
    
    # Simpler approach: use the fact that gradient of sum_j w_j * conv[j] w.r.t. h
    # is 2 * (w * h) where * is circular convolution (for the full conv case, it's a bit different)
    
    # For full convolution: grad_h[k] = sum_j weights[j] * d(conv[j])/d(h[k])
    # = sum_j weights[j] * (h[j-k] + h[k-j]) with boundary conditions
    # = sum_j weights[j] * h[j-k] + sum_j weights[j] * h[k-j]
    # Let l = j-k: first sum = sum_l weights[l+k] * h[l]
    # Let l = k-j: second sum = sum_l weights[k-l] * h[l]
    # So grad_h[k] = sum_l h[l] * (weights[l+k] + weights[k-l])
    # This is: grad_h = h * (weights reversed + weights) in some sense
    # Actually it's the convolution of h with (weights + weights_reversed)
    
    # Let me use a different approach: just compute it directly for correctness
    # grad_h[k] = sum_j weights[j] * (h[j-k] if 0<=j-k<N else 0 + h[k-j] if 0<=k-j<N else 0)
    
    # Use FFT for efficiency
    # grad_h = ifft(fft(h) * fft(weights_extended)) where weights_extended handles the boundary
    # Actually, let's just do it with the correlation formula
    
    # The gradient of the smooth max of conv w.r.t. h is:
    # grad_h = 2 * (weights ⊛ h) where ⊛ is the appropriate correlation
    # For full convolution, it's: grad_h[k] = sum_j w_j * h[j-k] + sum_j w_j * h[k-j]
    
    # Let me construct this using FFT
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < 2 * size:
        fsize *= 2
    
    # Pad weights to size
    w_padded = np.zeros(fsize)
    w_padded[:size] = weights
    
    # h_padded
    h_padded = np.zeros(fsize)
    h_padded[:n] = h
    
    # conv of h with reversed weights gives one part
    # Actually, let's think again.
    # sum_j w_j * h[j-k] for fixed k: this is (w * h)[k] in the full convolution sense
    # where we flip one of them. Actually it's the cross-correlation.
    
    # Let me just compute it directly since N=400 is small enough
    # grad_h[k] = sum_{j=0}^{2N-2} w_j * (h[j-k] if 0<=j-k<N else 0) + (h[k-j] if 0<=k-j<N else 0)
    
    # For speed, use the fact that this is a convolution
    # Part 1: sum_j w_j * h[j-k] = (w_flip * h)[k] where w_flip is w reversed and padded
    # Part 2: sum_j w_j * h[k-j] = (w * h)[k] (standard convolution)
    
    # So grad_h = conv(w, h) + conv(w_flip, h) where conv is full convolution
    
    w_flip = weights[::-1]
    
    # Use FFT for these convolutions
    def _full_conv(a, b):
        na, nb = len(a), len(b)
        size_c = na + nb - 1
        fs = 1
        while fs < size_c:
            fs *= 2
        A = np.fft.rfft(a, n=fs)
        B = np.fft.rfft(b, n=fs)
        return np.fft.irfft(A * B, n=fs)[:size_c]
    
    part1 = _full_conv(weights, h)  # sum_j w_j * h[k-j]... wait, need to be careful
    part2 = _full_conv(w_flip, h)
    
    # Actually, let me just verify with a small example and use the direct formula
    # For now, use a simpler but correct approach:
    # The Jacobian of conv w.r.t. h is a (2N-1) x N matrix J where J[j,k] = d(conv[j])/d(h[k])
    # J[j,k] = 1 if j-k is in [0,N-1] then h[j-k]... no, J[j,k] = h[j-k] if 0<=j-k<N, plus h[k-j] if 0<=k-j<N
    # But we want grad_h[k] = sum_j w_j * J[j,k]
    
    # Let me just compute it directly - N=400, size=799, so 400*799 ~ 320k operations, fine
    grad_h = np.zeros(N)
    for k in range(N):
        # sum_j w_j * (h[j-k] if 0<=j-k<N else 0)
        # j-k in [0,N-1] => j in [k, k+N-1]
        j_lo = max(0, k)
        j_hi = min(2*N-2, k+N-1)
        if j_lo <= j_hi:
            idx = np.arange(j_lo, j_hi+1)
            h_idx = idx - k  # h[j-k]
            grad_h[k] += np.sum(weights[j_lo:j_hi+1] * h[h_idx])
        
        # sum_j w_j * (h[k-j] if 0<=k-j<N else 0)
        # k-j in [0,N-1] => j in [k-N+1, k]
        j_lo2 = max(0, k-N+1)
        j_hi2 = min(2*N-2, k)
        if j_lo2 <= j_hi2:
            idx2 = np.arange(j_lo2, j_hi2+1)
            h_idx2 = k - idx2  # h[k-j]
            grad_h[k] += np.sum(weights[j_lo2:j_hi2+1] * h[h_idx2])
    
    # d(val)/d(h) where val = abs(2N * LSE / s^2)
    # Assume s > 0 and LSE > 0 (typical case), so val = 2N * LSE / s^2
    # d(val)/d(h[k]) = 2N * (grad_LSE[k] * s^2 - LSE * 2*s * 1) / s^4
    # = 2N * (grad_LSE[k] * s - 2 * LSE) / s^3
    # where grad_LSE[k] = d(LSE)/d(h[k]) = sum_j w_j * d(conv[j])/d(h[k]) = grad_h[k] as computed
    
    # But we need to handle the abs and the sign
    sign = np.sign(2.0 * N * log_sum_exp / (s * s)) if (2.0 * N * log_sum_exp / (s * s)) != 0 else 1.0
    
    # d(val)/d(h[k]) = sign * 2N * (grad_h[k] * s - 2 * LSE) / s^3
    # where LSE = log_sum_exp
    LSE = log_sum_exp
    grad_val = sign * 2.0 * N * (grad_h * s - 2.0 * LSE) / (s ** 3)
    
    return val, grad_val

def _dct_basis(K):
    """Create DCT basis matrix (N x K) using orthonormal DCT-II."""
    n = np.arange(N)
    basis = np.zeros((N, K))
    for k in range(K):
        if k == 0:
            basis[:, k] = 1.0 / np.sqrt(N)
        else:
            basis[:, k] = np.sqrt(2.0 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return basis

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value_exact(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    # DCT basis with K modes
    K = 20
    B = _dct_basis(K)  # (N, K)
    B_T = B.T  # (K, N)
    
    def _dct_h(coeffs, base_val):
        """Generate h from DCT coefficients and base."""
        return base_val + B @ coeffs
    
    def _dct_smooth_obj(params, temp=0.1):
        """Smooth objective for DCT parameters: [coeffs..., base]."""
        coeffs = params[:-1]
        base_val = params[-1]
        h = _dct_h(coeffs, base_val)
        return _smooth_max_value(h, temperature=temp)
    
    def _dct_smooth_obj_grad(params, temp=0.1):
        """Smooth objective and gradient for DCT parameters."""
        coeffs = params[:-1]
        base_val = params[-1]
        h = _dct_h(coeffs, base_val)
        val, grad_h = _smooth_max_value_grad(h, temperature=temp)
        # grad_params = [B.T @ grad_h, sum(grad_h)]
        grad_coeffs = B_T @ grad_h
        grad_base = np.sum(grad_h)
        return val, np.concatenate([grad_coeffs, [grad_base]])
    
    # Phase 1: Random restarts with projected gradient descent on DCT basis
    np.random.seed(42)
    temps = [0.5, 0.2, 0.1, 0.05]
    
    for temp in temps:
        if not deadline_ok():
            break
        
        for restart in range(5):
            if not deadline_ok():
                break
            
            # Random initialization
            if restart == 0:
                # Start from ones
                h_init = np.ones(N)
            else:
                # Random DCT coefficients
                coeffs_init = np.random.randn(K) * 0.5
                base_init = 1.0 + np.random.randn() * 0.5
                h_init = _dct_h(coeffs_init, base_init)
            
            # Get initial DCT representation
            mean_h = np.mean(h_init)
            # Project onto DCT basis
            coeffs = np.zeros(K)
            for k in range(K):
                if k == 0:
                    coeffs[k] = np.mean(h_init)
                else:
                    coeffs[k] = np.sqrt(2.0 / N) * np.sum(h_init * np.cos(np.pi * (2 * np.arange(N) + 1) * k / (2 * N)))
            # Remove mean from coeffs[0] since we have separate base
            coeffs[0] = 0.0  # base handles the mean
            base = mean_h
            
            params = np.concatenate([coeffs, [base]])
            
            # Projected gradient descent
            lr = 0.01
            for step in range(500):
                if not deadline_ok():
                    break
                
                val, grad = _dct_smooth_obj_grad(params, temp)
                
                if not np.isfinite(val):
                    break
                
                # Adaptive step size
                params_new = params - lr * grad
                
                # Check if sum is too close to zero
                h_new = _dct_h(params_new[:-1], params_new[-1])
                s_new = np.sum(h_new)
                if abs(s_new) < 0.1:
                    # Push base away from making sum zero
                    params_new[-1] += 0.1 * np.sign(s_new) if s_new != 0 else 0.1
                    h_new = _dct_h(params_new[:-1], params_new[-1])
                
                val_new = _smooth_max_value(h_new, temperature=temp)
                
                if val_new < val:
                    params = params_new
                    # Decrease learning rate slightly
                    lr *= 0.98
                else:
                    # Backtrack
                    lr *= 0.5
                    if lr < 1e-8:
                        break
            
            # Evaluate exact value
            h_final = _dct_h(params[:-1], params[-1])
            v_exact = _value_exact(h_final)
            if v_exact < best_v:
                best_v = v_exact
                best_h = h_final.copy()
    
    # Phase 2: L-BFGS-B refinement on DCT basis
    if deadline_ok():
        try:
            # Initialize from best_h
            h_best = best_h
            mean_h = np.mean(h_best)
            coeffs = np.zeros(K)
            for k in range(K):
                if k == 0:
                    coeffs[k] = 0.0
                else:
                    coeffs[k] = np.sqrt(2.0 / N) * np.sum(h_best * np.cos(np.pi * (2 * np.arange(N) + 1) * k / (2 * N)))
            params = np.concatenate([coeffs, [mean_h]])
            
            res = minimize(
                _dct_smooth_obj,
                params,
                args=(0.05,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res.x is not None:
                h = _dct_h(res.x[:-1], res.x[-1])
                v = _value_exact(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish on DCT basis
    if deadline_ok():
        try:
            h_best = best_h
            mean_h = np.mean(h_best)
            coeffs = np.zeros(K)
            for k in range(K):
                if k == 0:
                    coeffs[k] = 0.0
                else:
                    coeffs[k] = np.sqrt(2.0 / N) * np.sum(h_best * np.cos(np.pi * (2 * np.arange(N) + 1) * k / (2 * N)))
            params = np.concatenate([coeffs, [mean_h]])
            
            res = minimize(
                _dct_smooth_obj,
                params,
                args=(0.05,),
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 200, 'maxfev': 500},
            )
            if res.x is not None:
                h = _dct_h(res.x[:-1], res.x[-1])
                v = _value_exact(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
