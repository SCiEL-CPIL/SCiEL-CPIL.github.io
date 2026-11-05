"""Rewritten program using DCT spectral shaping with projected gradient descent."""
import numpy as np
import time

N = 400
_DEADLINE = 130.0  # safety margin before 150s kill


def _compute_value(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _smooth_max_surrogate(conv, alpha=50.0):
    """
    Smooth approximation of max(conv) using log-sum-exp.
    Returns (surrogate_value, gradient_wrt_conv).
    """
    # Scale for numerical stability
    m = np.max(conv)
    shifted = conv - m
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 0)
    exp_shifted = np.exp(alpha * shifted)
    log_sum_exp = m + np.log(np.sum(exp_shifted)) / alpha
    
    # Gradient: d/dconv_j [log_sum_exp] = exp(alpha*(conv_j - m)) / sum(exp(alpha*(conv - m)))
    grad = exp_shifted / np.sum(exp_shifted)
    return log_sum_exp, grad


def _build_h_from_dct(dct_coeffs):
    """Build h from DCT coefficients. dct_coeffs[0] is the DC component."""
    # Use scipy's dct/IDCT with ortho normalization
    from scipy.fft import dct, idct
    return idct(dct_coeffs, type=2, norm='ortho')


def _dct_of_h(h):
    """Compute DCT of h."""
    from scipy.fft import dct
    return dct(h, type=2, norm='ortho')


def _objective_and_grad(dct_coeffs, alpha=50.0):
    """
    Compute smooth surrogate objective and its gradient w.r.t. DCT coefficients.
    """
    h = _build_h_from_dct(dct_coeffs)
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10, np.zeros_like(dct_coeffs)
    
    # Compute autoconvolution via FFT for speed
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n_fft)
    conv = np.fft.ifft(H * H).real[:2 * N - 1]
    
    surrogate, grad_conv = _smooth_max_surrogate(conv, alpha)
    
    # Value: abs(2*N*surrogate / s^2)
    # Since we want to minimize, and surrogate ≈ max(conv), we minimize surrogate/s^2
    val = 2.0 * N * surrogate / (s * s)
    
    # Gradient of val w.r.t. h:
    # val = 2N * surrogate / s^2
    # dval/dh = 2N * (grad_surr * s^2 - surrogate * 2*s * grad_s) / s^4
    #         = 2N * (grad_surr / s^2 - 2*surrogate * grad_s / s^3)
    # where grad_s = d(surrogate)/dh = sum_k grad_conv[k] * d(conv[k])/dh
    # conv[k] = sum_n h[n]*h[n+k], so d(conv[k])/dh[n] = h[n+k] + h[n-k] (with boundary handling)
    
    # For full convolution: conv[j] = sum_{n} h[n] * h[n + j - (N-1)]
    # Actually convolve(h,h)[j] = sum_n h[n] * h[n + j - N + 1] for j in [0, 2N-2]
    # d(conv[j])/dh[m] = h[m + j - N + 1] if valid + h[m - j + N - 1] if valid
    
    # Simpler: use the fact that conv = ifft(fft(h)^2)
    # d(conv)/dh = ifft(2*fft(h) * d(fft(h))/dh)
    # d(fft(h))/dh[m] = exp(-2*pi*i*m*n/n_fft) for each n
    # This gets complicated. Let's use a different approach.
    
    # Alternative: compute gradient numerically-free via the formula
    # d/dh[m] [sum_j grad_conv[j] * conv[j]] 
    # conv[j] = sum_n h[n] h[n+j-N+1]
    # Let k = j - N + 1, so conv[j] = sum_n h[n] h[n+k] where k ranges from -(N-1) to N-1
    # d/dh[m] conv[j] = h[m+k] (if 0<=m+k<N) + h[m-k] (if 0<=m-k<N)
    
    grad_h = np.zeros(N)
    for j in range(2 * N - 1):
        k = j - N + 1
        gc = grad_conv[j]
        if gc == 0:
            continue
        # d(conv[j])/dh[m] = h[m+k] + h[m-k] (with bounds)
        if k >= 0:
            # h[m+k] valid for m in [0, N-1-k]
            if N - 1 - k >= 0:
                grad_h[:N-k] += gc * h[k:]
            # h[m-k] valid for m in [k, N-1]
            grad_h[k:] += gc * h[:N-k]
        else:
            kk = -k
            # h[m-kk] valid for m in [0, N-1-kk]
            if N - 1 - kk >= 0:
                grad_h[:N-kk] += gc * h[kk:]
            # h[m+kk] valid for m in [kk, N-1]
            grad_h[kk:] += gc * h[:N-kk]
    
    s_grad = np.ones(N)  # d(s)/dh[m] = 1
    val_grad_h = 2.0 * N * (grad_h / (s * s) - 2.0 * surrogate * s_grad / (s * s * s))
    
    # Gradient w.r.t. DCT coefficients: grad_dct = DCT(val_grad_h)
    from scipy.fft import dct
    grad_dct = dct(val_grad_h, type=2, norm='ortho')
    
    return val, grad_dct


def _project(dct_coeffs, h_min=-10.0, h_max=10.0):
    """
    Project DCT coefficients so that resulting h is within [h_min, h_max].
    Use iterative projection.
    """
    # Simple approach: just return as-is, and clamp h after reconstruction
    # For better behavior, we could do iterative projection but it's expensive
    return dct_coeffs


def _pgd_step(dct_coeffs, grad_dct, lr, alpha=50.0):
    """One step of projected gradient descent in DCT domain."""
    new_dct = dct_coeffs - lr * grad_dct
    
    # Check if resulting h is valid
    h_new = _build_h_from_dct(new_dct)
    if not np.all(np.isfinite(h_new)):
        return dct_coeffs, False
    
    # Clamp h to reasonable range
    h_clamped = np.clip(h_new, -10.0, 10.0)
    
    # Re-derive DCT coefficients from clamped h
    from scipy.fft import dct
    new_dct = dct(h_clamped, type=2, norm='ortho')
    
    return new_dct, True


def _optimize_dct(dct_init, max_iters=500, alpha=50.0, deadline=None):
    """
    Optimize DCT coefficients using projected gradient descent.
    """
    t0 = time.time()
    dct_curr = dct_init.copy()
    best_val = 1e10
    best_dct = dct_init.copy()
    
    # Adaptive learning rate
    lr = 0.1
    min_lr = 1e-6
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        val, grad = _objective_and_grad(dct_curr, alpha)
        
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_dct = dct_curr.copy()
        
        # Adaptive step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Backtracking line search
        step = lr
        improved = False
        for _ in range(20):
            new_dct, valid = _pgd_step(dct_curr, grad, step, alpha)
            if not valid:
                step *= 0.5
                continue
            
            new_val, _ = _objective_and_grad(new_dct, alpha)
            if np.isfinite(new_val) and new_val < val - 1e-8 * abs(val):
                dct_curr = new_dct
                improved = True
                break
            step *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < min_lr:
                break
        else:
            lr = min(lr * 1.1, 1.0)
    
    return best_dct, best_val


def _local_refine(h, deadline, max_iter=5000):
    """Local refinement with random perturbations."""
    best_val = _compute_value(h)
    best_h = h.copy()
    rng = np.random.default_rng(42)
    step_size = 0.1
    
    for step in range(max_iter):
        if time.time() > deadline:
            break
        
        n_perturb = rng.integers(1, 6)
        idx = rng.choice(N, size=n_perturb, replace=False)
        h_new = best_h.copy()
        h_new[idx] += rng.normal(0, step_size, n_perturb)
        
        v_new = _compute_value(h_new)
        if np.isfinite(v_new) and v_new < best_val - 1e-10:
            best_val = v_new
            best_h = h_new
            step_size *= 0.98
        elif step % 1000 == 0:
            step_size *= 0.999
    
    return best_h, best_val


def construct():
    """
    Construct step heights using DCT spectral shaping with projected gradient descent.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    rng = np.random.default_rng(123)
    
    best_val = 2.0  # seed value
    best_h = np.ones(N).copy()
    
    # Phase 1: DCT-based optimization with multiple restarts
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Create diverse initial spectral shapes
        if restart == 0:
            # Start from uniform (DC only)
            dct_init = np.zeros(N)
            dct_init[0] = np.sqrt(N) * 1.0  # h = ones
        elif restart == 1:
            # Low-frequency dominant
            dct_init = np.zeros(N)
            dct_init[0] = np.sqrt(N) * 1.0
            dct_init[1] = rng.uniform(-0.5, 0.5) * np.sqrt(N) / 2
            dct_init[2] = rng.uniform(-0.3, 0.3) * np.sqrt(N) / 2
        elif restart == 2:
            # Medium frequency
            dct_init = np.zeros(N)
            dct_init[0] = np.sqrt(N) * 1.0
            k = rng.integers(5, 20)
            dct_init[k] = rng.uniform(-0.5, 0.5) * np.sqrt(N) / 2
        elif restart == 3:
            # High frequency
            dct_init = np.zeros(N)
            dct_init[0] = np.sqrt(N) * 1.0
            k = rng.integers(50, 150)
            dct_init[k] = rng.uniform(-0.3, 0.3) * np.sqrt(N) / 2
        elif restart == 4:
            # Broad spectrum
            dct_init = np.zeros(N)
            dct_init[0] = np.sqrt(N) * 1.0
            for k in range(1, 10):
                dct_init[k] = rng.uniform(-0.2, 0.2) * np.sqrt(N) / (k + 1)
        elif restart == 5:
            # Alternating pattern (high freq)
            h_alt = np.ones(N)
            h_alt[::2] = -0.5
            from scipy.fft import dct
            dct_init = dct(h_alt, type=2, norm='ortho')
        elif restart == 6:
            # Random smooth spectrum
            dct_init = np.zeros(N)
            dct_init[0] = np.sqrt(N) * 1.0
            for k in range(1, 30):
                dct_init[k] = rng.normal(0, 0.1) * np.sqrt(N) / (k + 1)
        else:
            # Pure random small perturbation
            dct_init = np.zeros(N)
            dct_init[0] = np.sqrt(N) * 1.0
            dct_init[1:10] = rng.normal(0, 0.05, 9) * np.sqrt(N)
        
        # Optimize
        try:
            opt_dct, opt_val = _optimize_dct(
                dct_init,
                max_iters=300,
                alpha=50.0,
                deadline=deadline - 15
            )
            
            if np.isfinite(opt_val) and opt_val < best_val:
                best_val = opt_val
                best_h = _build_h_from_dct(opt_dct)
        except Exception:
            pass
    
    # Phase 2: Local refinement
    if time.time() < deadline - 10:
        best_h, best_val = _local_refine(best_h, deadline - 5)
    
    # Phase 3: Try a few more DCT restarts if time permits
    if time.time() < deadline - 25:
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            
            # More aggressive spectral shapes
            dct_init = np.zeros(N)
            dct_init[0] = np.sqrt(N) * rng.uniform(0.5, 2.0)
            n_harmonics = rng.integers(3, 15)
            for k in range(1, n_harmonics + 1):
                dct_init[k] = rng.normal(0, 0.15) * np.sqrt(N) / (k + 1)
            
            try:
                opt_dct, opt_val = _optimize_dct(
                    dct_init,
                    max_iters=200,
                    alpha=50.0,
                    deadline=deadline - 10
                )
                
                if np.isfinite(opt_val) and opt_val < best_val:
                    best_val = opt_val
                    best_h = _build_h_from_dct(opt_dct)
            except Exception:
                pass
    
    # Phase 4: Final local refinement
    if time.time() < deadline - 5:
        best_h, best_val = _local_refine(best_h, deadline - 2, max_iter=10000)
    
    # Final check
    final_val = _compute_value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
    
    return best_h
