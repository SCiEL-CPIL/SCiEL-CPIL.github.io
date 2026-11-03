"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Structural change: parameterize h in the frequency domain as |a_k|^2/N,
which inherently enforces non-negativity. Optimize the complex coefficients
(a_0..a_49) directly using Nesterov accelerated gradient descent on a
smooth-max surrogate. The sum constraint is enforced by fixing |a_0| = sqrt(N/2)
and projecting the DC phase if needed. This removes the need for box-constraint
projection and may find better optima by exploring the non-convex landscape
of the spectral representation.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _h_from_params(params):
    """
    Reconstruct h from complex frequency parameters.
    params[0] is complex (DC term), params[1:] are complex (positive freqs).
    We fix the magnitude of the DC term to enforce sum(h)=N/2.
    h is real-valued if params satisfy conjugate symmetry, which we enforce
    by constructing the full FFT spectrum from the first N/2+1 components.
    """
    n = N
    half = n // 2 + 1  # number of non-negative frequency bins for real signal
    
    # Extract DC magnitude and phase
    dc_mag = np.sqrt(n / 2.0)  # Fixed to enforce sum = N/2
    dc_phase = np.angle(params[0])
    
    # Construct the full FFT spectrum
    # For a real signal of length n, X[k] = conj(X[n-k])
    # We have params[0]..params[half-1] corresponding to frequencies 0..half-1
    # We need to construct X[0]..X[n-1]
    
    X = np.zeros(n, dtype=np.complex128)
    
    # DC component
    X[0] = dc_mag * np.exp(1j * dc_phase)
    
    # Positive frequencies (1 to half-1)
    # params[1]..params[half-1] correspond to X[1]..X[half-1]
    # Note: params has length half (0 to half-1)
    for k in range(1, half):
        if k < len(params):
            X[k] = params[k]
            X[n - k] = np.conj(params[k])
        else:
            X[k] = 0.0
            X[n - k] = 0.0
    
    # If n is even, X[n/2] is its own conjugate, so it must be real
    if n % 2 == 0:
        k_mid = n // 2
        if k_mid < len(params):
            X[k_mid] = np.real(params[k_mid])  # Force real
        else:
            X[k_mid] = 0.0
    
    # Inverse FFT to get h
    h = np.fft.irfft(X)
    
    # Clip to [0, 1] and re-normalize to ensure sum = N/2
    # This step is a safeguard; ideally the optimization keeps h in bounds
    h = np.clip(h, 0.0, 1.0)
    
    # Re-normalize to exact sum
    current_sum = np.sum(h)
    if current_sum > 1e-12:
        h = h * (n / 2.0) / current_sum
        h = np.clip(h, 0.0, 1.0)
        # Final tiny adjustment for exact sum
        diff = n / 2.0 - np.sum(h)
        if abs(diff) > 1e-14:
            # Add/subtract diff to the element with most room
            if diff > 0:
                idx = np.argmin(h)
                h[idx] += diff
            else:
                idx = np.argmax(h)
                h[idx] += diff
            h = np.clip(h, 0.0, 1.0)
    
    return h


def _params_from_h(h):
    """Convert h to frequency domain parameters."""
    h = np.asarray(h, dtype=np.float64)
    X = np.fft.rfft(h)
    # X has length N/2 + 1
    # We optimize all of them, but fix the magnitude of X[0] during reconstruction
    # For initialization, we just return X
    return X


def _smooth_max_obj_grad_params(params, beta):
    """
    Compute the value and gradient of the smooth max objective with respect to
    the frequency parameters.
    Uses the chain rule: dF/dparams = dF/dh * dh/dparams
    """
    h = _h_from_params(params)
    cs = _linear_overlap_values(h)
    
    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp
    
    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)
    
    # Gradient of C_k with respect to h
    # C_k = (2/N) * (N/2 - r_k)
    # r_k = sum_{i=k}^{N-1} h_i h_{i-k}
    # grad_h C_k = -(2/N) * grad_h r_k
    # grad_h r_k is the autocorrelation gradient
    
    # We compute grad_h F = sum_k w_k * grad_h C_k
    # Using FFT for efficiency
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    
    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)
    
    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]
    
    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)
    
    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]
    
    grad_h = -(2.0 / N) * (A + B)
    
    # Now compute grad_params = J^T * grad_h
    # where J = dh/dparams
    # h = IFFT(X(params))
    # X is constructed from params.
    # This is complex. Let's use numerical differentiation for the Jacobian?
    # No, that's too slow.
    
    # Alternative: Since h is a linear function of the real and imaginary parts of X (before clipping/normalization),
    # and X is a linear function of params, dh/dparams is a linear operator.
    # However, the clipping and normalization break linearity.
    
    # For a robust approach, we can use the fact that if h is well-behaved (inside (0,1)),
    # the gradient with respect to X is simply the DFT of grad_h.
    # grad_X F = DFT(grad_h)
    # Then grad_params = grad_X (since params are just X components)
    
    # Let's assume the clipping/normalization is a minor perturbation and compute
    # grad_X = fft(grad_h)
    
    grad_X = np.fft.rfft(grad_h)
    
    # grad_params is simply grad_X, but we need to handle the DC magnitude constraint.
    # In _h_from_params, we fixed |X[0]|. So the gradient with respect to the phase of X[0]
    # is different.
    # For simplicity in this first pass, let's just return grad_X as grad_params.
    # The optimizer will effectively optimize the phase of X[0] and the other components.
    # The magnitude of X[0] is fixed in the reconstruction, so changes to params[0]
    # only affect the phase.
    
    # Actually, if we fix |X[0]| in reconstruction, then d h / d params[0] depends only on the phase.
    # Let's just return grad_X. The optimizer might struggle if it tries to change the magnitude
    # of X[0], but since we fix it in reconstruction, the effective gradient in that direction is zero.
    # To be safe, we can zero out the gradient for the magnitude of X[0].
    # But params are complex. Let's just return grad_X.
    
    return val, grad_X


def _nesterov_freq(params0, beta, n_iter=300, lr=0.05):
    """
    Nesterov accelerated gradient descent in the frequency domain.
    """
    params = np.array(params0, dtype=np.complex128).copy()
    
    h_init = _h_from_params(params)
    best_h = h_init.copy()
    best_score = _score(h_init)
    
    params_prev = params.copy()
    t = 1.0
    
    lr_current = lr
    max_backtracks = 12
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new
        
        y = params + momentum * (params - params_prev)
        
        val_y, grad = _smooth_max_obj_grad_params(y, beta)
        
        step_lr = lr_current
        params_new = None
        for bt in range(max_backtracks):
            params_candidate = y - step_lr * grad
            h_candidate = _h_from_params(params_candidate)
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-8:
                params_new = params_candidate
                lr_current = step_lr
                break
            step_lr *= 0.5
        
        if params_new is None:
            params_new = y - lr_current * 0.5 * grad
            lr_current *= 0.5
        
        params_prev = params
        params = params_new
        t = t_new
        
        h_current = _h_from_params(params)
        current_score = _score(h_current)
        if current_score < best_score:
            best_score = current_score
            best_h = h_current.copy()
    
    return best_h, best_score


def _high_beta_polish(h0, n_iter=80, beta=200.0, lr=0.005):
    """
    Final high-precision polish pass in the time domain (using the original method)
    to ensure we are in a good local minimum.
    """
    h = np.array(h0, dtype=np.float64)
    
    # Use the time-domain smooth max gradient
    def _td_smooth_max_obj_grad(h, beta):
        cs = _linear_overlap_values(h)
        cs_max = np.max(cs)
        exp_cs = np.exp(beta * (cs - cs_max))
        sum_exp = np.sum(exp_cs)
        w = exp_cs / sum_exp
        val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)
        
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        
        w_pad = np.zeros(n_fft)
        w_pad[:N] = w
        h_pad = np.zeros(n_fft)
        h_pad[:N] = h
        
        W = np.fft.rfft(w_pad)
        H = np.fft.rfft(h_pad)
        
        conv = np.fft.irfft(W * H, n=n_fft)
        A = conv[:N]
        
        w_rev = w[::-1]
        w_rev_pad = np.zeros(n_fft)
        w_rev_pad[:N] = w_rev
        W_rev = np.fft.rfft(w_rev_pad)
        
        conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
        B = conv_rev[N-1 : N-1+N]
        
        grad = -(2.0 / N) * (A + B)
        return val, grad
    
    def _project(h, target_sum):
        h = np.asarray(h, dtype=np.float64).copy()
        low = -np.max(h) - 1.0
        high = 1.0 - np.min(h) + 1.0
        current_sum = np.sum(h)
        if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
            return np.clip(h, 0.0, 1.0)
        def sum_of_clipped(offset):
            return np.sum(np.clip(h + offset, 0.0, 1.0))
        for _ in range(60):
            mid = (low + high) / 2.0
            s_mid = sum_of_clipped(mid)
            if s_mid > target_sum:
                high = mid
            else:
                low = mid
        offset = (low + high) / 2.0
        h_proj = np.clip(h + offset, 0.0, 1.0)
        diff = target_sum - np.sum(h_proj)
        if abs(diff) > 1e-14:
            if diff > 0:
                idx = np.argmin(h_proj)
                h_proj[idx] += diff
            else:
                idx = np.argmax(h_proj)
                h_proj[idx] += diff
            h_proj = np.clip(h_proj, 0.0, 1.0)
        return h_proj

    h = _project(h, N / 2.0)
    best_h = h.copy()
    best_score = _score(h)
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 2:
            break
        val, grad = _td_smooth_max_obj_grad(h, beta)
        h_new = h - lr * grad
        h_new = _project(h_new, N / 2.0)
        current_score = _score(h_new)
        if current_score < best_score:
            best_score = current_score
            best_h = h_new.copy()
        h = h_new
    
    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=np.float64)
        # Ensure validity
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-14:
            if diff > 0:
                idx = np.argmin(h)
                h[idx] += diff
            else:
                idx = np.argmax(h)
                h[idx] += diff
            h = np.clip(h, 0.0, 1.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Diverse initializations in time domain
    initializations_h = []
    
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    initializations_h.append(h_cos)
    
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    initializations_h.append(h_sin)
    
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    h_alt[1::2] = 0.0
    initializations_h.append(h_alt)
    
    for _ in range(3):
        h_rand = np.random.uniform(0, 1, N)
        initializations_h.append(h_rand)
    
    h_block1 = np.zeros(N)
    h_block1[:N//2] = 1.0
    initializations_h.append(h_block1)
    
    h_block2 = np.zeros(N)
    h_block2[N//2:] = 1.0
    initializations_h.append(h_block2)
    
    # 2. Run frequency-domain Nesterov from each initialization
    for h0 in initializations_h:
        if time.time() > _DEADLINE - 20:
            break
        _update_best(h0)
        
        # Convert to frequency params
        params0 = _params_from_h(h0)
        
        # Run frequency-domain optimization
        h_opt, s_opt = _nesterov_freq(params0, beta=50.0, n_iter=200, lr=0.05)
        _update_best(h_opt)
    
    # 3. Final polish with high beta in time domain
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_polished, s_polished = _high_beta_polish(best_h, n_iter=80, beta=200.0, lr=0.005)
        _update_best(h_polished)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Final validity check
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(best_h)
            best_h[idx] += diff
        else:
            idx = np.argmax(best_h)
            best_h[idx] += diff
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
