import time
import numpy as np

N = 400
_t0 = time.time()
_DEADLINE = 135.0

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _grad_soft_max_conv(h, beta):
    """
    Compute gradient of soft-max of auto-correlation.
    L = log(sum(exp(beta * C_k)))
    grad = sum_k p_k * dC_k/dh
    dC_k/dh is the kernel.
    """
    conv = _fft_convolve(h)
    # Soft max weights
    max_c = np.max(conv)
    exp_c = np.exp(beta * (conv - max_c))
    p = exp_c / np.sum(exp_c)
    
    # dL/dh_j = sum_k p_k * (h_{j+k} + h_{j-k})
    # This is equivalent to:
    # Term 1: sum_k p_k h_{j+k} -> Correlation of p and h
    # Term 2: sum_k p_k h_{j-k} -> Correlation of p_rev and h
    # Note: p is defined on indices 0..2N-2.
    # We need to handle the convolution/correlation carefully.
    
    # Let's use FFT for the gradient calculation.
    # dC_k/dh_j is non-zero only if indices are valid.
    # Actually, the operator dC/dh is a convolution with a specific kernel?
    # C = h * h.
    # dC/dh = h * I + I * h = 2 * (h * I) ? No, h is not constant.
    # dC_k = sum_i (dh_i h_{i+k} + h_i dh_{i+k})
    # Let dh be a vector.
    # dC = convolve(dh, h) + convolve(h, dh)
    # Since conv is symmetric, dC = 2 * convolve(dh, h) ?
    # Wait, convolve(dh, h) gives terms dh_i h_{i+k}.
    # convolve(h, dh) gives terms h_i dh_{i+k} = dh_{i+k} h_i.
    # Let j = i+k. h_{j-k} dh_j.
    # So dC_k = sum_j (dh_j h_{j+k} + dh_j h_{j-k}).
    # dL = sum_k p_k dC_k = sum_k p_k sum_j dh_j (h_{j+k} + h_{j-k})
    # dL/dh_j = sum_k p_k (h_{j+k} + h_{j-k}).
    
    # Let's compute G_j = sum_k p_k h_{j+k} + sum_k p_k h_{j-k}.
    # Let p be extended to size 2N-1.
    # We can compute this using FFT.
    # Term 1: sum_k p_k h_{j+k}. This is like a cross-correlation.
    # If we define p_ext such that p_ext[k] = p[k] for k>=0?
    # Let's just do it directly with FFT multiplication.
    
    size = 2 * N
    P = np.fft.rfft(p, size)
    H = np.fft.rfft(h, size)
    
    # The operation sum_k p_k h_{j+k} is equivalent to convolving p with h, 
    # but with p reversed?
    # Let's check indices.
    # y_j = sum_k p_k h_{j+k}
    # Let m = j+k => k = m-j.
    # y_j = sum_m p_{m-j} h_m
    # This is correlation.
    # np.correlate(p, h) ?
    # Using FFT: ifft(fft(p_rev) * fft(h)) ?
    # Let's use the property that conv(a, b) = ifft(fft(a)*fft(b)).
    # We want sum_k p_k h_{j+k}.
    # This is the coefficient of x^j in P(x) H(x^{-1})?
    # Simpler: 
    # G1_j = sum_k p_k h_{j+k}
    # G2_j = sum_k p_k h_{j-k}
    
    # G2 is standard convolution of p and h?
    # (p * h)_j = sum_k p_k h_{j-k}. Yes.
    # So G2 = convolve(p, h).
    
    # G1_j = sum_k p_k h_{j+k}.
    # Let l = -k. sum_l p_{-l} h_{j-l}.
    # This is convolve(p_rev, h) where p_rev[l] = p_{-l}.
    # p is defined on 0..2N-2.
    # p_rev should be defined such that p_rev[k] = p[2N-2-k]?
    # Let's verify.
    # If p is symmetric (which it is, since C is symmetric), p_rev = p.
    # Auto-correlation C_k = C_{-k}. So p_k = p_{-k}.
    # So G1 = G2.
    # Therefore grad = 2 * convolve(p, h).
    
    # Let's verify symmetry.
    # C_k = sum_i h_i h_{i+k}.
    # C_{-k} = sum_i h_i h_{i-k}. Let j = i-k => i = j+k.
    # C_{-k} = sum_j h_{j+k} h_j = C_k.
    # Yes, C is symmetric. So p is symmetric.
    # So grad = 2 * convolve(p, h).
    
    # We need to convolve p (length 2N-1) with h (length N).
    # Result length 3N-2.
    # We only need the part corresponding to h indices 0..N-1.
    # The gradient vector g has length N.
    # g_j = 2 * sum_k p_k h_{j-k} (where h is 0 outside 0..N-1).
    # This is the central part of the full convolution of p and h.
    
    # Let's compute full conv of p and h.
    # p length 2N-1. h length N.
    # conv length 3N-2.
    # The term for j=0 in grad corresponds to index in conv?
    # conv_idx = j + (len(p)-1) ?
    # Let's check.
    # conv[m] = sum_i p_i h_{m-i}.
    # We want sum_k p_k h_{j-k}.
    # So m = j.
    # So grad_j = 2 * conv[j].
    # Wait, p is indexed 0..2N-2.
    # h is indexed 0..N-1.
    # conv[0] = p_0 h_0.
    # conv[j] = sum_i p_i h_{j-i}.
    # If j=0, sum_i p_i h_{-i} = p_0 h_0.
    # If j=N-1, sum_i p_i h_{N-1-i}.
    # This matches the formula.
    
    conv_p_h = np.convolve(p, h)
    # We take the first N elements?
    # Let's check the range.
    # p has support 0..2N-2.
    # h has support 0..N-1.
    # conv has support 0..3N-3.
    # We need grad_j for j=0..N-1.
    # grad_j = 2 * conv[j].
    
    grad = 2.0 * conv_p_h[:N]
    
    return grad

def construct():
    global _t0
    _t0 = time.time()
    
    # Initialize with ones
    h = np.ones(N)
    # Normalize sum to 1 for optimization stability, then scale back or just keep scale invariant
    # V is scale invariant. We can fix sum(h) = 1.
    h = h / np.sum(h)
    
    best_h = h.copy()
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    # Gradient Descent on Soft Max
    # V = 2N * L / S^2
    # We minimize V.
    # dV/dh = (2N/S^2) * (dL/dh - 2L/S * dS/dh)
    # dS/dh = 1 (vector of ones)
    
    beta = 50.0
    lr = 0.01
    steps = 5000
    
    for i in range(steps):
        if not deadline_ok():
            break
            
        s = np.sum(h)
        if s == 0:
            break
            
        # Compute L and grad L
        conv = _fft_convolve(h)
        max_c = np.max(conv)
        exp_c = np.exp(beta * (conv - max_c))
        sum_exp = np.sum(exp_c)
        L = max_c + np.log(sum_exp) / beta
        
        # Grad L
        grad_L = _grad_soft_max_conv(h, beta)
        
        # Grad V
        # V = 2N L / S^2
        # dV = 2N ( S^2 dL - L 2S dS ) / S^4
        # dV = 2N/S^2 ( dL - 2L/S dS )
        dS = np.ones(N)
        grad_V = (2.0 * N / (s * s)) * (grad_L - (2.0 * L / s) * dS)
        
        # Gradient Descent Step
        h_new = h - lr * grad_V
        
        # Project to sum=1 to maintain scale invariance and avoid drift
        s_new = np.sum(h_new)
        if s_new != 0:
            h_new = h_new / s_new
        
        # Check value
        v_new = _value(h_new)
        
        if v_new < best_v:
            update_best(h_new)
            h = h_new
            # Decay learning rate
            lr *= 0.99
        else:
            # If it fails, try smaller step or restart?
            # For now, just decay lr
            lr *= 0.95
            
        if lr < 1e-10:
            lr = 1e-10

    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    
    return best_h
