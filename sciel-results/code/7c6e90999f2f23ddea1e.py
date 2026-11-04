"""Combined approach: spectral flatness initialization + efficient local search."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv(x):
    """Compute full linear autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N
    X = np.fft.fft(x, n=n)
    C = np.fft.ifft(X * X).real[:2*N-1]
    return C

def _eval_ratio(x):
    """Compute the raw value: max|c_k| / (sum x)^2 for k != 0."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    if len(c) > 1:
        max_sidelobe = np.max(np.abs(c[1:]))
    else:
        max_sidelobe = 0.0
    return max_sidelobe / (s * s)

def _spectral_objective(x):
    """Objective: mean |DFT|^2 (flat magnitude) + penalty on sum(x).
    x is continuous in R^N. Returns (value, grad)."""
    F = np.fft.rfft(x)
    mag2 = np.abs(F) ** 2
    # We want mag2 flat => minimize variance of mag2, equivalently minimize sum mag2^2
    val = np.sum(mag2 * mag2)
    # sum penalty: encourage x to have nonzero sum (so denominator large)
    # but we'll handle normalization at the end.
    grad = np.zeros_like(x)
    # d/dx sum |F|^4 : use chain rule
    # |F|^2 = F * conj(F); d|F|^2/dx = 2 Re(conj(F) * dF/dx)
    # We'll compute via: g = 2 * mag2 * F  (in freq), then grad = 2 * irfft(g)
    g = 2.0 * mag2 * F
    grad = np.fft.irfft(g, n=N) * 2.0
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with all +1
    best_x = np.ones(N, dtype=np.int8)
    best_val = _eval_ratio(best_x)
    
    np.random.seed(42)
    rng = np.random.default_rng(0)
    
    # Try a few random initializations
    for init_iter in range(10):
        if time.time() > deadline:
            break
        
        # Random initialization with bias toward +1
        x = np.random.choice([1, -1], size=N, p=[0.8, 0.2])
        x = x.astype(np.int8)
        
        current_val = _eval_ratio(x)
        
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
    
    # Gradient descent on continuous relaxation to flatten spectrum
    # Objective: minimize sum |DFT(x)|^4  subject to x near +/-1
    # Use projected gradient: project onto box [-1,1], then round.
    
    x = best_x.astype(float).copy()
    lr = 0.01
    for it in range(2000):
        if time.time() > deadline:
            break
        val, grad = _spectral_objective(x)
        # Projected gradient step
        x = x - lr * grad
        # Project to [-1, 1]
        x = np.clip(x, -1.0, 1.0)
        
        # Round and evaluate periodically
        if it % 20 == 0 or it == 1999:
            s = np.sign(x)
            s[s == 0] = 1.0
            v = _eval_ratio(s)
            if v < best_val:
                best_val = v
                best_x = s.astype(np.int8).copy()
        
        # Adaptive learning rate
        if it % 200 == 199:
            lr *= 0.5
    
    # Local search: flip single elements to reduce peak
    s = best_x.copy()
    v = _eval_ratio(s)
    best_local = s.copy()
    best_local_v = v
    
    for _round in range(6):
        if time.time() > deadline:
            break
        improved = False
        ac = _autoconv(s)
        peak = np.max(np.abs(ac[1:]))
        s_sum = s.sum()
        denom = s_sum * s_sum
        
        for i in range(0, N, 1):
            if time.time() > deadline:
                break
            # Try flipping i
            s[i] = -s[i]
            # Recompute autoconv efficiently? Just recompute (N=400, fast)
            ac2 = _autoconv(s)
            peak2 = np.max(np.abs(ac2[1:]))
            s_sum2 = s.sum()
            denom2 = s_sum2 * s_sum2
            v2 = peak2 / denom2 if denom2 != 0 else peak2
            if v2 < best_local_v - 1e-12:
                best_local_v = v2
                best_local = s.copy()
                improved = True
            else:
                s[i] = -s[i]  # revert
        if not improved:
            break
        s = best_local.copy()
    
    if best_local_v < best_val:
        best_val = best_local_v
        best_x = best_local.copy()
    
    return best_x
# EVOLVE-BLOCK-END
