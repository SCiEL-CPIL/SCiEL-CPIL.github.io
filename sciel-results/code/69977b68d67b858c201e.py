"""Second autocorrelation inequality - optimize step heights."""
import numpy as np
import time

N = 50

def _autoconv(x):
    """Compute autoconvolution of sequence x."""
    n = len(x)
    ac = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            ac[i + j] += x[i] * x[j]
    return ac

def _score(x):
    """Compute the objective: (L2^2) / (L1 * sup|autoconv|)"""
    x = np.asarray(x, dtype=np.float64)
    l1 = np.sum(x)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(x * x)
    ac = _autoconv(x)
    ac_max = np.max(np.abs(ac))
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def _autoconv_fast(x):
    """Faster autoconvolution using FFT."""
    n = len(x)
    size = 2 * n
    # Pad to avoid circular convolution
    xf = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(xf * xf, n=size)
    # The autoconvolution of length-n signal has 2n-1 elements
    # ac[0] = x[0]*x[0], ac[1] = x[0]*x[1]+x[1]*x[0], etc.
    # But we need the "linear" autoconvolution
    # Actually for linear autoconvolution:
    # ac[k] = sum_{i+j=k} x[i]*x[j] for k=0..2n-2
    # Using FFT: we need to be careful
    # Let's just use direct for correctness with n=50
    return ac

def _autoconv_direct(x):
    """Direct autoconvolution - correct for our purpose."""
    n = len(x)
    ac = np.zeros(2 * n - 1)
    for i in range(n):
        if x[i] != 0:
            for j in range(n):
                ac[i + j] += x[i] * x[j]
    return ac

def _score_fast(x):
    """Compute score using direct autoconvolution."""
    x = np.asarray(x, dtype=np.float64)
    l1 = np.sum(x)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(x * x)
    ac = _autoconv_direct(x)
    ac_max = np.max(ac)  # all nonnegative so max is positive
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def _score_vectorized(x):
    """Score with vectorized autoconv."""
    x = np.asarray(x, dtype=np.float64)
    l1 = np.sum(x)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.dot(x, x)
    # Vectorized autoconv
    ac = np.zeros(2 * N - 1)
    for k in range(2 * N - 1):
        # ac[k] = sum_{i=0}^{min(k,N-1)} x[i]*x[k-i] where k-i in [0,N-1]
        i_min = max(0, k - (N - 1))
        i_max = min(k, N - 1)
        if i_min <= i_max:
            ac[k] = np.dot(x[i_min:i_max+1], x[k-i_max:k-i_min+1][::-1])
    ac_max = np.max(ac)
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def _score_numpy(x):
    """Score using numpy for autoconv via correlation trick."""
    x = np.asarray(x, dtype=np.float64)
    l1 = np.sum(x)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.dot(x, x)
    # Autoconvolution = correlation of x with reversed x
    # Use np.correlate
    ac = np.correlate(x, x, mode='full')
    ac_max = np.max(ac)
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def _gradient(x, eps=1e-5):
    """Numerical gradient of score."""
    n = len(x)
    grad = np.zeros(n)
    s0 = _score_numpy(x)
    for i in range(n):
        x_p = x.copy()
        x_p[i] += eps
        s_p = _score_numpy(x_p)
        x_m = x.copy()
        x_m[i] = max(0, x[i] - eps)
        s_m = _score_numpy(x_m)
        if x[i] > eps:
            grad[i] = (s_p - s_m) / (2 * eps)
        else:
            grad[i] = (s_p - s_m) / eps
    return grad

def _project(x, total=None):
    """Project to nonnegative, optionally scale to have sum=total."""
    x = np.maximum(0, x)
    if total is not None and np.sum(x) > 1e-12:
        x = x * (total / np.sum(x))
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin
    
    # Start with uniform
    best_x = np.ones(N)
    best_score = _score_numpy(best_x)
    
    # Try some structured initializations
    candidates = []
    # Uniform
    candidates.append(np.ones(N))
    # Triangular (peak in middle)
    tri = np.linspace(0.1, 1.0, N//2)
    candidates.append(np.concatenate([tri, tri[::-1]]))
    # Inverted triangular (peak at edges)
    inv_tri = np.linspace(1.0, 0.1, N//2)
    candidates.append(np.concatenate([inv_tri, inv_tri[::-1]]))
    # Sine wave
    candidates.append(0.5 + 0.5 * np.sin(np.linspace(0, np.pi, N)))
    # Cosine
    candidates.append(0.5 + 0.5 * np.cos(np.linspace(0, np.pi, N)))
    # Two peaks
    two_peak = np.zeros(N)
    two_peak[N//4] = 1.0
    two_peak[3*N//4] = 1.0
    candidates.append(two_peak)
    # Random
    np.random.seed(42)
    for _ in range(5):
        candidates.append(np.random.rand(N))
    
    # Evaluate candidates
    for c in candidates:
        c = _project(c)
        if np.sum(c) < 1e-12:
            continue
        s = _score_numpy(c)
        if s > best_score:
            best_score = s
            best_x = c.copy()
    
    # Local optimization with projected gradient ascent
    x = best_x.copy()
    lr = 0.1
    min_lr = 1e-6
    
    while time.time() < deadline:
        # Compute gradient
        grad = _gradient(x)
        
        # Projected gradient step
        x_new = x + lr * grad
        x_new = _project(x_new, total=np.sum(x))
        
        s_new = _score_numpy(x_new)
        
        if s_new > _score_numpy(x):
            x = x_new
            best_score = max(best_score, s_new)
            best_x = x.copy()
            # Increase learning rate slightly
            lr = min(lr * 1.05, 1.0)
        else:
            # Decrease learning rate
            lr *= 0.7
            if lr < min_lr:
                # Try a different random restart
                if time.time() < deadline - 20:
                    x = np.random.rand(N)
                    x = _project(x)
                    s = _score_numpy(x)
                    if s > best_score:
                        best_score = s
                        best_x = x.copy()
                    lr = 0.1
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final: return the best
    # Ensure nonnegative
    best_x = np.maximum(0, best_x)
    if np.sum(best_x) < 1e-12:
        best_x = np.ones(N)
    
    return list(best_x)

# EVOLVE-BLOCK-START
# (construct is defined above, this block is for clarity)
# EVOLVE-BLOCK-END
