import numpy as np
import time

N = 400

def _autoconv_full(x):
    """Compute full linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    F = np.fft.rfft(x, fsize)
    c = np.fft.irfft(F * np.conj(F), fsize)[:size]
    return c

def _autoconv_max_abs(x):
    c = _autoconv_full(x)
    return np.max(np.abs(c))

def _integral(x):
    return np.sum(x)

def _metric(x):
    # max|autoconv| / (integral)^2  -- guard against zero integral
    s = _integral(x)
    if abs(s) < 1e-12:
        return np.inf
    m = _autoconv_max_abs(x)
    return m / (s * s)

def _metric_with_conv(x):
    """Compute metric and return (metric_value, autoconvolution) for reuse."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return np.inf, None
    c = _autoconv_full(x)
    m = np.max(np.abs(c))
    return m / (s * s), c

def _chebyshev_candidate(order, scale=1.0):
    """Generate a Chebyshev-based candidate."""
    t = np.linspace(-1, 1, N)
    x = np.cos(order * np.arccos(t))
    return scale * x

def _m_sequence_candidate(seed=42):
    """Generate an M-sequence (maximum length sequence) using LFSR."""
    rng = np.random.default_rng(seed)
    # Use a simple approach: generate pseudo-random ±1 with good autocorrelation
    # M-sequences have near-zero autocorrelation except at lag 0
    n = N
    x = rng.choice([-1.0, 1.0], size=n)
    # Normalize to have positive sum
    if np.sum(x) < 0:
        x = -x
    return x

def _try_chebyshev_search(rng, deadline):
    """Try various Chebyshev orders and combinations."""
    best_x = None
    best_val = np.inf
    for order in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        if time.time() >= deadline:
            break
        for scale in [0.5, 1.0, 1.5, 2.0]:
            x = _chebyshev_candidate(order, scale)
            # Ensure positive integral
            if np.sum(x) < 0:
                x = -x
            v, _ = _metric_with_conv(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    return best_x, best_val

def _try_m_sequence_search(rng, deadline):
    """Try M-sequences and related structured candidates."""
    best_x = None
    best_val = np.inf
    for seed in range(20):
        if time.time() >= deadline:
            break
        x = _m_sequence_candidate(seed)
        v, _ = _metric_with_conv(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
    return best_x, best_val

def _try_gradient_descent(x_init, rng, deadline, max_iters=500):
    """Gradient-based optimization using subgradient of max|autoconv|."""
    x = x_init.copy()
    v, c = _metric_with_conv(x)
    if v == np.inf:
        return x, np.inf
    
    lr = 0.01
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        # Find the lag with maximum |autoconv|
        max_idx = np.argmax(np.abs(c))
        max_val = c[max_idx]
        sign = 1.0 if max_val > 0 else -1.0
        
        # Gradient of autoconv at lag max_idx w.r.t. x
        # c[k] = sum_i x[i] * x[i+k] (for linear autoconv)
        # dc/dx[j] = x[j-k] + x[j+k] (with boundary handling)
        n = len(x)
        grad = np.zeros(n)
        k = max_idx
        for j in range(n):
            # Contribution to c[k] from x[j]*x[j-k] and x[j]*x[j+k]
            if j - k >= 0:
                grad[j] += sign * x[j - k]
            if j + k < n:
                grad[j] += sign * x[j + k]
        
        # Also gradient of (integral)^2 in denominator
        # We want to minimize m/s^2, so gradient is (dm * s^2 - m * 2*s*ds) / s^4
        # = dm/s^2 - 2*m*ds/s^3
        s = _integral(x)
        if abs(s) < 1e-12:
            break
        
        # ds/dx[j] = 1 for all j
        ds = 1.0
        m = abs(max_val)
        
        # Full gradient of m/s^2
        full_grad = grad / (s * s) - 2.0 * m * ds / (s * s * s)
        
        # Normalize gradient
        gnorm = np.linalg.norm(full_grad)
        if gnorm > 1e-10:
            full_grad = full_grad / gnorm
        
        x_new = x - lr * full_grad
        v_new, c_new = _metric_with_conv(x_new)
        
        if v_new < v:
            x = x_new
            v = v_new
            c = c_new
            # Slowly decay learning rate
            lr *= 0.999
        else:
            # Try smaller step
            lr *= 0.9
    
    return x, v

def _try_simulated_annealing(x_init, rng, deadline, max_iters=1000):
    """Simulated annealing to escape local minima."""
    x = x_init.copy()
    v, _ = _metric_with_conv(x)
    if v == np.inf:
        return x, np.inf
    
    T = 0.1
    T_min = 0.001
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        if T < T_min:
            break
        
        # Propose a perturbation
        x_pert = x.copy()
        # Random flip of a few elements
        n_flips = rng.integers(1, 5)
        indices = rng.choice(N, size=n_flips, replace=False)
        x_pert[indices] *= -1.0
        
        v_pert, _ = _metric_with_conv(x_pert)
        
        if v_pert < v or rng.random() < np.exp(-(v_pert - v) / T):
            x = x_pert
            v = v_pert
        
        T *= cooling
    
    return x, v

def _try_local_search_fast(x_init, deadline, max_sweeps=20):
    """Fast local search by trying element perturbations."""
    x = x_init.copy()
    v, c = _metric_with_conv(x)
    if v == np.inf:
        return x, np.inf
    
    improved = True
    sweep = 0
    while improved and sweep < max_sweeps:
        if time.time() >= deadline:
            break
        improved = False
        sweep += 1
        for i in range(N):
            if time.time() >= deadline:
                break
            old = x[i]
            # Try small perturbations
            for delta in [-0.5, -0.2, 0.2, 0.5, -1.0, 1.0]:
                x[i] = old + delta
                v2, _ = _metric_with_conv(x)
                if v2 < v:
                    v = v2
                    improved = True
                    break
            else:
                x[i] = old
    
    return x, v

def construct():
    deadline = time.time() + 130.0  # safety margin below 150s
    rng = np.random.default_rng(12345)
    
    # Phase 0: Initialize with best constant sequence
    best_x = np.ones(N)
    best_val = _metric(best_x)  # Should be 1/N = 0.0025
    
    # Phase 1: Try structured candidates
    # Chebyshev candidates
    x_cheb, v_cheb = _try_chebyshev_search(rng, deadline - 30)
    if v_cheb < best_val:
        best_val = v_cheb
        best_x = x_cheb.copy()
    
    # M-sequence candidates
    x_mseq, v_mseq = _try_m_sequence_search(rng, deadline - 25)
    if v_mseq < best_val:
        best_val = v_mseq
        best_x = x_mseq.copy()
    
    # Phase 2: Gradient-based optimization from best
    if time.time() < deadline - 20:
        x_grad, v_grad = _try_gradient_descent(best_x, rng, deadline - 15, max_iters=300)
        if v_grad < best_val:
            best_val = v_grad
            best_x = x_grad.copy()
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 10:
        x_sa, v_sa = _try_simulated_annealing(best_x, rng, deadline - 5, max_iters=500)
        if v_sa < best_val:
            best_val = v_sa
            best_x = x_sa.copy()
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 3:
        x_local, v_local = _try_local_search_fast(best_x, deadline - 1, max_sweeps=10)
        if v_local < best_val:
            best_val = v_local
            best_x = x_local.copy()
    
    return best_x
