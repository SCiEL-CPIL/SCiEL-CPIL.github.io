"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 115.0  # safety margin before the 150 s kill
_N = N


def _overlaps_fft(h):
    """S(k) = sum_i h[i] * (1 - h[(i+k) mod N]) for all shifts k using FFT."""
    n = len(h)
    one_minus = 1.0 - h
    # Circular cross-correlation via FFT
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus)
    # Cross-correlation: ifft(conj(H) * O) gives sum_i h[i] * one_minus[(i+k) mod n]
    S = np.fft.ifft(np.conj(H) * O).real
    return S


def _overlaps_direct(h):
    """S(k) = sum_i h[i] * (1 - h[(i+k) mod N]) for all shifts k (direct)."""
    n = len(h)
    one_minus = 1.0 - h
    S = np.empty(n)
    for k in range(n):
        S[k] = np.dot(h, np.roll(one_minus, k))
    return S


def _softmax(S, beta):
    """Smooth max of S via log-sum-exp (scaled)."""
    m = np.max(S)
    return m + np.log(np.sum(np.exp(beta * (S - m)))) / beta


def _project(h, n):
    """Project h onto [0,1]^n with sum = n/2."""
    h = np.clip(h, 0.0, 1.0)
    h += (n / 2.0 - h.sum()) / n
    h = np.clip(h, 0.0, 1.0)
    h += (n / 2.0 - h.sum()) / n
    return h


def _gradient(h, S, beta, n):
    """Compute gradient of softmax(S, beta) w.r.t. h."""
    e = np.exp(beta * (S - np.max(S)))
    e = e / e.sum()
    g = np.zeros(n)
    for k in range(n):
        ek = e[k]
        if ek < 1e-12:
            continue
        g += ek * ((1.0 - np.roll(h, k)) - np.roll(h, -k))
    g -= g.mean()
    return g


def _line_search(h, g, lr, beta, n, max_iter=10):
    """Backtracking line search to find good step size."""
    S_curr = _overlaps_fft(h)
    val_curr = _softmax(S_curr, beta)
    for i in range(max_iter):
        lr_try = lr / (2 ** i)
        h_try = _project(h - lr_try * g, n)
        S_try = _overlaps_fft(h_try)
        val_try = _softmax(S_try, beta)
        if val_try < val_curr:
            return h_try, val_try
    return h, val_curr


def construct():
    t0 = time.time()
    n = _N

    best_h = np.full(n, 0.5)
    best_val = float('inf')
    
    # Multiple restarts with different random initializations
    rng = np.random.default_rng(42)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Generate initial point
        if restart == 0:
            h = np.full(n, 0.5)
            # Small random perturbation to break symmetry
            d = rng.standard_normal(n)
            d -= d.mean()
            h = _project(h + 0.1 * d / (np.linalg.norm(d) + 1e-12), n)
        else:
            # Larger random perturbation for diversity
            h = rng.uniform(0, 1, n)
            h = _project(h, n)
        
        # Adaptive gradient descent
        beta = 30.0
        lr = 0.05
        steps = 3000
        
        for it in range(steps):
            if time.time() > _DEADLINE - 5.0:
                break
            
            S = _overlaps_fft(h)
            val = _softmax(S, beta)
            
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            # Compute gradient
            g = _gradient(h, S, beta, n)
            
            # Line search
            h_new, val_new = _line_search(h, g, lr, beta, n)
            
            # Adaptive learning rate
            if val_new < val:
                lr = min(lr * 1.1, 0.2)  # Increase lr slightly
            else:
                lr = max(lr * 0.5, 0.001)  # Decrease lr
            
            h = h_new
        
        # Polish this restart's best
        if time.time() > _DEADLINE - 8.0:
            break
        
        h = best_h.copy()
        beta2 = 80.0
        lr2 = 0.01
        for it in range(1500):
            if time.time() > _DEADLINE - 5.0:
                break
            
            S = _overlaps_fft(h)
            val = _softmax(S, beta2)
            
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            g = _gradient(h, S, beta2, n)
            h_new, val_new = _line_search(h, g, lr2, beta2, n, max_iter=5)
            
            if val_new < val:
                lr2 = min(lr2 * 1.05, 0.05)
            else:
                lr2 = max(lr2 * 0.7, 0.0005)
            
            h = h_new
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 3.0:
        h = best_h.copy()
        beta3 = 150.0
        lr3 = 0.003
        for it in range(1000):
            if time.time() > _DEADLINE - 1.0:
                break
            
            S = _overlaps_fft(h)
            val = _softmax(S, beta3)
            
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            g = _gradient(h, S, beta3, n)
            h_new, val_new = _line_search(h, g, lr3, beta3, n, max_iter=3)
            
            if val_new < val:
                lr3 = min(lr3 * 1.02, 0.02)
            else:
                lr3 = max(lr3 * 0.8, 0.0001)
            
            h = h_new
    
    # Enforce exact sum and bounds one last time
    best_h = _project(best_h, n)
    
    return best_h
# EVOLVE-BLOCK-END
