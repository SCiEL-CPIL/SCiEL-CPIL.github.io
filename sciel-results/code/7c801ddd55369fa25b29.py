"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START


def _evaluate(h):
    """Compute the objective value for a given height vector."""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv)) / (s * s)


def _autoconv_fft(h):
    """Compute full autoconvolution using FFT."""
    n = len(h)
    n_fft = 2 * n - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * n - 1]


def _soft_max(x, k=50.0):
    """Soft-max via log-sum-exp for smooth optimization."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(k * (x - m)))) / k


def _grad_soft_max(x, k=50.0):
    """Gradient of the soft-max w.r.t. x."""
    m = np.max(x)
    e = np.exp(k * (x - m))
    return e / np.sum(e)


def _grad_autoconv_max(h, k=50.0):
    """
    Gradient of soft_max(autoconv(h)) w.r.t. h.
    
    autoconv_j = sum_i h_i * h_{j-i}
    d(autoconv_j)/d(h_l) = 2 * h_{j-l} if 0 <= j-l < N, else 0
    
    So grad_h[l] = 2 * sum_j w_j * h_{j-l} (valid indices)
    This is 2 * cross-correlation of w with h.
    """
    n = len(h)
    conv = _autoconv_fft(h)
    w = _grad_soft_max(conv, k=k)
    
    n_fft = 2 * n - 1
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    cross_corr = np.fft.irfft(W * np.conj(H), n=n_fft)
    
    return 2.0 * cross_corr[:n]


def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    
    # Start with the seed
    h = np.ones(N)
    best_h = h.copy()
    best_val = _evaluate(best_h)
    
    # Try structured initializations
    candidates = [np.ones(N)]
    
    # Symmetric patterns with varying amplitudes
    for period in [2, 4, 8, 16, 32]:
        for phase in range(period):
            c = np.ones(N)
            c[phase::period] = -1.0
            if np.sum(c) != 0:
                candidates.append(c)
    
    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(30):
        c = 1.0 + 0.3 * rng.randn(N)
        if abs(np.sum(c)) > 1.0:
            candidates.append(c)
    
    for _ in range(20):
        c = rng.randn(N)
        if abs(np.sum(c)) > 1.0:
            candidates.append(c)
    
    # Evaluate all candidates
    for c in candidates:
        c = c.astype(np.float64)
        val = _evaluate(c)
        if val < best_val:
            best_val = val
            best_h = c.copy()
    
    # Gradient-based optimization with scale invariance
    # Fix sum(h) = S (use S = N for normalization)
    S_target = float(N)
    
    h = best_h.copy()
    h = h * (S_target / np.sum(h))
    
    k_smooth = 20.0
    lr = 0.05
    momentum = 0.9
    v = np.zeros(N)
    
    for it in range(8000):
        if time.time() > deadline:
            break
        
        # Compute gradient
        grad = _grad_autoconv_max(h, k=k_smooth)
        
        # Momentum step
        v = momentum * v - lr * grad
        h_new = h + v
        
        # Project onto sum(h) = S_target
        h_new = h_new + (S_target - np.sum(h_new)) / N
        
        # Check improvement
        val = _evaluate(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive learning rate and smoothness
        if it % 200 == 199:
            if best_val > 2.0:
                lr *= 0.7
            elif best_val > 1.5:
                lr *= 0.9
            k_smooth = max(5.0, k_smooth * 0.9)
        
        # Occasionally restart with small perturbation to escape local minima
        if it % 1000 == 999:
            perturbation = 0.05 * rng.randn(N)
            h_trial = best_h + perturbation
            h_trial = h_trial * (S_target / np.sum(h_trial))
            val_trial = _evaluate(h_trial)
            if val_trial < best_val:
                best_val = val_trial
                best_h = h_trial.copy()
                h = best_h.copy()
                v = np.zeros(N)
    
    # Final polish: coordinate descent
    t1 = time.time()
    if t1 < deadline - 15:
        for trial in range(200):
            if time.time() > deadline - 5:
                break
            
            h_trial = best_h.copy()
            idx = rng.randint(N)
            delta = 0.01 * rng.randn()
            h_trial[idx] += delta
            
            h_trial = h_trial * (S_target / np.sum(h_trial))
            
            val = _evaluate(h_trial)
            if val < best_val:
                best_val = val
                best_h = h_trial.copy()
        
        # Multi-coordinate polish
        for trial in range(100):
            if time.time() > deadline - 5:
                break
            
            h_trial = best_h.copy()
            n_coords = rng.randint(2, 6)
            idxs = rng.choice(N, size=n_coords, replace=False)
            deltas = 0.01 * rng.randn(n_coords)
            h_trial[idxs] += deltas
            
            h_trial = h_trial * (S_target / np.sum(h_trial))
            
            val = _evaluate(h_trial)
            if val < best_val:
                best_val = val
                best_h = h_trial.copy()
    
    # Final evaluation
    final_val = _evaluate(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
# EVOLVE-BLOCK-END
