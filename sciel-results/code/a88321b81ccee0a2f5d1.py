import time
import numpy as np

N = 600
DEADLINE = 145.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _start_time)

def _autoconv_peak_ratio(heights):
    """
    Compute the normalized maximum of the autoconvolution.
    Metric: N * max_k (sum_i h_i * h_{i+k}) / (sum_i h_i)^2
    For uniform, this equals 1.0.
    """
    h = np.asarray(heights, dtype=np.float64)
    total = h.sum()
    if total <= 0:
        return float('inf')
    # Use FFT for fast convolution
    size = 2 * N
    f = np.fft.rfft(h, n=size)
    c = np.fft.irfft(f * f, n=size)
    max_conv = np.max(c)
    return N * max_conv / (total * total)

def _score(h):
    """Objective function for optimization."""
    h = np.maximum(h, 0.0)
    return _autoconv_peak_ratio(h)

def _gradient(h):
    """
    Compute gradient of the objective function.
    Uses the fact that the max of convolution occurs at some lag,
    and the gradient with respect to h is the flipped h shifted by that lag.
    """
    h = np.maximum(h, 0.0)
    total = h.sum()
    if total <= 0:
        return np.zeros_like(h)
    
    size = 2 * N
    f = np.fft.rfft(h, n=size)
    c = np.fft.irfft(f * f, n=size)
    
    m_star = int(np.argmax(c))
    peak = float(c[m_star])
    
    # The convolution c[k] = sum_i h[i] * h[i + k - (N-1)]
    # d(c[m_star])/d(h[j]) = h[j - (m_star - (N-1))] if valid
    offset = m_star - (N - 1)
    g = np.zeros_like(h)
    
    if offset >= 0:
        lo = offset
        hi = N
        if lo < hi:
            g[lo:hi] = h[lo - offset:hi - offset]
    else:
        lo = 0
        hi = N + offset
        if lo < hi:
            g[lo:hi] = h[lo - offset:hi - offset]
    
    # Quotient rule for N * peak / total^2
    # d/dh = N * (g * total^2 - peak * 2 * total * h) / total^4
    grad = N * (g * total * total - peak * 2.0 * total * h) / (total ** 4)
    return grad

def _normalize_mass(h, target_mass):
    """Normalize h to have the given total mass."""
    s = h.sum()
    if s <= 1e-12:
        return h
    return h * (target_mass / s)

def _initialization_asymmetric(seed):
    """Generate an asymmetric smooth profile."""
    rng = np.random.RandomState(seed)
    h = np.ones(N)
    x = np.linspace(0, 1, N)
    
    # Add several sine waves with random amplitudes and phases
    for freq in [1, 2, 3, 5, 7]:
        amp = rng.uniform(0.1, 0.3)
        phase = rng.uniform(0, 2 * np.pi)
        h += amp * np.sin(2 * np.pi * freq * x + phase)
    
    # Add small noise
    h += rng.normal(0, 0.02, N)
    
    h = np.maximum(h, 0.0)
    return _normalize_mass(h, N)

def _gradient_descent(h0, max_iters=1000, lr_init=0.1):
    """
    Gradient descent with mass preservation and adaptive learning rate.
    """
    h = h0.copy()
    best_h = h.copy()
    best_val = _score(h)
    lr = lr_init
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        grad = _gradient(h)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        h_new = h - lr * grad
        h_new = np.maximum(h_new, 0.0)
        h_new = _normalize_mass(h_new, h.sum())
        
        val = _score(h_new)
        
        if val < best_val - 1e-12:
            best_val = val
            best_h = h_new.copy()
            h = h_new
            # Increase learning rate slightly if making good progress
            lr = min(lr * 1.01, 0.5)
        else:
            # Decrease learning rate
            lr *= 0.95
            h = h_new
        
        if lr < 1e-8:
            break
    
    return best_h, best_val

def _simulated_annealing(h0, max_iters=200, t0=0.5, t1=0.01):
    """
    Simulated annealing for fine-tuning.
    """
    h = h0.copy()
    best_h = h.copy()
    best_val = _score(h)
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        T = t0 * (t1 / t0) ** (it / max_iters)
        
        # Random perturbation
        h_new = h.copy()
        n_perturb = rng.randint(10, 50)
        indices = rng.choice(N, size=n_perturb, replace=False)
        for idx in indices:
            delta = rng.normal(0, 0.1 * np.mean(h))
            h_new[idx] = max(0.0, h_new[idx] + delta)
        
        h_new = _normalize_mass(h_new, h.sum())
        val_new = _score(h_new)
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            h = h_new
        else:
            delta = val_new - best_val
            if delta > 0 and rng.random() < np.exp(-delta / (T + 1e-10)):
                h = h_new
    
    return best_h, best_val

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with uniform
    best_h = np.ones(N)
    best_score = _autoconv_peak_ratio(best_h)
    
    # Phase 1: Try multiple asymmetric initializations with gradient descent
    n_initials = 30
    for seed in range(n_initials):
        if _time_left() < 15.0:
            break
        
        h0 = _initialization_asymmetric(seed)
        h_gd, val_gd = _gradient_descent(h0, max_iters=500, lr_init=0.1)
        
        if val_gd < best_score - 1e-12:
            best_score = val_gd
            best_h = h_gd.copy()
    
    # Phase 2: Refine the best solution with more gradient descent
    if _time_left() > 20.0:
        h_refined, val_refined = _gradient_descent(best_h, max_iters=1000, lr_init=0.05)
        if val_refined < best_score - 1e-12:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Phase 3: Simulated annealing polish
    if _time_left() > 15.0:
        h_sa, val_sa = _simulated_annealing(best_h, max_iters=300, t0=0.3, t1=0.005)
        if val_sa < best_score - 1e-12:
            best_score = val_sa
            best_h = h_sa.copy()
    
    # Safety: never return worse than uniform
    uniform_score = _autoconv_peak_ratio(np.ones(N))
    if best_score > uniform_score + 1e-9:
        best_h = np.ones(N)
        best_score = uniform_score
    
    # Ensure nonnegativity and return as list
    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]

_start_time = 0.0
