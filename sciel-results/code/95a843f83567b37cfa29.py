import numpy as np
import time

N = 400
_DEADLINE = 140.0  # wall-clock limit with safety margin

def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _project_sum_to_one(h):
    """Project h onto the hyperplane sum(h) = 1."""
    s = np.sum(h)
    if s == 0:
        return h
    return h / s

def _compute_grads(h):
    """
    Compute gradients of the autoconvolution entries with respect to h.
    conv[k] = sum_j h[j] h[k-j]
    d conv[k] / d h[i] = h[k-i] + h[i-k] (with appropriate bounds)
    
    We only care about the gradient of max(conv). Since max is non-differentiable,
    we use the gradient at the argmax index.
    """
    conv = np.convolve(h, h)
    max_idx = np.argmax(conv)
    
    # Gradient of conv[max_idx] w.r.t h[i]
    # conv[max_idx] = sum_{j} h[j] h[max_idx - j]
    # d/dh[i] = h[max_idx - i] + h[i - max_idx] (if in range)
    
    grad = np.zeros(N)
    for i in range(N):
        j1 = max_idx - i
        j2 = i - max_idx
        val = 0.0
        if 0 <= j1 < N:
            val += h[j1]
        if 0 <= j2 < N:
            val += h[j2]
        grad[i] = val
    return grad

def _spectral_init(rng):
    """
    Generate an initial h using a sparse spectral representation.
    We pick a small number of random frequency bins and assign random amplitudes.
    This creates a structured signal with potentially low autocorrelation sidelobes.
    """
    n_freqs = rng.integers(3, 10)
    freqs = rng.choice(N, size=n_freqs, replace=False)
    phases = rng.uniform(0, 2 * np.pi, size=n_freqs)
    amps = rng.normal(0, 1, size=n_freqs)
    
    # Build h via IDFT of sparse spectrum
    H = np.zeros(N, dtype=complex)
    for f, p, a in zip(freqs, phases, amps):
        H[f] = a * np.exp(1j * p)
        # Ensure Hermitian symmetry for real h
        H[N - f] = np.conj(H[f])
    
    h = np.real(np.fft.ifft(H))
    h = _project_sum_to_one(h)
    return h

def _gradient_descent(h, steps=5000, lr_schedule=True):
    """
    Perform projected gradient descent to minimize max(conv).
    We minimize f(h) = max(conv(h)) subject to sum(h) = 1.
    """
    best_h = h.copy()
    best_val = _evaluate(h)
    
    for step in range(steps):
        if time.time() > _DEADLINE - 5:
            break
            
        conv = np.convolve(h, h)
        max_val = np.max(conv)
        max_idx = np.argmax(conv)
        
        # Compute gradient of conv[max_idx] w.r.t h
        grad = np.zeros(N)
        for i in range(N):
            j1 = max_idx - i
            j2 = i - max_idx
            val = 0.0
            if 0 <= j1 < N:
                val += h[j1]
            if 0 <= j2 < N:
                val += h[j2]
            grad[i] = val
        
        # Learning rate decay
        if lr_schedule:
            lr = 0.01 / (1 + step * 0.001)
        else:
            lr = 0.01
        
        # Gradient step
        h_new = h - lr * grad
        
        # Project onto sum(h) = 1
        h_new = _project_sum_to_one(h_new)
        
        # Check if improved
        val_new = _evaluate(h_new)
        if np.isfinite(val_new) and val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
        
        h = h_new
    
    return best_h, best_val

def _refine_greedy(h, budget=20000):
    """
    Greedy local search: randomly perturb individual elements and accept if improved.
    """
    best_h = h.copy()
    best_val = _evaluate(h)
    rng = np.random.default_rng(42)
    
    for _ in range(budget):
        if time.time() > _DEADLINE - 2:
            break
            
        idx = rng.integers(0, N)
        h_new = best_h.copy()
        h_new[idx] += rng.normal(0, 0.05)
        
        # Project to sum = 1
        h_new = _project_sum_to_one(h_new)
        
        val_new = _evaluate(h_new)
        if np.isfinite(val_new) and val_new < best_val - 1e-10:
            best_val = val_new
            best_h = h_new
    
    return best_h, best_val

def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Best solution so far
    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # Should be 2.0
    
    rng = np.random.default_rng(123)
    
    # Phase 1: Generate multiple spectral initializations and run gradient descent
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        h_init = _spectral_init(rng)
        h_opt, val_opt = _gradient_descent(h_init, steps=2000)
        
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Phase 2: Refine the best solution with greedy local search
    if time.time() < deadline - 10:
        best_h, best_val = _refine_greedy(best_h, budget=30000)
    
    # Phase 3: Try a few more spectral inits if time remains
    if time.time() < deadline - 5:
        for i in range(5):
            if time.time() > deadline - 3:
                break
            h_init = _spectral_init(rng)
            h_opt, val_opt = _gradient_descent(h_init, steps=1000)
            if np.isfinite(val_opt) and val_opt < best_val:
                best_val = val_opt
                best_h = h_opt
                # Quick refinement
                if time.time() < deadline - 2:
                    best_h, best_val = _refine_greedy(best_h, budget=5000)
    
    # Final validation
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
    
    return best_h
