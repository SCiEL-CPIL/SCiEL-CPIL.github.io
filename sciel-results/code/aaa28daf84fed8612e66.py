"""Second autocorrelation inequality: maximize ||h||2^2 / (||h||1 * ||h*h||inf).

Structural change: reformulate h_i = x_i^2 for x in R^50, turning the non-negative
constraint into a smooth optimization. Use a stochastic subgradient method that
samples the active lag of the autoconvolution peak to compute gradients efficiently.
This is a game-theoretic approach: we minimize the peak relative to energy by
pushing mass away from the currently-active lag.
"""
import time
import numpy as np

N = 50
DEADLINE = 140.0  # safety margin under the 150 s kill


def _autoconv(h):
    """Full linear autoconvolution (non-circular), h*h with h[i]h[i+k]."""
    return np.convolve(h, h, mode='full')


def _ratio(h):
    """||h||2^2 / (||h||1 * ||h*h||inf). Returns 0 for degenerate inputs."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 1e-300:
        return 0.0
    l2sq = float(np.dot(h, h))
    peak = float(np.max(np.abs(_autoconv(h))))
    if peak <= 1e-300:
        return 0.0
    return l2sq / (l1 * peak)


def _stochastic_subgradient(x, rng, step_size, n_samples):
    """
    One stochastic subgradient step for minimizing -ratio(h) where h = x^2.
    
    We sample the active lag of the autoconvolution peak and compute the
    subgradient of the peak term with respect to x.
    """
    h = x ** 2
    ac = _autoconv(h)
    peak_idx = np.argmax(np.abs(ac))
    peak_val = abs(ac[peak_idx])
    
    if peak_val <= 1e-300:
        return x, 0.0
    
    # The autoconvolution at lag k is sum_i h[i]*h[i+k] (for k >= 0)
    # For the full convolution, lag j corresponds to index j in ac
    # ac[j] = sum_{i} h[i] * h[j - i] where 0 <= i <= N-1 and 0 <= j-i <= N-1
    
    # Subgradient of peak w.r.t. h:
    # If peak at lag L (0-indexed in full conv, so actual lag = L - (N-1))
    # ac[L] = sum_{i=max(0,L-(N-1))}^{min(N-1,L)} h[i] * h[L-i]
    # d(ac[L])/d(h[i]) = h[L-i] if valid, else 0
    
    L = peak_idx
    grad_h = np.zeros(N)
    for i in range(N):
        j = L - i
        if 0 <= j < N:
            grad_h[i] += h[j]
    
    # Chain rule: d(h[i])/d(x[i]) = 2*x[i]
    grad_x = 2.0 * x * grad_h
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad_x)
    if grad_norm > 1e-300:
        grad_x = grad_x / grad_norm
    
    # Also add a small regularization to keep x from collapsing
    reg_grad = 0.01 * x
    
    total_grad = grad_x + reg_grad
    
    # Update: x_new = x - step_size * total_grad
    x_new = x - step_size * total_grad
    
    # Compute ratio for tracking
    h_new = x_new ** 2
    val = _ratio(h_new)
    
    return x_new, val


def _optimize_from_x0(x0, rng, max_iter=5000):
    """Run stochastic subgradient optimization from initial x0."""
    t_start = time.time()
    x = x0.copy()
    h = x ** 2
    best_x = x.copy()
    best_val = _ratio(h)
    
    # Adaptive step size
    base_step = 0.1
    n_samples = 1
    
    for it in range(max_iter):
        if time.time() - t_start > 10.0:  # per-iteration time limit
            break
        
        x, val = _stochastic_subgradient(x, rng, base_step, n_samples)
        
        if val > best_val:
            best_val = val
            best_x = x.copy()
        
        # Anneal step size
        base_step *= 0.999
        
        # Occasionally add noise for exploration
        if it % 50 == 0 and it > 0:
            x += 0.01 * rng.standard_normal(N)
    
    return best_x, best_val


def _structured_x0(seed):
    """Generate structured initial points in x-space."""
    rng = np.random.default_rng(seed)
    x = np.zeros(N)
    
    strategy = seed % 5
    
    if strategy == 0:
        # Sparse: few large spikes
        k = rng.integers(3, 8)
        idx = rng.choice(N, size=k, replace=False)
        x[idx] = rng.uniform(0.5, 2.0, size=k)
        
    elif strategy == 1:
        # Clustered: consecutive indices
        start = rng.integers(0, N - 10)
        k = rng.integers(5, 12)
        idx = np.arange(start, min(start + k, N))
        x[idx] = rng.uniform(0.3, 1.5, size=len(idx))
        
    elif strategy == 2:
        # Arithmetic progression
        start = rng.integers(0, N // 2)
        step = rng.integers(2, 5)
        idx = list(range(start, N, step))
        if len(idx) < 3:
            idx = list(range(0, N, 3))
        x[idx] = rng.uniform(0.4, 1.2, size=len(idx))
        
    elif strategy == 3:
        # Symmetric pattern
        center = N // 2
        for offset in range(0, N // 2, 2):
            if center - offset >= 0:
                x[center - offset] = rng.uniform(0.3, 1.0)
            if center + offset < N:
                x[center + offset] = rng.uniform(0.3, 1.0)
        
    else:
        # Random dense
        x = rng.uniform(0.1, 1.0, size=N)
    
    # Ensure non-negative (x can be negative, h = x^2 is always non-negative)
    return x


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    best_x = np.zeros(N)
    best_x[0] = 1.0
    best_h = best_x ** 2
    best_val = _ratio(best_h)
    
    # Try multiple structured initializations
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > DEADLINE:
            break
        
        x0 = _structured_x0(i * 7 + 1)
        
        # Also try a random initialization
        if i % 2 == 1:
            x0 = rng.standard_normal(N) * 0.5
        
        x_opt, val = _optimize_from_x0(x0, rng, max_iter=3000)
        
        if val > best_val:
            best_val = val
            best_x = x_opt
            best_h = best_x ** 2
    
    # Final local refinement on the best solution
    if time.time() < DEADLINE - 5.0:
        x_refine = best_x.copy()
        for _ in range(1000):
            if time.time() > DEADLINE:
                break
            x_refine, val = _stochastic_subgradient(x_refine, rng, 0.01, 1)
            h_refine = x_refine ** 2
            if val > best_val:
                best_val = val
                best_h = h_refine
    
    # Ensure valid output
    if best_val <= 0.0:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    # Clip to ensure non-negative
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
