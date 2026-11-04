"""Erdos minimum overlap: minimize max overlap of h with shifted (1-h)."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 140  # seconds, safety margin before 150s kill

def _compute_overlaps(h):
    """Compute overlap values for all shifts k=1..N-1.
    overlap_k = sum_i h[i] * (1 - h[(i+k) % N])
    """
    h = np.asarray(h)
    overlaps = np.zeros(N)
    for k in range(1, N):
        shifted = np.roll(h, -k)
        overlaps[k] = np.sum(h * (1.0 - shifted))
    return overlaps

def _max_overlap(h):
    """Maximum overlap over all non-trivial shifts."""
    h = np.asarray(h)
    overlaps = _compute_overlaps(h)
    return np.max(overlaps[1:])

def _soft_max_overlap(h, beta=20.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    h = np.asarray(h)
    overlaps = _compute_overlaps(h)
    ov = overlaps[1:]  # exclude k=0
    # log-sum-exp for numerical stability
    m = np.max(ov)
    if m < 1e-30:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (ov - m)))) / beta

def _project(h):
    """Project onto {h in [0,1]^N : sum(h) = N/2}."""
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    target = N / 2.0
    if s > 0:
        h = h + (target - s) / N
    h = np.clip(h, 0.0, 1.0)
    # Fine adjustment for sum constraint
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute correction
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h

def _grad_soft_max_overlap(h, beta=20.0):
    """Gradient of soft-max overlap with respect to h."""
    h = np.asarray(h, dtype=float)
    overlaps = _compute_overlaps(h)
    ov = overlaps[1:]
    m = np.max(ov)
    if m < 1e-30:
        return np.zeros(N)
    weights = np.exp(beta * (ov - m)) / np.sum(np.exp(beta * (ov - m)))
    
    # d(overlap_k)/d(h[i]) = (1 - h[(i+k)%N]) + h[i] * d(1-h[(i+k)%N])/d(h[i])
    # = (1 - h[(i+k)%N]) - h[i] * delta(i == (j-k) for j)
    # Actually: overlap_k = sum_i h[i] * (1 - h[(i+k)%N])
    # d/dh[j] overlap_k = (1 - h[(j+k)%N]) + sum_i h[i] * (-delta(i+k=j))
    # = (1 - h[(j+k)%N]) - h[(j-k)%N]
    
    grad = np.zeros(N)
    for idx, k in enumerate(range(1, N)):
        w = weights[idx]
        # Contribution of shift k to gradient
        for j in range(N):
            grad[j] += w * ((1.0 - h[(j + k) % N]) - h[(j - k) % N])
    return grad

def _sturmian_init(alpha, N_pts):
    """Generate a Sturmian-like sequence projected to [0,1] with sum N/2."""
    # Use a quadratic irrational for quasi-periodic structure
    h = np.zeros(N_pts)
    for i in range(N_pts):
        # Sturmian word: floor((i+1)*alpha) - floor(i*alpha) gives 0/1
        # For continuous, use fractional part
        frac = (i * alpha) % 1.0
        h[i] = frac
    # Project to sum = N/2
    h = _project(h)
    return h

def _sinusoidal_init(freq, N_pts):
    """Sinusoidal initialization."""
    h = 0.5 + 0.5 * np.sin(2 * np.pi * freq * np.arange(N_pts) / N_pts)
    h = _project(h)
    return h

def _random_init(N_pts):
    """Random initialization projected to constraints."""
    h = np.random.uniform(0.0, 1.0, N_pts)
    h = _project(h)
    return h

def _binary_swap_local_search(h, max_iters=5000, beta=20.0):
    """Local search by swapping pairs to reduce max overlap."""
    h = np.array(h, dtype=float)
    best_val = _max_overlap(h)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        improved = False
        # Try random pair swaps
        for _ in range(20):
            i, j = np.random.choice(N, 2, replace=False)
            if h[i] == h[j]:
                continue
            # Swap
            h[i], h[j] = h[j], h[i]
            val = _max_overlap(h)
            if val < best_val - 1e-10:
                best_val = val
                improved = True
            else:
                h[i], h[j] = h[j], h[i]
        if not improved:
            # Try small perturbations
            i, j = np.random.choice(N, 2, replace=False)
            delta = 0.05 * (1.0 - abs(h[i] - h[j]))
            new_i = np.clip(h[i] + delta, 0, 1)
            new_j = np.clip(h[j] - delta, 0, 1)
            h[i], h[j] = new_i, new_j
            h = _project(h)
            val = _max_overlap(h)
            if val < best_val - 1e-10:
                best_val = val
                improved = True
            else:
                h = _project(h)
                # Revert
                h[i] = np.clip(h[i] - delta + (h[i] - new_i), 0, 1)
                h[j] = np.clip(h[j] + delta + (h[j] - new_j), 0, 1)
                h = _project(h)
        if not improved and it > 500:
            break
    
    return h, best_val

def _gradient_descent(h_init, beta=20.0, lr=0.01, max_iters=2000):
    """Projected gradient descent on soft-max overlap."""
    h = np.array(h_init, dtype=float)
    best_h = h.copy()
    best_val = _max_overlap(h)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 10:
            break
        grad = _grad_soft_max_overlap(h, beta)
        h_new = h - lr * grad
        h_new = _project(h_new)
        
        val = _max_overlap(h_new)
        if val < best_val - 1e-12:
            best_val = val
            best_h = h_new.copy()
        h = h_new
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.9
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    best_h = None
    best_score = float('inf')
    
    # Multiple initializations
    inits = []
    
    # Sturmian with golden ratio
    phi = (1 + np.sqrt(5)) / 2
    inits.append(_sturmian_init(1.0 / phi, N))
    inits.append(_sturmian_init(1.0 / (phi * phi), N))
    inits.append(_sturmian_init(0.618, N))
    inits.append(_sturmian_init(0.707, N))
    
    # Sinusoidal
    inits.append(_sinusoidal_init(1, N))
    inits.append(_sinusoidal_init(2, N))
    inits.append(_sinusoidal_init(3, N))
    inits.append(_sinusoidal_init(5, N))
    
    # Random
    for _ in range(3):
        inits.append(_random_init(N))
    
    # Uniform (baseline)
    inits.append(np.full(N, 0.5))
    
    for h_init in inits:
        if time.time() > DEADLINE - 15:
            break
        
        # Gradient descent polish
        h_gd, val_gd = _gradient_descent(h_init, beta=20.0, lr=0.005, max_iters=1500)
        if val_gd < best_score:
            best_score = val_gd
            best_h = h_gd.copy()
        
        # Binary swap local search
        h_ls, val_ls = _binary_swap_local_search(h_gd, max_iters=3000)
        if val_ls < best_score:
            best_score = val_ls
            best_h = h_ls.copy()
    
    # Final polish on best
    if best_h is not None and time.time() < DEADLINE - 5:
        h_final, val_final = _binary_swap_local_search(best_h, max_iters=5000)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Ensure exact sum = N/2
    best_h = _project(best_h)
    
    return best_h
