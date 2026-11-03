"""Erdos minimum overlap - equal-width step heights on [0,2].

We choose N step heights h_i in [0,1] with sum = N/2 (integral 1 over [0,2]).
Goal: minimize max_shift of overlap between h and its shifted complement (1-h).

The overlap at shift s is sum_i h_i * (1 - h_{i-s}) for the shifted indices.
We minimize the maximum over all shifts.
"""

N = 100
import numpy as np
import time

def _compute_overlap(h):
    """Compute max overlap of h with shifted 1-h over all cyclic shifts."""
    # For each shift s, overlap_s = sum_i h[i] * (1 - h[(i-s) % N])
    # = sum_i h[i] - sum_i h[i] * h[(i-s) % N]
    # = N/2 - autocorrelation at lag s
    # So minimizing max overlap = minimizing max(N/2 - acorr[s]) = maximizing min(acorr[s])
    # Wait, that's not quite right. Let me reconsider.
    
    # overlap at shift s: sum_i h[i] * (1 - h[(i-s) mod N])
    # = sum_i h[i] - sum_i h[i]*h[(i-s) mod N]
    # = N/2 - acorr[s]
    
    # So max overlap = N/2 - min(acorr[s])
    # We want to minimize this, i.e., maximize min(acorr[s])
    
    # Compute autocorrelation for all lags
    n = len(h)
    # acorr[s] = sum_i h[i] * h[(i-s) % n]
    # This is the circular cross-correlation with itself
    # Use FFT for efficiency
    H = np.fft.rfft(h, n=2*n)
    acorr_full = np.fft.irfft(H * np.conj(H), n=2*n)[:n]
    # acorr_full[s] for s in 0..n-1 gives sum_i h[i]*h[(i-s)%n]
    
    return n/2.0 - np.min(acorr_full)

def _random_feasible():
    """Generate a random feasible h: sum = N/2, each in [0,1]."""
    # Start with random values, then adjust
    h = np.random.dirichlet(np.ones(N)) * N  # sums to N, each positive
    # Scale to sum N/2
    h = h / h.sum() * (N / 2.0)
    # Clip to [0,1] and renormalize
    h = np.clip(h, 0, 1)
    diff = N/2.0 - h.sum()
    if diff > 0:
        # Add to smallest elements
        idx = np.argsort(h)
        for i in idx:
            room = 1.0 - h[i]
            add = min(room, diff)
            h[i] += add
            diff -= add
            if diff <= 1e-12:
                break
    else:
        # Remove from largest elements
        idx = np.argsort(h)[::-1]
        for i in idx:
            rem = min(h[i], -diff)
            h[i] -= rem
            diff += rem
            if diff >= -1e-12:
                break
    # Final projection
    h = np.clip(h, 0, 1)
    # Adjust to exact sum
    diff = N/2.0 - h.sum()
    h += diff / N
    h = np.clip(h, 0, 1)
    # Small residual fix
    h[np.argmin(h)] += (N/2.0 - h.sum())
    return h

def _gradient_step(h, lr=0.01):
    """One gradient descent step on max overlap."""
    # Compute overlap at each shift
    n = len(h)
    H = np.fft.rfft(h, n=2*n)
    acorr_full = np.fft.irfft(H * np.conj(H), n=2*n)[:n]
    overlap = n/2.0 - acorr_full  # overlap[s] = N/2 - acorr[s]
    
    # We want to minimize max(overlap). Use subgradient: gradient w.r.t. h_i
    # overlap[s] = sum_j h[j] * (1 - h[(j-s)%n])
    # d overlap[s] / d h[i] = sum_j [delta_{j,i}*(1-h[(j-s)%n]) + h[j]*(-delta_{(j-s)%n, i})]
    #                        = (1 - h[(i-s)%n]) + sum_j h[j]*(-1 if (j-s)%n == i else 0)
    #                        = (1 - h[(i-s)%n]) - h[(i+s)%n]
    
    # For the max, use the shifts that achieve (or near) the maximum
    max_ov = np.max(overlap)
    # Soft max: weight shifts by how close they are to max
    eps = 0.01
    weights = np.exp((overlap - max_ov) / eps)
    weights /= weights.sum()
    
    # Gradient of weighted sum of overlaps
    # grad_h[i] = sum_s weights[s] * (1 - h[(i-s)%n] - h[(i+s)%n])
    grad = np.zeros(n)
    for s in range(n):
        w = weights[s]
        if w < 1e-10:
            continue
        # (i-s) % n and (i+s) % n for all i
        grad += w * (1.0 - np.roll(h, s) - np.roll(h, -s))
    
    # Constraint: sum(h) = N/2, h in [0,1]
    # Project gradient onto sum constraint: subtract mean
    grad -= np.mean(grad)
    
    h_new = h - lr * grad
    # Project to [0,1] and sum constraint
    h_new = np.clip(h_new, 0, 1)
    diff = N/2.0 - h_new.sum()
    h_new += diff / n
    h_new = np.clip(h_new, 0, 1)
    # Final fix
    h_new[np.argmin(h_new)] += (N/2.0 - h_new.sum())
    return h_new

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _compute_overlap(best_h)
    
    # Try a few different initializations
    initializations = [
        np.full(N, 0.5),
        _random_feasible(),
        np.random.choice([0.0, 1.0], size=N) * 0.5 + 0.25,  # mixed
    ]
    # Ensure feasibility for random ones
    for idx in range(1, len(initializations)):
        h = initializations[idx]
        h = np.clip(h, 0, 1)
        diff = N/2.0 - h.sum()
        h += diff / N
        h = np.clip(h, 0, 1)
        h[np.argmin(h)] += (N/2.0 - h.sum())
        initializations[idx] = h
    
    for h0 in initializations:
        if time.time() > deadline - 10:
            break
        
        h = h0.copy()
        score = _compute_overlap(h)
        
        # Gradient descent with adaptive learning rate
        lr = 0.05
        for step in range(5000):
            if time.time() > deadline - 5:
                break
            
            h = _gradient_step(h, lr)
            
            # Adaptive: reduce lr if score doesn't improve
            new_score = _compute_overlap(h)
            if new_score < score - 1e-8:
                score = new_score
                # Keep lr or slightly increase
                lr = min(lr * 1.1, 0.5)
            else:
                lr *= 0.5
            
            if score < best_score - 1e-9:
                best_score = score
                best_h = h.copy()
            
            # Early stop if converged
            if lr < 1e-6:
                break
    
    # Return best found
    # Ensure exact feasibility
    best_h = np.clip(best_h, 0, 1)
    diff = N/2.0 - best_h.sum()
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    best_h[np.argmin(best_h)] += (N/2.0 - best_h.sum())
    
    return best_h
