import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _time_remaining(deadline):
    return deadline - time.time()

def _compute_overlaps(h):
    """Compute all N circular overlaps of h with (1-h shifted by k)."""
    overlaps = np.empty(N)
    for k in range(N):
        shifted = np.roll(h, -k)
        overlaps[k] = np.sum(h * (1.0 - shifted))
    return overlaps

def _max_overlap_value(h):
    """Return the maximum overlap value."""
    return float(np.max(_compute_overlaps(h)))

def _soft_max_overlap(h, tau=0.05):
    """Softmax-smoothed max overlap for differentiable optimization."""
    overlaps = _compute_overlaps(h)
    # Softmax approximation to max
    scaled = overlaps / tau
    max_s = np.max(scaled)
    exps = np.exp(scaled - max_s)
    soft_max = max_s + tau * np.log(np.sum(exps))
    return float(soft_max)

def _soft_max_overlap_grad(h, tau=0.05):
    """Gradient of softmax-smoothed max overlap."""
    overlaps = _compute_overlaps(h)
    scaled = overlaps / tau
    max_s = np.max(scaled)
    exps = np.exp(scaled - max_s)
    weights = exps / np.sum(exps)
    
    # d(overlap_k)/d(h) for each k
    grad = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, -k)
        # overlap_k = sum_i h[i] * (1 - h[(i+k)%N])
        # d/dh[j] = (1 - h[(j+k)%N]) - h[j-k mod N] * 1... 
        # More carefully: overlap_k = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
        # d/dh[j] = 1 - h[(j+k)%N] - h[(j-k)%N] (from the second term)
        # Wait, let me be more careful:
        # overlap_k = sum_i h[i] * (1 - h[(i+k)%N])
        # = sum_i h[i] - sum_i h[i] * h[(i+k)%N]
        # d/dh[j] of first term: 1
        # d/dh[j] of second term: -h[(j+k)%N] - h[(j-k)%N] (two contributions when i=j and when i+k=j)
        # So d/dh[j] = 1 - h[(j+k)%N] - h[(j-k)%N]
        # But wait, that's not quite right either. Let me reconsider.
        # overlap_k = sum_i h[i] * (1 - h[(i+k)%N])
        # For a fixed j, h[j] appears in:
        # - The term where i=j: h[j] * (1 - h[(j+k)%N])
        # - The term where (i+k)%N = j, i.e., i = (j-k)%N: h[(j-k)%N] * (1 - h[j])
        # So d/dh[j] = (1 - h[(j+k)%N]) + h[(j-k)%N] * (-1)
        # = 1 - h[(j+k)%N] - h[(j-k)%N]
        
        # Actually, I think I need to use FFT-based approach or just compute directly
        # Let me use a simpler approach: compute gradient numerically or use the structure
    
    # Use FFT-based gradient computation
    # overlap_k = sum_i h[i] * (1 - h[(i+k)%N]) = N/2 - autocorr_k
    # where autocorr_k = sum_i h[i] * h[(i+k)%N]
    # autocorr_k can be computed via FFT
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[k] = sum_i h[i] * h[(i-k)%N] (circular)
    # So overlap_k = N/2 - autocorr[k] (with appropriate indexing)
    
    # d(autocorr_k)/d(h[j]) = h[(j-k)%N] + h[(j+k)%N] for k != 0
    # For k=0: d(autocorr_0)/d(h[j]) = 2*h[j]
    
    # Let me just compute the gradient of each overlap_k w.r.t. h directly
    grad = np.zeros(N)
    for k in range(N):
        # overlap_k = sum_i h[i] * (1 - h[(i+k)%N])
        # d/dh[j] = (1 - h[(j+k)%N]) - h[(j-k)%N]
        # (the -h[(j-k)%N] comes from the second factor when i such that (i+k)%N = j)
        contrib = 1.0 - h[(j + k) % N] if False else None  # placeholder
        
    # Simpler: just compute gradient numerically for the weighted sum
    # grad = sum_k weights[k] * d(overlap_k)/d(h)
    # d(overlap_k)/d(h[j]) = (1 - h[(j+k)%N]) - h[(j-k)%N]
    # for k=0: d(overlap_0)/d(h[j]) = 1 - h[j] - h[j] = 1 - 2*h[j]
    
    grad = np.zeros(N)
    for k in range(N):
        w = weights[k]
        if w < 1e-12:
            continue
        if k == 0:
            grad += w * (1.0 - 2.0 * h)
        else:
            for j in range(N):
                grad[j] += w * (1.0 - h[(j + k) % N] - h[(j - k) % N])
    
    return grad

def _project_to_feasible(h):
    """Project h onto {x in [0,1]^N : sum(x) = N/2} using bisection water-filling."""
    target_sum = N / 2.0
    current_sum = np.sum(h)
    
    if abs(current_sum - target_sum) < 1e-12:
        return np.clip(h, 0.0, 1.0)
    
    # Bisection on the Lagrange multiplier for the sum constraint
    # We want to find t such that sum(clip(h - t, 0, 1)) = target_sum
    
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        projected = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(projected)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    
    t = (lo + hi) / 2.0
    projected = np.clip(h - t, 0.0, 1.0)
    
    # Fine-tune to get exact sum
    diff = target_sum - np.sum(projected)
    if abs(diff) > 1e-10:
        # Adjust elements that are not at bounds
        not_at_bounds = (projected > 1e-8) & (projected < 1.0 - 1e-8)
        n_not_bound = np.sum(not_at_bounds)
        if n_not_bound > 0:
            adj = diff / n_not_bound
            projected[not_at_bounds] += adj
            projected = np.clip(projected, 0.0, 1.0)
    
    return projected

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    # Initial guess: uniform 0.5
    best_h = np.full(N, 0.5)
    best_score = _max_overlap_value(best_h)
    
    np.random.seed(42)
    
    # Try multiple restarts with different initializations
    for restart in range(30):
        if _time_remaining(deadline) < 5.0:
            break
        
        if restart == 0:
            h_init = np.full(N, 0.5)
        elif restart == 1:
            # Alternating pattern
            h_init = np.zeros(N)
            h_init[::2] = 1.0
        elif restart == 2:
            # Fourier-based initialization: low-frequency components
            h_init = np.zeros(N)
            for freq in range(1, 5):
                amp = 0.3 / freq
                h_init += amp * np.cos(2 * np.pi * freq * np.arange(N) / N)
            h_init += 0.5
            h_init = np.clip(h_init, 0.0, 1.0)
            h_init = _project_to_feasible(h_init)
        else:
            # Random smooth initialization
            freqs = np.random.choice(range(1, 10), size=3, replace=False)
            phases = np.random.uniform(0, 2 * np.pi, size=3)
            amps = np.random.uniform(0.1, 0.4, size=3)
            h_init = np.full(N, 0.5)
            for f, p, a in zip(freqs, phases, amps):
                h_init += a * np.cos(2 * np.pi * f * np.arange(N) / N + p)
            h_init = np.clip(h_init, 0.0, 1.0)
            h_init = _project_to_feasible(h_init)
        
        # Gradient descent with softmax-smoothed max overlap
        h = h_init.copy()
        tau = 0.05  # softmax temperature
        lr = 0.1
        n_iter = 200
        
        for it in range(n_iter):
            if _time_remaining(deadline) < 3.0:
                break
            
            # Compute gradient of soft max overlap
            overlaps = _compute_overlaps(h)
            scaled = overlaps / tau
            max_s = np.max(scaled)
            exps = np.exp(scaled - max_s)
            weights = exps / np.sum(exps)
            
            # Compute gradient
            grad = np.zeros(N)
            for k in range(N):
                w = weights[k]
                if w < 1e-12:
                    continue
                if k == 0:
                    grad += w * (1.0 - 2.0 * h)
                else:
                    grad += w * (1.0 - np.roll(h, k) - np.roll(h, -k))
            
            # Gradient step
            h = h - lr * grad
            h = _project_to_feasible(h)
            
            # Decrease learning rate and temperature
            lr *= 0.995
            tau *= 0.999
        
        # Evaluate
        score = _max_overlap_value(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Final refinement with smaller tau
    if _time_remaining(deadline) > 10.0:
        h = best_h.copy()
        tau = 0.01
        lr = 0.01
        for it in range(300):
            if _time_remaining(deadline) < 3.0:
                break
            
            overlaps = _compute_overlaps(h)
            scaled = overlaps / tau
            max_s = np.max(scaled)
            exps = np.exp(scaled - max_s)
            weights = exps / np.sum(exps)
            
            grad = np.zeros(N)
            for k in range(N):
                w = weights[k]
                if w < 1e-12:
                    continue
                if k == 0:
                    grad += w * (1.0 - 2.0 * h)
                else:
                    grad += w * (1.0 - np.roll(h, k) - np.roll(h, -k))
            
            h = h - lr * grad
            h = _project_to_feasible(h)
            lr *= 0.998
            tau *= 0.9995
        
        score = _max_overlap_value(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Ensure exact constraints
    best_h = _project_to_feasible(best_h)
    
    return best_h
