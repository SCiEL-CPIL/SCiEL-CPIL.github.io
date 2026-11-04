import numpy as np
import time

N = 100
DEADLINE = 140.0  # safety margin before 150s limit

def _overlap_max(h):
    """Compute max over all shifts k of sum_i h_i * (1 - h_{i+k})."""
    # autocorr[k] = sum_i h_i * h_{i+k} for k = 0..N-1
    # overlap[k] = N/2 - autocorr[k]
    h = np.asarray(h, dtype=np.float64)
    autocorr = np.correlate(h, h, mode='full')
    # 'full' mode gives indices -(N-1)..N-1; we want k=0..N-1 which is indices N-1..2N-2
    autocorr = autocorr[N-1:2*N-1]
    overlap = N / 2.0 - autocorr
    return np.max(overlap)

def _smooth_max(overlap, temperature=0.01):
    """Log-sum-exp approximation of max."""
    m = np.max(overlap)
    if m < -1e10:
        return 0.0
    return m + temperature * np.log(np.sum(np.exp((overlap - m) / temperature)))

def _smooth_max_grad(overlap, temperature=0.01):
    """Value and gradient of log-sum-exp."""
    m = np.max(overlap)
    if m < -1e10:
        return 0.0, np.zeros_like(overlap)
    exp_terms = np.exp((overlap - m) / temperature)
    s = np.sum(exp_terms)
    val = m + temperature * np.log(s)
    grad = exp_terms / s
    return val, grad

def _autocorr_and_overlap(h):
    """Return autocorr array and overlap array for shifts k=0..N-1."""
    autocorr = np.correlate(h, h, mode='full')[N-1:2*N-1]
    overlap = N / 2.0 - autocorr
    return autocorr, overlap

def _project(h):
    """Project h onto [0,1]^N with sum = N/2. Simple iterative projection."""
    h = h.copy()
    for _ in range(50):
        # Clip to [0,1]
        h = np.clip(h, 0.0, 1.0)
        # Adjust to satisfy sum constraint
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-12:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final exact correction
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-15:
        # Distribute among non-saturated elements
        mask = (h > 1e-8) & (h < 1.0 - 1e-8)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            # Renormalize
            diff2 = N / 2.0 - np.sum(h)
            if abs(diff2) > 1e-15:
                h += diff2 / N
                h = np.clip(h, 0.0, 1.0)
    return h

def _objective_and_grad(h, temperature=0.01):
    """Compute smooth max overlap and its gradient w.r.t. h."""
    autocorr, overlap = _autocorr_and_overlap(h)
    val, grad_overlap = _smooth_max(overlap, temperature)
    # overlap[k] = N/2 - autocorr[k]
    # autocorr[k] = sum_i h_i * h_{i+k}
    # d(autocorr[k])/dh_j = h_{j+k} + (if j-k >= 0) h_{j-k} ... 
    # Actually: autocorr[k] = sum_i h_i h_{i+k}
    # d(autocorr[k])/dh_j = h_{j+k} (when j+k < N) + h_{j-k} (when j-k >= 0)
    # So d(overlap[k])/dh_j = -h_{j+k} - h_{j-k} (with boundary handling)
    
    # Build gradient of overlap w.r.t. h
    grad_h = np.zeros(N)
    for k in range(N):
        g = grad_overlap[k]
        if g == 0.0:
            continue
        # d(autocorr[k])/dh_j for j in [0, N-1]:
        # autocorr[k] = sum_{i=0}^{N-1-k} h[i] * h[i+k]
        # d/dh[j] = h[j+k] if j+k < N, plus h[j-k] if j-k >= 0
        # For j in [0, N-1]:
        #   term1: h[j+k] if j+k < N
        #   term2: h[j-k] if j-k >= 0
        # So d(overlap[k])/dh[j] = -(h[j+k] if j+k<N else 0) - (h[j-k] if j-k>=0 else 0)
        
        # Vectorized: for each j, contribution is -h[j+k] (if valid) - h[j-k] (if valid)
        # j+k < N means j < N-k
        # j-k >= 0 means j >= k
        
        # For j in [0, N-k-1]: d/dh[j] includes -h[j+k]
        # For j in [k, N-1]: d/dh[j] includes -h[j-k]
        
        # This is equivalent to: grad_h -= g * (np.roll(h, k) + np.roll(h, -k)) but with boundary handling
        # Actually let's just do it directly for correctness
        
        # d(overlap[k])/dh[j] = -h[j+k] (if j+k<N) - h[j-k] (if j-k>=0)
        # = -(h[j+k] if j < N-k else 0) - (h[j-k] if j >= k else 0)
        
        # For j from 0 to N-1:
        #   if j < N-k: subtract h[j+k]
        #   if j >= k: subtract h[j-k]
        
        # Vectorized approach:
        # term1: for j in [0, N-k-1], h[j+k] = h[k:N]
        # term2: for j in [k, N-1], h[j-k] = h[0:N-k]
        
        if N - k > 0:
            grad_h[:N-k] -= g * h[k:]
        if N - k > 0:
            grad_h[k:] -= g * h[:N-k]
    
    return val, grad_h

def _gradient_descent(h0, max_iters=5000, lr=0.01, temperature=0.01, deadline=None):
    """Run gradient descent to minimize smooth max overlap."""
    h = h0.copy()
    best_h = h.copy()
    best_score = _overlap_max(h)
    
    start = time.time()
    for it in range(max_iters):
        if deadline is not None and time.time() - start > deadline - 10:
            break
        
        val, grad_h = _objective_and_grad(h, temperature)
        
        # Gradient step
        h_new = h - lr * grad_h
        
        # Project to feasible set
        h_new = _project(h_new)
        
        # Check score
        score = _overlap_max(h_new)
        if score < best_score:
            best_score = score
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.95
    
    return best_h, best_score

def _polish(h, max_iters=2000, deadline=None):
    """High-precision polish using smaller steps and tighter projection."""
    h = h.copy()
    best_h = h.copy()
    best_score = _overlap_max(h)
    
    start = time.time()
    lr = 0.001
    temperature = 0.001
    
    for it in range(max_iters):
        if deadline is not None and time.time() - start > deadline - 5:
            break
        
        val, grad_h = _objective_and_grad(h, temperature)
        
        h_new = h - lr * grad_h
        h_new = _project(h_new)
        
        score = _overlap_max(h_new)
        if score < best_score:
            best_score = score
            best_h = h_new.copy()
        
        h = h_new
        
        if it % 200 == 0:
            lr *= 0.9
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial guess: uniform 0.5
    h = np.full(N, 0.5)
    
    # Try multiple random initializations
    best_h = h.copy()
    best_score = _overlap_max(h)
    
    rng = np.random.default_rng(42)
    
    # Run gradient descent from several starting points
    num_starts = 5
    for s in range(num_starts):
        if time.time() > deadline - 20:
            break
        
        if s == 0:
            h0 = np.full(N, 0.5)
        else:
            # Random perturbation around 0.5
            h0 = 0.5 + 0.1 * rng.standard_normal(N)
            h0 = np.clip(h0, 0.0, 1.0)
            # Ensure sum constraint
            h0 = _project(h0)
        
        h_opt, score = _gradient_descent(h0, max_iters=3000, lr=0.01, temperature=0.01, deadline=deadline - start_time)
        
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Polish the best result
    if time.time() < deadline - 10:
        h_polished, score_polished = _polish(best_h, max_iters=2000, deadline=deadline - start_time)
        if score_polished < best_score:
            best_score = score_polished
            best_h = h_polished
    
    # Final projection to ensure exact constraints
    best_h = _project(best_h)
    
    # Verify
    assert len(best_h) == N
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10)
    assert abs(np.sum(best_h) - N/2.0) < 1e-6
    
    # Clean up tiny numerical errors
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N/2.0 - np.sum(best_h)
    if abs(diff) > 1e-10:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h

# EVOLVE-BLOCK-START
# (The above functions are the implementation)
# EVOLVE-BLOCK-END
