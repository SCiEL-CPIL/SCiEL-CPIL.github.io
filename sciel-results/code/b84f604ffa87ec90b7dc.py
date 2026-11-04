import numpy as np
import time

N = 400

def _compute_acf(h):
    """Compute the autocorrelation (autoconvolution) of h."""
    # Use FFT for initial computation
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[0] is the DC component (sum of squares = N)
    # We are interested in acf[1:N]
    return acf[:N]

def _update_acf(acf, h, i):
    """
    Update the autocorrelation array after flipping h[i].
    h[i] changes sign.
    Returns the new acf array (modified in place or new array).
    """
    new_acf = acf.copy()
    old_val = h[i]
    
    for k in range(1, N):
        delta = 0.0
        # Contribution from j=i
        if i <= N - 1 - k:
            delta += old_val * h[i + k]
        # Contribution from j=i-k
        if i >= k:
            delta += h[i - k] * old_val
        
        # The new term will be -delta (because h[i] flips sign)
        # So the change is -2 * delta
        new_acf[k] -= 2.0 * delta
        
    return new_acf

def _local_search(h, acf, deadline):
    """
    Perform local search by flipping individual bits.
    Uses a greedy approach: repeatedly find the best single flip.
    Returns the best h and acf found, and the best score.
    """
    current_score = np.max(np.abs(acf[1:N]))
    
    # To avoid getting stuck in a local minimum, we can use a simulated annealing
    # style approach or just a simple greedy one. Given the time constraint,
    # a simple greedy with occasional restarts is often effective.
    # However, the current program's local search is already greedy.
    # Let's try to make it more efficient or robust.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0.0
        best_idx = -1
        
        # Evaluate all possible flips
        # Vectorization is tricky because the update depends on the specific index.
        # But we can pre-compute some terms?
        # For a flip at i, the change in acf[k] is -2 * (h[i]h[i+k] + h[i-k]h[i])
        # = -2 * h[i] * (h[i+k] + h[i-k])
        # So the new acf[k] = acf[k] - 2 * h[i] * (h[i+k] + h[i-k])
        # We want to minimize max(|new_acf[1:N]|)
        
        # A faster heuristic: only try flips that seem promising.
        # Or, just do the full search but optimize the inner loop.
        
        for i in range(N):
            # Fast check: estimate the gain without full update?
            # Hard to do for max norm.
            
            new_acf = _update_acf(acf, h, i)
            new_score = np.max(np.abs(new_acf[1:N]))
            gain = current_score - new_score
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx != -1 and best_gain > 1e-9:
            # Apply flip
            h[best_idx] *= -1
            acf = _update_acf(acf, h, best_idx)
            current_score -= best_gain
            improved = True
            
    return h, acf, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Run multiple restarts if time permits
    rng = np.random.RandomState(42)
    
    # Phase 1: Multiple random restarts with local search
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random signs
        h = rng.choice([-1, 1], size=N)
        
        # Compute initial ACF
        acf = _compute_acf(h)
        
        # Local search
        h, acf, current_score = _local_search(h, acf, deadline - 5)
        
        # Update best
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        # If we found a very good score, we might stop early?
        # Theoretical minimum is around sqrt(N) = 20.
        if best_score < 22:
            break

    # Phase 2: If time remains, try a different initialization strategy
    # e.g., alternating blocks or random blocks
    if time.time() < deadline - 10 and best_score > 25:
        block_sizes = [2, 4, 8, 16]
        for bs in block_sizes:
            if time.time() > deadline - 5:
                break
            
            h = np.zeros(N)
            for i in range(0, N, bs):
                sign = rng.choice([-1, 1])
                h[i:i+bs] = sign
            
            acf = _compute_acf(h)
            h, acf, current_score = _local_search(h, acf, deadline - 5)
            
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
                
            if best_score < 22:
                break

    if best_h is None:
        # Fallback to ones if something went wrong
        return np.ones(N)
        
    return best_h
