import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def _overlap_score(h):
    """Compute the maximum overlap of h with shifted 1-h over linear shifts."""
    one_minus_h = 1.0 - h
    # For each shift k (0 to N-2), compute sum_{i=0}^{N-1-k} h[i] * (1-h[i+k])
    # Using np.correlate in 'valid' mode:
    # np.correlate(h, one_minus_h, 'valid') gives:
    #   for k=0: sum h[i]*one_minus_h[i]
    #   for k=1: sum h[i]*one_minus_h[i+1]
    #   ...
    overlaps = np.correlate(h, one_minus_h, mode='valid')
    return np.max(overlaps)

def _construct_step():
    """
    Construct a step function: first N/2 values are 0, last N/2 values are 1.
    Sum = N/2 = 50.
    """
    h = np.zeros(N)
    h[N//2:] = 1.0
    return h

def _construct_alternating():
    """
    Construct an alternating 0-1 pattern.
    h[i] = 1 if i even, 0 if i odd (or vice versa).
    Sum = N/2 = 50.
    """
    h = np.zeros(N)
    h[::2] = 1.0
    return h

def _construct_sinusoid(f, A):
    """
    Construct h[i] = 0.5 + A * sin(2*pi*f*i/N), clipped to [0,1],
    then adjusted to have sum exactly N/2.
    """
    i = np.arange(N)
    h = 0.5 + A * np.sin(2 * np.pi * f * i / N)
    h = np.clip(h, 0, 1)
    
    # Adjust to have sum exactly N/2
    current_sum = np.sum(h)
    if current_sum > 0:
        h = h * (N / 2.0) / current_sum
    else:
        # If all zeros, set to 0.5
        h[:] = 0.5
    
    h = np.clip(h, 0, 1)
    
    # Final correction for exact sum
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        # Add diff/N to each element
        h += diff / N
        h = np.clip(h, 0, 1)
        
        # If still not exact due to clipping, adjust one element
        diff2 = N / 2.0 - np.sum(h)
        if abs(diff2) > 1e-12:
            if diff2 > 0:
                # Need to add, find element with most room
                room = 1.0 - h
                idx = np.argmax(room)
                h[idx] += min(diff2, room[idx])
            else:
                # Need to subtract, find element with most value
                idx = np.argmax(h)
                h[idx] += max(diff2, -h[idx])
    
    return h

def _construct():
    """
    Try multiple construction strategies and return the best one.
    """
    candidates = []
    
    # 1. Step function
    candidates.append(_construct_step())
    
    # 2. Alternating
    candidates.append(_construct_alternating())
    
    # 3. Sinusoids with various frequencies and amplitudes
    for f in range(1, N // 2 + 1):
        for A in [0.1, 0.2, 0.3, 0.4, 0.45, 0.49]:
            h = _construct_sinusoid(f, A)
            candidates.append(h)
    
    # Find the best candidate
    best_h = None
    best_score = float('inf')
    
    for h in candidates:
        score = _overlap_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    return best_h, best_score

def construct():
    """
    Returns a sequence of step heights in [0, 1] whose sum is exactly len/2.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Quick constructions
    h, score = _construct()
    
    # Check time
    if time.time() - start_time > deadline:
        return h
    
    # Try to improve with local search
    # Use swap-based local search to maintain sum = N/2
    current = h.copy()
    current_score = score
    
    # Local search: try swapping pairs of elements
    # We want to reduce the maximum overlap
    # This is a heuristic improvement step
    
    n_iter = 0
    max_iter = 100000
    
    while time.time() - start_time < deadline and n_iter < max_iter:
        n_iter += 1
        
        # Pick two random indices to swap
        i, j = np.random.choice(N, 2, replace=False)
        
        if abs(current[i] - current[j]) < 1e-12:
            continue
        
        # Try the swap
        old_i, old_j = current[i], current[j]
        current[i], current[j] = old_j, old_i
        
        new_score = _overlap_score(current)
        
        if new_score < current_score:
            current_score = new_score
        else:
            # Revert
            current[i], current[j] = old_i, old_j
    
    return current

# EVOLVE-BLOCK-END
