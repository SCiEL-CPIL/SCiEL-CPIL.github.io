import numpy as np
import time

N = 100
HALF = N / 2.0
DEADLINE = 135.0  # Safety margin before 150s kill

def _value(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    h_clipped = np.clip(h, 0.0, 1.0)
    corr = np.correlate(h_clipped, 1.0 - h_clipped, mode="full")
    return float(np.max(corr)) / N * 2.0

def _max_corr(h):
    h = np.asarray(h, dtype=np.float64)
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr))

def _fix_sum(h):
    """Adjust h to satisfy sum == N/2, keeping in [0,1]."""
    h = h.copy()
    for _ in range(50):
        diff = HALF - np.sum(h)
        if abs(diff) < 1e-12:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _corr_diff(h, i, j):
    """Compute the change in correlation vector when swapping h[i] and h[j].
    Returns the full correlation difference vector."""
    # h[i] becomes h[j], h[j] becomes h[i]
    # delta_i = h[j] - h[i], delta_j = h[i] - h[j] = -delta_i
    delta_i = h[j] - h[i]
    delta_j = -delta_i
    
    # corr[k] = sum_l h[l] * (1-h)[l+k-N] for k in 0..2N-2
    # When we change h[i] by delta_i, the correlation changes by:
    #   delta_i * (1-h)[i+k-N] for the h part
    #   -h[i] * delta_i for the (1-h) part... wait, let me think more carefully
    
    # corr[k] = sum_l h[l] * (1-h[l+k-N])
    # If h[i] changes by delta_i:
    #   term where l=i: delta_i * (1-h[i+k-N])
    #   term where l+k-N=i i.e. l=i-k+N: h[i-k+N] * (-delta_i)
    # So delta_corr[k] = delta_i * (1-h[i+k-N]) - delta_i * h[i-k+N]
    #                    = delta_i * (1 - h[i+k-N] - h[i-k+N])
    
    # But we need to handle boundaries: indices must be in [0, N-1]
    # Let's just compute the full correlation difference directly for speed
    
    # Actually for binary case, delta_i is either 0, 1, or -1
    # Let's just recompute for the specific case
    
    # For efficiency, let's precompute what changes
    # The correlation at lag k involves pairs (l, l+k-N)
    # Changing h[i] affects:
    #   As h part: l=i, so corr[k] += delta_i * (1-h[i+k-N]) for valid i+k-N
    #   As (1-h) part: l+k-N=i => l=i-k+N, so corr[k] -= delta_i * h[i-k+N] for valid i-k+N
    
    # This is complex to vectorize perfectly. Let's just use the fact that
    # for binary swaps, we can compute the new max more efficiently.
    pass

def _binary_local_search(h_bin, time_limit=30.0):
    """Simulated annealing on binary vectors with exactly 50 ones."""
    h = h_bin.astype(np.float64)
    current_val = _max_corr(h)
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    
    if len(ones) == 0 or len(zeros) == 0:
        return h
    
    best_h = h.copy()
    best_val = current_val
    t_start = time.time()
    T = 0.5  # Initial temperature
    cooling = 0.995
    
    rng = np.random.default_rng(42)
    
    while time.time() - t_start < time_limit:
        # Pick random one and zero to swap
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        
        h[i] = 0.0
        h[j] = 1.0
        new_val = _max_corr(h)
        
        delta = new_val - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            # Accept
            current_val = new_val
            # Update index sets
            ones = np.where(h > 0.5)[0]
            zeros = np.where(h < 0.5)[0]
            
            if new_val < best_val:
                best_h = h.copy()
                best_val = new_val
        else:
            # Reject
            h[i] = 1.0
            h[j] = 0.0
        
        T *= cooling
    
    return best_h

def _binary_hill_climb(h_bin, max_iters=200):
    """Greedy hill climbing on binary vectors."""
    h = h_bin.astype(np.float64)
    current_val = _max_corr(h)
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    
    if len(ones) == 0 or len(zeros) == 0:
        return h
    
    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        best_gain = 1e-12
        best_i = -1
        best_j = -1
        for i in ones:
            for j in zeros:
                h[i] = 0.0
                h[j] = 1.0
                new_val = _max_corr(h)
                gain = current_val - new_val
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
                h[i] = 1.0
                h[j] = 0.0
        if best_i >= 0:
            h[best_i] = 0.0
            h[best_j] = 1.0
            current_val = current_val - best_gain
            ones = np.where(h > 0.5)[0]
            zeros = np.where(h < 0.5)[0]
            improved = True
    return h

def _generate_binary_starts():
    """Generate diverse binary starting points with exactly 50 ones."""
    rng = np.random.default_rng(42)
    starts = []
    
    # Random binary with 50 ones
    for _ in range(50):
        idx = rng.choice(N, size=50, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        starts.append(h)
    
    # Periodic patterns
    for period in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 50]:
        for offset in range(min(period, 10)):
            h = np.zeros(N)
            for i in range(N):
                if (i + offset) % period < period // 2:
                    h[i] = 1.0
            # Adjust to have exactly 50 ones
            if np.sum(h) != 50:
                # Sort by value and adjust
                if np.sum(h) > 50:
                    # Turn off some ones
                    ones = np.where(h == 1)[0]
                    to_off = np.sum(h) - 50
                    for idx in ones[:to_off]:
                        h[idx] = 0.0
                else:
                    # Turn on some zeros
                    zeros = np.where(h == 0)[0]
                    to_on = 50 - np.sum(h)
                    for idx in zeros[:to_on]:
                        h[idx] = 1.0
            starts.append(h)
    
    # Block patterns
    for block_size in [1, 2, 3, 4, 5, 10]:
        for offset in range(block_size):
            h = np.zeros(N)
            for i in range(N):
                if (i + offset) % (2 * block_size) < block_size:
                    h[i] = 1.0
            if np.sum(h) != 50:
                if np.sum(h) > 50:
                    ones = np.where(h == 1)[0]
                    to_off = int(np.sum(h) - 50)
                    for idx in ones[:to_off]:
                        h[idx] = 0.0
                else:
                    zeros = np.where(h == 0)[0]
                    to_on = int(50 - np.sum(h))
                    for idx in zeros[:to_on]:
                        h[idx] = 1.0
            starts.append(h)
    
    return starts

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE
    
    # Start with seed
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    rng = np.random.default_rng(42)
    
    # We will run multiple SA instances to avoid getting stuck in local minima.
    # The hill climbing in the original code was too slow (O(N^2) per step with Python loops).
    # SA is O(1) per step (with O(N^2) cost for correlation, but in C).
    
    # Allocate time to SA runs.
    # Let's do 10 runs of ~10s each, if time allows.
    max_runs = 10
    time_per_run = 10.0
    
    for run in range(max_runs):
        # Check if we have enough time for this run and a safety buffer
        if time.time() > deadline - (time_per_run + 5.0):
            break
            
        # Generate a random binary start with 50 ones
        idx = rng.choice(N, size=50, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        
        # Run Simulated Annealing
        h_opt = _binary_local_search(h, time_limit=time_per_run)
        val = _value(h_opt)
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
            
    # Ensure valid output
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)
    
    return best_h.tolist()
