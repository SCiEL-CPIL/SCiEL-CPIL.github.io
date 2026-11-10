import time
import random
import numpy as np

def _compute_counts(A_list):
    """Compute sum multiplicities for A_list over [128, 256]."""
    counts = np.zeros(129, dtype=np.int32)
    n = len(A_list)
    for i in range(n):
        a = A_list[i]
        for j in range(i, n):
            s = a + A_list[j]
            if 128 <= s <= 256:
                counts[s - 128] += 1
    return counts

def _gerchberg_saxton(x, target_autocorr, num_iters=500, step_size=0.5):
    """
    Apply Gerchberg-Saxton style iterations to binary vector x.
    x: binary vector of length 129
    target_autocorr: target autocorrelation values for lags 128..256
    Returns: improved binary vector
    """
    x = x.astype(np.float64).copy()
    N = len(x)
    
    # Precompute FFT of target autocorrelation
    # Autocorrelation of x: R[k] = sum_i x[i] * x[i+k]
    # We want R[k] for k in 128..256 to be close to target
    # In frequency domain, autocorrelation magnitude relates to |FFT(x)|^2
    
    # Target power spectrum: if we know R[k] for all k, we can get |F[k]|^2
    # But we only know R[k] for k in 128..256. 
    # For a binary vector, the full autocorrelation determines the power spectrum.
    # However, we don't have the full autocorrelation.
    
    # Alternative: use the constraint that R[k] = target[k] for k in 128..256
    # and try to satisfy this via iterative projection.
    
    # Let's use a simpler approach: iterative refinement
    # 1. Compute current autocorrelation
    # 2. Adjust x to reduce discrepancy
    
    for it in range(num_iters):
        # Compute current autocorrelation for lags 128..256
        curr_autocorr = np.zeros(129, dtype=np.float64)
        for lag in range(128, 257):
            c = 0
            for i in range(0, 129 - lag):
                c += x[i] * x[i + lag]
            curr_autocorr[lag - 128] = c
        
        # Compute error
        error = curr_autocorr - target_autocorr
        total_err = np.sum(error ** 2)
        
        if total_err < 1e-6:
            break
        
        # Try to adjust x
        # For each position i, consider flipping x[i]
        # Compute how much this would reduce the error
        
        # This is expensive, so let's use a gradient-like approach
        # Actually, for binary vectors, let's just try random flips and accept if better
        
        # Pick a random position to flip
        pos = random.randint(0, 128)
        old_val = x[pos]
        x[pos] = 1.0 - x[pos]
        
        # Recompute autocorrelation
        new_autocorr = np.zeros(129, dtype=np.float64)
        for lag in range(128, 257):
            c = 0
            for i in range(0, 129 - lag):
                c += x[i] * x[i + lag]
            new_autocorr[lag - 128] = c
        
        new_err = np.sum((new_autocorr - target_autocorr) ** 2)
        
        if new_err > total_err:
            x[pos] = old_val
    
    # Threshold to binary
    x_binary = (x > 0.5).astype(np.int32)
    return x_binary

def _greedy_cover():
    """Greedily construct a set that covers [128, 256]."""
    A = set()
    counts = np.zeros(129, dtype=np.int32)
    
    # Start with some base elements
    for i in range(0, 129, 4):
        A.add(i)
    
    # Compute initial counts
    A_list = sorted(A)
    counts = _compute_counts(A_list)
    
    # Iteratively add elements to cover uncovered sums
    for _ in range(50):
        if (counts > 0).all():
            break
        
        # Find uncovered sums
        uncovered = [i for i in range(129) if counts[i] == 0]
        
        # For each candidate, compute how many uncovered sums it would cover
        best_cand = -1
        best_gain = 0
        
        for cand in range(129):
            if cand in A:
                continue
            
            gain = 0
            for idx in uncovered:
                target_sum = idx + 128
                partner = target_sum - cand
                if 0 <= partner <= 128 and partner in A:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        
        if best_cand == -1:
            # Add random
            A.add(random.randint(0, 128))
        else:
            A.add(best_cand)
        
        # Recompute counts
        A_list = sorted(A)
        counts = _compute_counts(A_list)
    
    return A

def _local_search(A_init, time_limit=50.0):
    """Local search to minimize peak multiplicity."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    A = set(A_init)
    in_A = np.zeros(129, dtype=bool)
    for x in A:
        in_A[x] = True
    
    A_list = [i for i in range(129) if in_A[i]]
    counts = _compute_counts(A_list)
    
    peak = int(counts.max())
    best_peak = peak
    best_in_A = in_A.copy()
    best_counts = counts.copy()
    
    # Track histogram for fast peak updates
    max_count = len(A_list)
    hist = np.zeros(max_count + 2, dtype=np.int32)
    for c in counts:
        hist[c] += 1
    
    current_peak = peak
    
    def get_peak():
        return current_peak
    
    def update_count(idx, delta):
        nonlocal current_peak
        old_c = counts[idx]
        new_c = old_c + delta
        hist[old_c] -= 1
        hist[new_c] += 1
        counts[idx] = new_c
        
        if new_c > current_peak:
            current_peak = new_c
        elif old_c == current_peak and hist[old_c] == 0:
            while current_peak > 0 and hist[current_peak] == 0:
                current_peak -= 1
    
    def swap(x, y):
        """Remove x, add y."""
        nonlocal current_peak
        
        # Remove x
        in_A[x] = False
        for z in range(129):
            if in_A[z]:
                s = x + z
                if 128 <= s <= 256:
                    update_count(s - 128, -1)
        
        # Add y
        in_A[y] = True
        for z in range(129):
            if in_A[z]:
                s = y + z
                if 128 <= s <= 256:
                    update_count(s - 128, 1)
    
    # Main loop
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
        
        # Pick x to remove (prefer high-multiplicity contributions)
        # Simple: random
        x = random.randint(0, 128)
        while not in_A[x]:
            x = random.randint(0, 128)
        
        # Pick y to add
        y = random.randint(0, 128)
        while in_A[y]:
            y = random.randint(0, 128)
        
        if x == y:
            continue
        
        # Save state
        old_in_A = in_A.copy()
        old_counts = counts.copy()
        old_hist = hist.copy()
        old_peak = current_peak
        
        # Perform swap
        swap(x, y)
        
        new_peak = current_peak
        new_uncov = int(np.sum(counts == 0))
        
        # Cost: peak + large penalty for uncovered
        old_uncov = int(np.sum(old_counts == 0))
        old_cost = old_peak + 1000 * old_uncov
        new_cost = new_peak + 1000 * new_uncov
        
        delta = new_cost - old_cost
        
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    accept = True
        
        if accept:
            if new_cost < old_cost:
                if new_peak < best_peak or (new_peak == best_peak and new_uncov == 0):
                    best_peak = new_peak
                    best_in_A = in_A.copy()
                    best_counts = counts.copy()
        else:
            in_A[:] = old_in_A
            counts[:] = old_counts
            hist[:] = old_hist
            current_peak = old_peak
        
        T *= cooling
    
    return [i for i in range(129) if best_in_A[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe
    
    # 1. Get initial valid set
    A_init = _greedy_cover()
    
    # 2. Local search
    A_result = _local_search(A_init, time_limit=80.0)
    
    # 3. Verify and return
    if len(A_result) == 0:
        return [0]
    
    return sorted(A_result)
