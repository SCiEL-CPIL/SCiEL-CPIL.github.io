import time
import numpy as np

def _sieve(limit):
    """Return numpy array of all primes <= limit using a standard sieve."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.where(sieve)[0]

def construct():
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    t_start = time.time()

    primes = _sieve(10000)
    n = len(primes)
    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]

    p_list = primes.tolist()
    
    # dp[i] will be a list of tuples (length, last_gap, prev_index)
    # representing Pareto-optimal states ending at prime index i.
    # A state (l1, g1) dominates (l2, g2) if l1 >= l2 and g1 <= g2.
    
    dp = [None] * n
    dp[0] = [(1, 0, -1)]
    
    best_len = 1
    best_end = 0
    best_state = (1, 0, -1)
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        candidates = []
        # Base case: sequence of length 1 ending at i
        candidates.append((1, 0, -1))
        
        pi = p_list[i]
        
        for j in range(i):
            gap_ij = pi - p_list[j]
            # For each state in dp[j], check if we can extend
            # State is (l, g, prev). New state is (l+1, gap_ij, j).
            # Condition: gap_ij >= g
            for l, g, prev in dp[j]:
                if gap_ij >= g:
                    candidates.append((l + 1, gap_ij, j))
        
        # Pareto Optimize candidates
        # 1. For each length, find the minimum gap
        len_to_mingap = {}
        len_to_prev = {}
        for l, g, prev in candidates:
            if l not in len_to_mingap or g < len_to_mingap[l]:
                len_to_mingap[l] = g
                len_to_prev[l] = prev
        
        # 2. Filter dominated lengths
        # Sort lengths descending
        lengths = sorted(len_to_mingap.keys(), reverse=True)
        pareto = []
        min_gap_so_far = float('inf')
        for l in lengths:
            g = len_to_mingap[l]
            # If this gap is smaller than the best gap seen for any larger length,
            # it's not dominated by any larger length state.
            # Note: If g == min_gap_so_far, it's dominated (same gap, smaller length).
            # If g < min_gap_so_far, it's a new Pareto point.
            if g < min_gap_so_far:
                pareto.append((l, g, len_to_prev[l]))
                min_gap_so_far = g
        
        dp[i] = pareto
        
        # Update global best
        if pareto:
            # pareto is sorted by length descending (because we iterated lengths desc)
            # The first element has the max length.
            # Among max length states, we kept the one with min gap.
            current_best = pareto[0]
            if current_best[0] > best_len:
                best_len = current_best[0]
                best_end = i
                best_state = current_best
    
    # Reconstruct
    if best_end == 0:
        return [p_list[0]]
        
    seq_indices = [best_end]
    cur = best_end
    target_len = best_len
    
    # We need to trace back.
    # dp[cur] contains states. We know the best state ending at cur is the one with length target_len.
    # However, there might be multiple states with length target_len?
    # No, in our Pareto optimization, for a given length, we only kept the one with min gap.
    # So there is at most one state in dp[cur] with length == target_len.
    
    while target_len > 1:
        found = False
        for l, g, prev in dp[cur]:
            if l == target_len:
                seq_indices.append(prev)
                cur = prev
                target_len -= 1
                found = True
                break
        if not found:
            # Should not happen if logic is correct
            break
            
    seq_indices.reverse()
    result = [p_list[i] for i in seq_indices]
    
    # Final verification
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        if not all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1)):
             # Fallback to a valid small sequence if reconstruction failed
             return [p_list[0], p_list[1]] if n > 1 else [p_list[0]]
             
    return result
