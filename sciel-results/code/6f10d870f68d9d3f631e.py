"""Additive bases with few representations - simulated annealing with vectorized evaluation."""
import time
import numpy as np
from collections import Counter

# Deadline guard: 90s wall-clock, keep 10s safety margin
_DEADLINE = 80.0

def _count_representations(A_set):
    """Count how many unordered pairs (a,b) with a<=b from A sum to each value in [128,256]."""
    A = np.array(sorted(A_set), dtype=np.int32)
    n = len(A)
    # Build all pair sums (unordered, with repetition)
    # For efficiency, use broadcasting on a subset or iterate
    # Number of pairs = n*(n+1)/2 which for n~65 is ~2145, manageable
    sums = []
    for i in range(n):
        for j in range(i, n):
            s = A[i] + A[j]
            if 128 <= s <= 256:
                sums.append(s)
    if not sums:
        return np.zeros(129, dtype=np.int32)
    arr = np.array(sums, dtype=np.int32)
    # Count occurrences of each sum in [128,256]
    counts = np.zeros(129, dtype=np.int32)
    for s in arr:
        counts[s - 128] += 1
    return counts

def _evaluate(A_set):
    """Return (peak_multiplicity, is_valid). Lower peak is better. Valid means all 128..256 covered."""
    counts = _count_representations(A_set)
    # Check coverage: every index 0..128 (sums 128..256) must have count >= 1
    if np.any(counts < 1):
        return (float('inf'), False)
    return (int(counts.max()), True)

def _fast_count(A_sorted):
    """Faster count using numpy where possible. A_sorted is a sorted numpy array."""
    n = len(A_sorted)
    counts = np.zeros(129, dtype=np.int32)
    for i in range(n):
        # For each a_i, find a_j >= a_i such that 128 <= a_i + a_j <= 256
        # a_j in [128 - a_i, 256 - a_i] and a_j >= a_i
        lo = max(A_sorted[i], 128 - A_sorted[i])
        hi = min(256 - A_sorted[i], 128)  # a_j <= 128 since A subset of [0,128]
        if lo > hi:
            continue
        # Binary search in A_sorted for elements in [lo, hi]
        idx_lo = np.searchsorted(A_sorted, lo, side='left')
        idx_hi = np.searchsorted(A_sorted, hi, side='right')
        # But we need a_j >= a_i, which is already ensured by lo >= A_sorted[i]
        # Also need to handle the case where a_j == a_i (i.e., 2*a_i)
        # The pairs are (a_i, a_j) for j >= i in the sorted list
        # Actually let's just count: for each i, number of j >= i with A[j] in [128-A[i], 256-A[i]]
        # Since A is sorted, j >= i means A[j] >= A[i]
        # So we need A[j] in [max(A[i], 128-A[i]), min(256-A[i], 128)]
        # and j >= i
        # The number of such j is idx_hi - max(idx_lo, i) if idx_hi > max(idx_lo, i)
        start = max(idx_lo, i)
        end = idx_hi
        if end > start:
            # Each such j gives one pair (A[i], A[j])
            # But wait, this double counts? No, because we fix i and count j >= i
            # Each unordered pair {a,b} with a<=b is counted exactly once: when i is the index of a
            # and j is the index of b (j >= i)
            # So the count for sum s = A[i] + A[j] gets +1
            # We need to add 1 to counts for each such pair's sum
            # This is O(n^2) in worst case but with numpy we can vectorize per-i
            js = np.arange(start, end)
            pair_sums = A_sorted[i] + A_sorted[js]
            # Filter to [128, 256]
            valid = (pair_sums >= 128) & (pair_sums <= 256)
            if np.any(valid):
                valid_sums = pair_sums[valid] - 128
                np.add.at(counts, valid_sums, 1)
    return counts

def construct():
    start_time = time.time()
    
    # Initial solution: the seed [64, 128]
    A = list(range(64, 129))
    A_arr = np.array(A, dtype=np.int32)
    current_peak, current_valid = _evaluate(A)
    
    # Try to find a better initial solution quickly
    # Let's also try a smaller set
    # For coverage of [128,256], we need pairs from A
    # A = [0, 128] gives sums {0, 128, 256} - not enough
    # We need A such that A+A covers [128, 256]
    
    best_A = A[:]
    best_peak = current_peak
    
    # Simulated annealing
    rng = np.random.default_rng(42)
    
    # Temperature schedule
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    # Precompute: for fast evaluation, we'll use the _fast_count approach
    # But even that is O(n^2). Let's try a different approach.
    
    # Alternative: represent A as a boolean array of size 129
    # Then pair sums can be computed via convolution
    # A_bool = np.zeros(129, dtype=np.int32)
    # A_bool[A] = 1
    # conv = np.convolve(A_bool, A_bool, mode='full')
    # conv[k] = number of ordered pairs (a,b) in A with a+b=k
    # For unordered pairs with repetition: 
    #   count_unordered(s) = (conv[s] + (1 if s is even and s//2 in A else 0)) // 2
    # Wait: conv[s] = sum over a in A of 1 if (s-a) in A = number of ordered pairs
    # Unordered pairs with a<=b: 
    #   For s even: pairs with a=b=s/2 contribute 1 if s/2 in A
    #   Pairs with a<b: each unordered pair {a,b} with a<b is counted twice in conv[s]
    #   So unordered_count(s) = (conv[s] - (1 if s even and s//2 in A else 0)) // 2 + (1 if s even and s//2 in A else 0)
    #   = (conv[s] + (1 if s even and s//2 in A else 0)) // 2
    
    def evaluate_with_conv(A_set):
        A_bool = np.zeros(129, dtype=np.int32)
        for x in A_set:
            A_bool[x] = 1
        conv = np.convolve(A_bool, A_bool, mode='full')
        # conv[s] for s in 0..256
        # We care about s in 128..256
        counts = np.zeros(129, dtype=np.int32)
        for s in range(128, 257):
            if s < len(conv):
                ordered = conv[s]
                if s % 2 == 0 and s // 2 <= 128:
                    diag = 1 if A_bool[s // 2] else 0
                else:
                    diag = 0
                counts[s - 128] = (ordered + diag) // 2
        if np.any(counts < 1):
            return (float('inf'), False)
        return (int(counts.max()), True)
    
    # Test the convolution approach
    peak_conv, valid_conv = evaluate_with_conv(A)
    # Should match
    # print(f"Direct: {current_peak}, Conv: {peak_conv}")
    
    # Use convolution-based evaluation for speed
    def eval_conv(A_set):
        A_bool = np.zeros(129, dtype=np.int32)
        for x in A_set:
            A_bool[x] = 1
        conv = np.convolve(A_bool, A_bool, mode='full')
        counts = np.zeros(129, dtype=np.int32)
        for s in range(128, min(257, len(conv))):
            ordered = conv[s]
            if s % 2 == 0 and s // 2 <= 128:
                diag = 1 if A_bool[s // 2] else 0
            else:
                diag = 0
            counts[s - 128] = (ordered + diag) // 2
        if np.any(counts < 1):
            return (float('inf'), False)
        return (int(counts.max()), True)
    
    # Verify
    p1, v1 = eval_conv(A)
    # print(f"Eval conv: {p1}, valid: {v1}")
    
    best_peak = p1
    best_A = A[:]
    
    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    # We'll represent A as a sorted list
    current_A = A[:]
    current_peak = p1
    
    # Try some random initial solutions to get a better starting point
    for init_try in range(20):
        # Random subset of [0,128] of size ~65
        size = rng.integers(50, 75)
        cand = sorted(rng.choice(129, size=size, replace=False).tolist())
        p, v = eval_conv(cand)
        if v and p < best_peak:
            best_peak = p
            best_A = cand[:]
            current_A = cand[:]
            current_peak = p
        elif v and current_peak == float('inf') and p < current_peak:
            current_A = cand[:]
            current_peak = p
    
    # If we still have no valid solution, use the seed
    if current_peak == float('inf'):
        current_A = list(range(64, 129))
        current_peak, _ = eval_conv(current_A)
        if current_peak < best_peak:
            best_peak = current_peak
            best_A = current_A[:]
    
    # Simulated annealing loop
    step = 0
    while time.time() - start_time < _DEADLINE - 10:
        step += 1
        
        # Generate a move: add, remove, or swap an element
        move_type = rng.integers(0, 3)
        new_A = current_A[:]
        
        if move_type == 0:  # Add an element
            # Add a random element not in A
            available = [x for x in range(129) if x not in set(new_A)]
            if available:
                x = rng.choice(available)
                new_A.append(x)
                new_A.sort()
        
        elif move_type == 1:  # Remove an element
            if len(new_A) > 10:  # Don't make it too small
                idx = rng.integers(0, len(new_A))
                new_A.pop(idx)
        
        else:  # Swap: remove one, add another
            if len(new_A) > 10:
                idx = rng.integers(0, len(new_A))
                removed = new_A.pop(idx)
                available = [x for x in range(129) if x not in set(new_A)]
                if available:
                    x = rng.choice(available)
                    new_A.append(x)
                    new_A.sort()
        
        new_peak, new_valid = eval_conv(new_A)
        
        if new_valid:
            if new_peak < current_peak:
                # Accept
                current_A = new_A
                current_peak = new_peak
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_A = new_A[:]
            else:
                # Metropolis acceptance
                delta = new_peak - current_peak
                if delta == 0:
                    # Accept with some probability
                    if rng.random() < 0.5:
                        current_A = new_A
                else:
                    # Accept with probability exp(-delta/T)
                    if T > T_min:
                        accept_prob = np.exp(-delta / T)
                        if rng.random() < accept_prob:
                            current_A = new_A
                            current_peak = new_peak
        else:
            # Invalid: only accept if we're stuck and T is high
            if T > 1.0 and rng.random() < 0.1:
                current_A = new_A
                current_peak = float('inf')
        
        # Cool down
        T *= cooling
        
        # Occasionally restart with a slightly perturbed best solution
        if step % 500 == 0 and best_peak < float('inf'):
            # Perturb best solution
            perturbed = best_A[:]
            for _ in range(rng.integers(1, 4)):
                if len(perturbed) > 10:
                    idx = rng.integers(0, len(perturbed))
                    perturbed.pop(idx)
                    available = [x for x in range(129) if x not in set(perturbed)]
                    if available:
                        perturbed.append(rng.choice(available))
                        perturbed.sort()
            pp, pv = eval_conv(perturbed)
            if pv:
                current_A = perturbed
                current_peak = pp
            else:
                current_A = best_A[:]
                current_peak = best_peak
    
    # Return the best solution found
    if best_peak == float('inf'):
        # Fallback to seed
        return list(range(64, 129))
    
    return best_A
