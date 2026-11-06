import time
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    powerful = set()
    max_b = int(limit ** (1.0 / 3.0)) + 2
    is_sq_free = [True] * (max_b + 1)
    if max_b >= 0:
        is_sq_free[0] = False
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        for j in range(sq, max_b + 1, sq):
            is_sq_free[j] = False
    
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b * b * b
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_N = 2
    best_count = 1
    best_pairs = [[1, 1]]
    
    import numpy as np
    
    # Try increasing limits
    for limit in [10**7, 10**8, 10**9, 10**10]:
        if time.time() > deadline - 30:
            break
        
        powerful = _generate_powerful(limit)
        M = len(powerful)
        
        if time.time() > deadline - 25:
            break
        
        if M < 2:
            continue
        
        pw = np.array(powerful, dtype=np.int64)
        
        # Compute all pair sums a + b where a <= b
        # Use a chunked approach to manage memory
        all_sums = []
        
        for i in range(M):
            if time.time() > deadline - 20:
                break
            a = pw[i]
            # b must be >= a, indices from i to M-1
            if i < M:
                sums = a + pw[i:]
                sums = sums[sums <= 10**18]
                if len(sums) > 0:
                    all_sums.append(sums)
        
        if not all_sums:
            continue
        
        all_sums = np.concatenate(all_sums)
        
        # Count using bincount
        min_sum = int(all_sums.min())
        max_sum = int(all_sums.max())
        range_size = max_sum - min_sum + 1
        
        if range_size <= 2 * 10**8:
            counts = np.bincount(all_sums - min_sum, minlength=range_size)
            best_offset = int(np.argmax(counts))
            count = int(counts[best_offset])
            N = min_sum + best_offset
        else:
            unique_sums, counts = np.unique(all_sums, return_counts=True)
            best_idx = int(np.argmax(counts))
            count = int(counts[best_idx])
            N = int(unique_sums[best_idx])
        
        if count > best_count:
            best_count = count
            best_N = N
            # Reconstruct pairs
            pw_set = set(powerful)
            pairs = []
            for a in powerful:
                b = N - a
                if a <= b and b in pw_set:
                    pairs.append([a, b])
            best_pairs = pairs
    
    # Final verification
    if not best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    max_val = max(max(a, b) for a, b in best_pairs)
    pw_set = set(_generate_powerful(max_val + 1000))
    valid_pairs = []
    for a, b in best_pairs:
        if a <= b and a + b == best_N and a in pw_set and b in pw_set:
            valid_pairs.append([a, b])
    
    valid_pairs = sorted(set(tuple(p) for p in valid_pairs))
    valid_pairs = [list(p) for p in valid_pairs]
    
    if not valid_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': valid_pairs}
