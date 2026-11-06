"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def _is_square_free(n):
    if n == 1:
        return True
    if n % 4 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % (d * d) == 0:
            return False
        d += 2
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form where b is squarefree."""
    pw = set()
    # b must be squarefree and b^3 <= limit
    b_max = int(limit ** (1.0/3.0)) + 1
    for b in range(1, b_max + 1):
        if not _is_square_free(b):
            continue
        b_cubed = b ** 3
        if b_cubed > limit:
            break
        a_max = isqrt(limit // b_cubed)
        for a in range(1, a_max + 1):
            val = a * a * b_cubed
            if val <= limit:
                pw.add(val)
    # Also add 1 (1 is powerful)
    pw.add(1)
    return sorted(pw)

def construct():
    start_time = time.time()
    deadline = 125.0  # safety margin before 150s kill
    
    # Generate powerful numbers up to a reasonable limit
    # Try progressively larger limits if time permits
    best_result = None
    best_score = 0
    
    # Start with a moderate limit
    for limit in [10**7, 10**8, 10**9]:
        if time.time() > deadline - 30:
            break
        
        pw = _generate_powerful(limit)
        pw_arr = np.array(pw, dtype=np.int64)
        n = len(pw_arr)
        
        if n < 2:
            continue
        
        # Compute all pairwise sums a[i] + a[j] where i <= j
        # Use numpy for efficiency
        # For memory, process in chunks if needed
        if n * n > 50_000_000:  # ~50M pairs, ~400MB for int64
            # Use chunked approach
            chunk_size = min(n, 5000)
            sum_counts = {}
            for i_start in range(0, n, chunk_size):
                i_end = min(i_start + chunk_size, n)
                chunk = pw_arr[i_start:i_end]
                # For each element in chunk, compute sums with all elements
                for idx in range(len(chunk)):
                    a = chunk[idx]
                    sums = a + pw_arr
                    # Only count a <= b, so filter where b >= a
                    valid = sums >= a
                    valid_sums = sums[valid]
                    # Count frequencies
                    vals, counts = np.unique(valid_sums, return_counts=True)
                    for v, c in zip(vals, counts):
                        sum_counts[v] = sum_counts.get(v, 0) + c
                if time.time() > deadline - 10:
                    break
            if sum_counts:
                best_N = max(sum_counts, key=sum_counts.get)
                best_count = sum_counts[best_N]
                if best_count > best_score:
                    # Find all pairs
                    pw_set = set(pw)
                    pairs = []
                    for a in pw:
                        b = best_N - a
                        if a <= b and b in pw_set:
                            pairs.append([a, b])
                    best_score = best_count
                    best_result = {'N': best_N, 'pairs': pairs}
        else:
            # Full pairwise sum
            # Create upper triangle indices
            i_idx, j_idx = np.triu_indices(n, k=0)
            sums = pw_arr[i_idx] + pw_arr[j_idx]
            vals, counts = np.unique(sums, return_counts=True)
            max_idx = np.argmax(counts)
            best_N = int(vals[max_idx])
            best_count = int(counts[max_idx])
            if best_count > best_score:
                pw_set = set(pw)
                pairs = []
                for a in pw:
                    b = best_N - a
                    if a <= b and b in pw_set:
                        pairs.append([a, b])
                best_score = best_count
                best_result = {'N': best_N, 'pairs': pairs}
        
        if time.time() > deadline - 15:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: use the original approach with a small limit
    limit = 20000
    powerful = sorted({a * a * b ** 3 for b in range(1, 28) for a in range(1, isqrt(limit // b ** 3) + 1)})
    members = set(powerful)
    best = max(range(2, limit + 1), key=lambda n: sum(1 for a in powerful if 2 * a <= n and n - a in members))
    return {'N': best, 'pairs': [[a, best - a] for a in powerful if 2 * a <= best and best - a in members]}
# EVOLVE-BLOCK-END
