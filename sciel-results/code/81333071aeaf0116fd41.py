"""Find N <= 10^18 with many representations as sum of two powerful numbers."""
import time
import random
from math import isqrt
import numpy as np

def _is_square_free(n):
    """Check if n is square-free."""
    if n < 2:
        return True
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    if n == 1:
        return True
    r = isqrt(n)
    if r * r == n:
        return False
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers <= limit.
    Powerful numbers are of the form a^2 * b^3 where b is square-free.
    """
    powerful = set([1])
    
    # Max b: b^3 <= limit, so b <= limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    
    # Precompute square-free status for b up to max_b using sieve
    sq_free = [True] * (max_b + 1)
    if max_b >= 0:
        sq_free[0] = False
    if max_b >= 1:
        sq_free[1] = True
    
    # Sieve for square-free
    limit_sieve = int(max_b ** (1/2)) + 2
    for p in range(2, limit_sieve):
        if sq_free[p]:
            p2 = p * p
            for k in range(p2, max_b + 1, p2):
                sq_free[k] = False
    
    for b in range(1, max_b + 1):
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    
    return sorted(powerful)

def _count_reps(N, powerful_list, powerful_set):
    """Count the number of pairs (a, b) with a <= b, a + b = N, a, b in powerful_set."""
    count = 0
    pairs = []
    for a in powerful_list:
        if 2 * a > N:
            break
        b = N - a
        if b in powerful_set:
            count += 1
            pairs.append([a, b])
    return count, pairs

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer before 150s limit
    
    # Generate powerful numbers up to 10^8 for fast vectorized O(K^2) counting
    limit_pw = 10**8
    
    powerful = _generate_powerful(limit_pw)
    powerful_set = set(powerful)
    n_pw = len(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > 30:
        # If generation took too long, reduce
        limit_pw = 10**7
        powerful = _generate_powerful(limit_pw)
        powerful_set = set(powerful)
        n_pw = len(powerful)
    
    # Use NumPy for vectorized sum counting if feasible
    if n_pw <= 30000:
        # Full O(n^2) enumeration using NumPy for speed
        arr = np.array(powerful, dtype=np.int64)
        # We need all pairs (i, j) with i <= j such that arr[i] + arr[j] <= 10^18
        # Since max arr is 10^8, max sum is 2*10^8 which is well under 10^18
        # So all pairs are valid
        
        # Build upper triangular indices
        i_idx, j_idx = np.triu_indices(n_pw)
        sums = arr[i_idx] + arr[j_idx]
        
        # Count frequencies using np.bincount or similar
        # Since sums can be up to 2*10^8, we can use a dictionary or sort-based counting
        # Sort and count
        sums_sorted = np.sort(sums)
        # Find unique sums and their counts
        unique_sums, counts = np.unique(sums_sorted, return_counts=True)
        
        if len(unique_sums) > 0:
            best_idx = np.argmax(counts)
            best_n = int(unique_sums[best_idx])
            best_count = int(counts[best_idx])
            
            # Verify and collect pairs
            count, pairs = _count_reps(best_n, powerful, powerful_set)
            if pairs:
                return {'N': best_n, 'pairs': pairs}
    else:
        # For larger sets, use sampling approach
        random.seed(42)
        sum_count = {}
        sample_size = min(5_000_000, (n_pw * n_pw) // 10)
        
        for _ in range(sample_size):
            if time.time() > deadline - 30:
                break
            i = random.randint(0, n_pw - 1)
            j = random.randint(0, n_pw - 1)
            if i > j:
                i, j = j, i
            a = powerful[i]
            b = powerful[j]
            s = a + b
            if s <= 10**18:
                sum_count[s] = sum_count.get(s, 0) + 1
        
        # Get top candidates
        if sum_count:
            top_candidates = sorted(sum_count.items(), key=lambda x: -x[1])[:50]
            
            # Verify each candidate with full scan
            best_n = None
            best_count = 0
            best_pairs = []
            
            for cand_n, _ in top_candidates:
                if time.time() > deadline - 10:
                    break
                count, pairs = _count_reps(cand_n, powerful, powerful_set)
                if count > best_count:
                    best_count = count
                    best_n = cand_n
                    best_pairs = pairs
            
            if best_n is not None and best_pairs:
                return {'N': best_n, 'pairs': best_pairs}
    
    # Fallback: try a few more random candidates
    random.seed(123)
    best_n = None
    best_pairs = []
    best_count = 0
    
    for _ in range(50000):
        if time.time() > deadline - 5:
            break
        idx1 = random.randint(0, n_pw - 1)
        idx2 = random.randint(0, n_pw - 1)
        n_candidate = powerful[idx1] + powerful[idx2]
        
        if n_candidate > 10**18 or n_candidate < 2:
            continue
        
        count, pairs = _count_reps(n_candidate, powerful, powerful_set)
        
        if count > best_count:
            best_count = count
            best_n = n_candidate
            best_pairs = pairs
    
    if best_n is not None and best_pairs:
        return {'N': best_n, 'pairs': best_pairs}
    
    # Last resort: return a simple valid answer
    return {'N': 2, 'pairs': [[1, 1]]}
