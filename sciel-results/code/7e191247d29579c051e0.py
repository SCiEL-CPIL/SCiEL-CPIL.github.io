import time
import numpy as np
from math import isqrt
from bisect import bisect_left, bisect_right

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit.
    Powerful = a^2 * b^3 where b is square-free.
    """
    max_b = int(limit ** (1.0 / 3.0)) + 2
    is_sq_free = np.ones(max_b + 1, dtype=bool)
    is_sq_free[0] = False
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        is_sq_free[sq::sq] = False
    is_sq_free[1] = True

    powerful = set()
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return powerful

def _count_pairs(N, powerful_list, powerful_set):
    """Count unordered pairs (a,b) with a<=b, a+b=N, both powerful."""
    count = 0
    pairs = []
    for a in powerful_list:
        if a > N // 2:
            break
        b = N - a
        if b in powerful_set:
            count += 1
            pairs.append([a, b])
    return count, pairs

def construct():
    t0 = time.time()
    deadline = t0 + 115

    best_result = None
    best_count = 0

    # Strategy: Candidate-based counting with progressively larger limits.
    # For each limit, generate powerful numbers, then sample N candidates
    # and count their representations. Larger limits explore more N values.
    
    limits_to_try = [10_000_000, 30_000_000, 50_000_000]
    
    for limit in limits_to_try:
        if time.time() > deadline - 30:
            break

        powerful = _generate_powerful(limit)
        n_powerful = len(powerful)
        
        if n_powerful < 10:
            continue

        powerful_list = sorted(powerful)
        powerful_set = powerful

        # Use FFT convolution for smaller limits to find top candidates
        if limit <= 30_000_000:
            size = 1
            while size < 2 * limit + 1:
                size *= 2
            
            indicator = np.zeros(size, dtype=np.float64)
            for p in powerful_list:
                indicator[p] = 1.0
            
            fft_ind = np.fft.fft(indicator)
            conv = np.fft.ifft(fft_ind * fft_ind)
            conv_real = np.round(np.real(conv)).astype(np.int64)
            
            max_N_candidate = 2 * limit
            top_k = min(300, max_N_candidate - 1)
            candidate_indices = np.argsort(conv_real[2:max_N_candidate + 1])[-top_k:][::-1] + 2
            candidates = [int(idx) for idx in candidate_indices]
        else:
            # For larger limits, use heuristic candidate selection
            # Sample N values that are sums of small powerful numbers (likely to have many reps)
            candidates = []
            # Use the smallest powerful numbers to generate likely-high-count N values
            small_pw = powerful_list[:min(5000, n_powerful)]
            seen = set()
            for i in range(min(2000, len(small_pw))):
                for j in range(i, min(i + 200, len(small_pw))):
                    N_cand = small_pw[i] + small_pw[j]
                    if 2 <= N_cand <= 2 * limit and N_cand not in seen:
                        seen.add(N_cand)
                        candidates.append(N_cand)
                    if len(candidates) >= 500:
                        break
                if len(candidates) >= 500:
                    break
            
            # Add some random candidates
            for _ in range(200):
                N_cand = int(np.random.randint(2, 2 * limit + 1))
                if N_cand not in seen:
                    seen.add(N_cand)
                    candidates.append(N_cand)
            
            candidates = candidates[:500]

        # Sort candidates by a heuristic: prefer N that are sums of many small powerful numbers
        # (approximated by conv value if available, else just process in order)
        
        for N_candidate in candidates:
            if time.time() > deadline - 10:
                break
            
            count, pairs = _count_pairs(N_candidate, powerful_list, powerful_set)
            
            if count > best_count:
                best_count = count
                best_result = {'N': N_candidate, 'pairs': pairs}
        
        # Additional random sampling for this limit
        n_random = 50
        for _ in range(n_random):
            if time.time() > deadline - 5:
                break
            N_candidate = int(np.random.randint(2, 2 * limit + 1))
            count, pairs = _count_pairs(N_candidate, powerful_list, powerful_set)
            if count > best_count:
                best_count = count
                best_result = {'N': N_candidate, 'pairs': pairs}

    if best_result is None:
        limit = 100_000
        powerful = _generate_powerful(limit)
        powerful_list = sorted(powerful)
        powerful_set = powerful
        best_result = {'N': 2, 'pairs': [[1, 1]]}

    return best_result
