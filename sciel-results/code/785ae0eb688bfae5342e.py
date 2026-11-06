import time
import sys
import numpy as np
from math import isqrt

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit using u^2 * v^3 representation."""
    powerful = set()
    max_v = int(limit ** (1.0 / 3.0)) + 2
    
    for v in range(1, max_v + 1):
        v3 = v * v * v
        if v3 > limit:
            break
        max_u = isqrt(limit // v3)
        for u in range(1, max_u + 1):
            val = u * u * v3
            if val <= limit:
                powerful.add(val)
            else:
                break
    return sorted(powerful)

def _count_pairs_for_N(N, powerful_list, powerful_set):
    """Count pairs (a, b) with a <= b, a + b = N, both powerful."""
    count = 0
    pairs = []
    half_N = N // 2
    for a in powerful_list:
        if a > half_N:
            break
        b = N - a
        if b in powerful_set:
            count += 1
            pairs.append([a, b])
    return count, pairs

def _count_pairs_numpy(N, powerful_arr, powerful_set):
    """Vectorized count using numpy for faster verification."""
    half_N = N // 2
    mask = powerful_arr <= half_N
    a_candidates = powerful_arr[mask]
    b_candidates = N - a_candidates
    # Check which b are in powerful_set
    valid = np.array([b in powerful_set for b in b_candidates], dtype=bool)
    count = int(np.sum(valid))
    pairs = [[int(a), int(b)] for a, b, v in zip(a_candidates, b_candidates, valid) if v]
    return count, pairs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_result = {'N': 5, 'pairs': [[1, 4]]}
    best_count_global = 1
    
    # Try multiple strategies with increasing M
    # Key insight: larger M gives more powerful numbers, denser pair sum clusters
    limits_to_try = [500_000_000, 400_000_000, 300_000_000, 200_000_000]
    
    for M in limits_to_try:
        try:
            if time.time() > deadline - 60:
                break
                
            # 1. Generate powerful numbers
            gen_start = time.time()
            powerful_list = _generate_powerful_numbers(M)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 50:
                continue
                
            K = len(powerful_list)
            if K == 0:
                continue
            
            powerful_arr = np.array(powerful_list, dtype=np.int64)
            powerful_set = set(powerful_list)
            
            # 2. FFT Convolution
            S = 2 * M + 1
            
            # Use float64 for precision
            arr = np.zeros(S, dtype=np.float64)
            arr[powerful_list] = 1.0
            
            # Perform FFT
            fft_arr = np.fft.rfft(arr)
            conv = np.fft.irfft(fft_arr * fft_arr, n=S)
            
            # 3. Find best N
            top_k = 500
            if S > top_k:
                top_indices = np.argpartition(conv, -top_k)[-top_k:]
            else:
                top_indices = np.arange(S)
                
            top_indices = top_indices[np.argsort(conv[top_indices])[::-1]]
            
            best_N = 2
            best_count = 1
            best_pairs = [[1, 1]]
            
            # Verify top candidates
            for idx in top_indices:
                if time.time() > deadline - 10:
                    break
                    
                N_cand = int(idx)
                
                # Use numpy vectorized count
                count, exact_pairs = _count_pairs_numpy(N_cand, powerful_arr, powerful_set)
                
                if count > best_count:
                    best_count = count
                    best_N = N_cand
                    best_pairs = exact_pairs
                    
                # Early termination: if conv value is below current best, 
                # and we've gone past the peak region, stop
                if conv[idx] < 2 * best_count + 2:
                    break
            
            # Update best result
            if best_count > len(best_result['pairs']):
                best_result = {'N': best_N, 'pairs': best_pairs}
                best_count_global = best_count
                
            # If we found a good result, we can try to improve with more candidates
            if best_count_global >= 5 and time.time() < deadline - 20:
                # Try additional candidates beyond top_k
                extended_k = 2000
                if S > extended_k:
                    extended_indices = np.argpartition(conv, -extended_k)[-extended_k:]
                else:
                    extended_indices = np.arange(S)
                extended_indices = extended_indices[np.argsort(conv[extended_indices])[::-1]]
                
                for idx in extended_indices:
                    if time.time() > deadline - 10:
                        break
                    N_cand = int(idx)
                    if N_cand == best_N:
                        continue
                    count, exact_pairs = _count_pairs_numpy(N_cand, powerful_arr, powerful_set)
                    if count > best_count_global:
                        best_count_global = count
                        best_result = {'N': N_cand, 'pairs': exact_pairs}
                    if conv[idx] < 2 * best_count_global + 2:
                        break
                
        except MemoryError:
            continue
        except Exception:
            continue

    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
