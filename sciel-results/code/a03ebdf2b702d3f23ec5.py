import numpy as np
import time
from math import isqrt

def _is_square_free(n):
    if n < 2:
        return True
    # Primes up to sqrt(1000) ~ 32
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for p in primes:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
    return True

def _generate_powerful(limit):
    powerful = set()
    max_b = int(limit ** (1/3)) + 1
    for b in range(1, max_b + 1):
        if not _is_square_free(b):
            continue
        b3 = b ** 3
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
    deadline = start_time + 110  # 10s safety margin
    
    # Strategy: Try largest feasible FFT size first
    # L = 2^31 allows N up to ~2*10^9
    # Powerful numbers up to 10^9
    
    limits = [10**9, 10**8, 10**7]
    fft_sizes = [1 << 31, 1 << 30, 1 << 29]
    
    best_result = None
    best_score = -1
    
    for limit, L in zip(limits, fft_sizes):
        if time.time() > deadline:
            break
            
        try:
            # 1. Generate powerful numbers
            powerful_list = _generate_powerful(limit)
            if not powerful_list:
                continue
            
            powerful_set = set(powerful_list)
            
            # 2. Prepare array
            A = np.zeros(L, dtype=np.float32)
            for p in powerful_list:
                if p < L:
                    A[p] = 1.0
                else:
                    # This shouldn't happen if limit < L/2
                    # But if limit=10^9 and L=2^31 ~ 2.1*10^9, it's fine.
                    pass
            
            # 3. FFT
            F = np.fft.rfft(A)
            F2 = F * F
            conv = np.fft.irfft(F2, n=L)
            
            # 4. Find max
            # We only care about N up to 2*limit
            search_range = min(2 * limit, L)
            max_idx = np.argmax(conv[:search_range])
            N = max_idx
            score = conv[max_idx]
            
            if score > best_score:
                best_score = score
                # Extract pairs
                pairs = []
                for a in powerful_list:
                    if a > N // 2:
                        break
                    b = N - a
                    if b in powerful_set:
                        pairs.append([a, b])
                
                best_result = {'N': N, 'pairs': pairs}
                
        except MemoryError:
            continue
        except Exception:
            continue

    if best_result is None:
        # Fallback to simple search
        limit = 10**6
        powerful_list = _generate_powerful(limit)
        powerful_set = set(powerful_list)
        best_n = 2
        best_pairs = []
        best_count = 0
        for n in range(2, limit + 1):
            count = 0
            pairs = []
            for a in powerful_list:
                if a > n // 2:
                    break
                b = n - a
                if b in powerful_set:
                    count += 1
                    pairs.append([a, b])
            if count > best_count:
                best_count = count
                best_n = n
                best_pairs = pairs
        
        best_result = {'N': best_n, 'pairs': best_pairs}

    return best_result
