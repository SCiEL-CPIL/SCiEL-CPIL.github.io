import time
from math import isqrt
import numpy as np
import bisect

def _generate_powerfuls(limit):
    """
    Generate all powerful numbers <= limit.
    """
    powerfuls = set()
    max_b = int(limit ** (1/3)) + 2
    
    sqrt_max_b = isqrt(max_b)
    primes = []
    for x in range(2, sqrt_max_b + 1):
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
    
    for b in range(1, max_b + 1):
        is_sq_free = True
        for p in primes:
            if p * p > b:
                break
            if b % (p * p) == 0:
                is_sq_free = False
                break
        if not is_sq_free:
            continue
        
        b3 = b ** 3
        if b3 > limit:
            break
        
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerfuls.add(val)
    
    return powerfuls

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Strategy: Memory-optimized exhaustive search using int16 indices.
    # This allows a larger L than the previous int64/int32 index approach.
    
    # Try L = 2 * 10^8 first.
    # P ~ sqrt(2*10^8) * C ~ 14000 * 2 ~ 28000.
    # P^2/2 ~ 4*10^8.
    # Indices (int16): 2 * 4*10^8 * 2 bytes = 1.6 GB.
    # Sums (int32): 4*10^8 * 4 bytes = 1.6 GB.
    # Total ~ 3.2 GB.
    
    L = 200_000_000
    
    powerfuls = _generate_powerfuls(L)
    powerfuls_list = sorted(powerfuls)
    powerfuls_set = set(powerfuls_list)
    P = len(powerfuls_list)
    
    elapsed_gen = time.time() - start_time
    if elapsed_gen > 20:
        # Generation too slow, reduce L
        L = 100_000_000
        powerfuls = _generate_powerfuls(L)
        powerfuls_list = sorted(powerfuls)
        powerfuls_set = set(powerfuls_list)
        P = len(powerfuls_list)

    if time.time() > deadline - 30:
        # Fallback to small brute force
        small_limit = 10000
        small_pows = [p for p in powerfuls_list if p <= small_limit]
        count_dict = {}
        pair_dict = {}
        for i in range(len(small_pows)):
            a = small_pows[i]
            for j in range(i, len(small_pows)):
                b = small_pows[j]
                s = a + b
                if s > small_limit * 2: break
                if s not in count_dict:
                    count_dict[s] = 0
                    pair_dict[s] = []
                count_dict[s] += 1
                pair_dict[s].append([a, b])
        if count_dict:
            best_N = max(count_dict, key=count_dict.get)
            return {'N': best_N, 'pairs': pair_dict[best_N]}
        return {'N': 2, 'pairs': [[1, 1]]}

    try:
        # Convert to numpy arrays
        # Use int32 for values to save memory in sums
        pows_arr = np.array(powerfuls_list, dtype=np.int32)
        
        # Use int16 for indices if P < 65536
        # P is approx 28000, so int16 is safe.
        if P > 65535:
            # Fall back to int32 indices if P is too large for int16
            # This will use more memory, might OOM.
            i_idx, j_idx = np.triu_indices(P, dtype=np.int32)
        else:
            i_idx, j_idx = np.triu_indices(P, dtype=np.int16)
            
        # Compute sums
        # pows_arr is int32. i_idx, j_idx are int16 or int32.
        # Result will be int32.
        sums = pows_arr[i_idx] + pows_arr[j_idx]
        
        # Find unique sums and counts
        # np.unique sorts the array.
        # This is the bottleneck.
        unique_sums, counts = np.unique(sums, return_counts=True)
        
        # Find the index with max count
        # We want N >= 2.
        # unique_sums is sorted.
        # Find max count.
        max_count = np.max(counts)
        
        # There might be multiple N with the same max count.
        # We just need one.
        # Find the first one with max count.
        best_idx = np.argmax(counts)
        best_N = int(unique_sums[best_idx])
        
        # Verify best_N
        if best_N < 2:
             # Fallback
             return {'N': 2, 'pairs': [[1, 1]]}
             
        # Extract pairs for best_N
        pairs = []
        half_N = best_N // 2
        end_idx = bisect.bisect_right(powerfuls_list, half_N)
        
        for a in powerfuls_list[:end_idx]:
            b = best_N - a
            if b in powerfuls_set:
                if a <= b:
                    pairs.append([int(a), int(b)])
        
        if pairs:
            return {'N': best_N, 'pairs': pairs}
        else:
            return {'N': 2, 'pairs': [[1, 1]]}
            
    except MemoryError:
        # Fallback to smaller L
        L = 100_000_000
        powerfuls = _generate_powerfuls(L)
        powerfuls_list = sorted(powerfuls)
        powerfuls_set = set(powerfuls_list)
        P = len(powerfuls_list)
        pows_arr = np.array(powerfuls_list, dtype=np.int32)
        
        if P > 0:
            i_idx, j_idx = np.triu_indices(P, dtype=np.int16)
            sums = pows_arr[i_idx] + pows_arr[j_idx]
            unique_sums, counts = np.unique(sums, return_counts=True)
            best_idx = np.argmax(counts)
            best_N = int(unique_sums[best_idx])
            
            if best_N < 2: best_N = 2
            
            pairs = []
            half_N = best_N // 2
            end_idx = bisect.bisect_right(powerfuls_list, half_N)
            for a in powerfuls_list[:end_idx]:
                b = best_N - a
                if b in powerfuls_set:
                    if a <= b:
                        pairs.append([int(a), int(b)])
            
            if pairs:
                return {'N': best_N, 'pairs': pairs}
                
        return {'N': 2, 'pairs': [[1, 1]]}
    except Exception:
        return {'N': 2, 'pairs': [[1, 1]]}
