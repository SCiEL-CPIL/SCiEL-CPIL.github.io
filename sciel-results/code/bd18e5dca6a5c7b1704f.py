import time
import math
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    pows = set([1])
    max_i = math.isqrt(limit)
    max_j_global = int(round(limit ** (1/3))) + 2
    cubes = [j ** 3 for j in range(1, max_j_global + 1)]
    
    for i in range(1, max_i + 1):
        sq = i * i
        if sq > limit:
            break
        max_j = int(round((limit // sq) ** (1/3)))
        if max_j > max_j_global:
            max_j = max_j_global
        for j in range(1, max_j + 1):
            val = sq * cubes[j - 1]
            if val <= limit:
                pows.add(val)
            else:
                break
    
    return sorted(pows)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_score = 1
    
    # Try increasing limits
    limits = [10_000_000, 50_000_000, 100_000_000, 200_000_000]
    
    for limit in limits:
        if time.time() > deadline - 30.0:
            break
        
        P_list = _generate_powerful(limit)
        K = len(P_list)
        
        if K == 0:
            continue
        
        P_set = set(P_list)
        P_arr = np.array(P_list, dtype=np.int64)
        
        # Use FFT to count representations
        size = 2 * limit + 1
        
        try:
            vec = np.zeros(size, dtype=np.float64)
            vec[P_arr] = 1.0
            
            fft_vec = np.fft.rfft(vec)
            conv = np.fft.irfft(fft_vec * fft_vec, n=size)
            counts = np.rint(conv).astype(np.int64)
            
            # Find top 50 candidates
            c_slice = counts[2:]
            if len(c_slice) < 50:
                top_indices_local = np.argsort(c_slice)[-len(c_slice):]
            else:
                top_indices_local = np.argpartition(c_slice, -50)[-50:]
            top_indices = top_indices_local + 2
            
            for s in top_indices:
                s_val = int(s)
                if s_val < 2 or s_val > 2 * limit:
                    continue
                
                # Collect pairs
                pairs = []
                half = s_val / 2.0
                for a in P_list:
                    if a > half:
                        break
                    b = s_val - a
                    if b in P_set:
                        pairs.append([a, b])
                
                score = len(pairs)
                if score > best_score:
                    best_score = score
                    best_N = s_val
                    best_pairs = pairs
                
                if time.time() > deadline - 10.0:
                    break
                    
        except MemoryError:
            break
        except Exception:
            continue
    
    return {'N': best_N, 'pairs': best_pairs}
