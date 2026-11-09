import time
import math
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        powerful = set()
        max_v = int(limit ** (1/3)) + 1
        sq_free_v = []
        for v in range(1, max_v + 1):
            if is_square_free(v):
                sq_free_v.append(v)
                
        for v in sq_free_v:
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        return sorted(powerful)

    best_result = None
    
    for M in [500_000_000, 300_000_000, 200_000_000, 100_000_000]:
        if time.time() > deadline - 30:
            break
            
        try:
            P = generate_powerful(M)
            if len(P) < 100:
                continue
                
            P_np = np.array(P, dtype=np.int64)
            n = len(P_np)
            
            # Create boolean lookup array
            P_bool = np.zeros(M + 1, dtype=bool)
            P_bool[P_np] = True
            
            # Sample candidate Ns
            random.seed(42)
            candidates = set()
            num_samples = min(100000, n * n // 10)
            for _ in range(num_samples):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                candidates.add(int(P_np[i] + P_np[j]))
            
            # Count representations for each candidate
            best_N = None
            best_count = 0
            best_pairs = []
            
            for N in candidates:
                if time.time() > deadline - 10:
                    break
                b_vals = N - P_np
                valid = (b_vals >= 0) & (b_vals <= M)
                if not np.any(valid):
                    continue
                in_P = P_bool[b_vals[valid]]
                ge_a = b_vals[valid] >= P_np[valid]
                count = int(np.sum(in_P & ge_a))
                if count > best_count:
                    best_count = count
                    best_N = N
                    mask = in_P & ge_a
                    valid_indices = np.where(valid)[0]
                    pairs = []
                    for idx in valid_indices[mask]:
                        a = int(P_np[idx])
                        b = int(N - a)
                        pairs.append([a, b])
                    best_pairs = pairs
            
            if best_N is not None:
                current_result = {'N': best_N, 'pairs': best_pairs}
                if best_result is None or len(best_pairs) > len(best_result['pairs']):
                    best_result = current_result
                return current_result
                
        except MemoryError:
            continue
        except Exception:
            continue

    if best_result:
        return best_result
        
    return {'N': 2, 'pairs': [[1, 1]]}
