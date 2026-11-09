import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        powerful = set()
        # Max v such that v^3 <= limit
        max_v = int(limit ** (1/3)) + 1
        # Precompute square-free v
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
    
    # Phase 1: Quick search with smaller limit to find promising N
    # Phase 2: Verify and extend with larger limit if time permits
    
    # Try multiple limits, starting from a reasonable size
    limits = [100_000_000, 50_000_000, 20_000_000, 10_000_000]
    
    for M in limits:
        if time.time() > deadline - 20:
            break
            
        try:
            P = generate_powerful(M)
            L = len(P)
            
            if L < 10 or L > 20000:
                continue
                
            P_np = np.array(P, dtype=np.int64)
            
            # Use a more memory-efficient counting approach
            # Instead of full outer sum, use a hash map for counts
            # This works well when L is moderate
            
            from collections import defaultdict
            counts = defaultdict(int)
            
            for i in range(L):
                a = P_np[i]
                for j in range(i, L):
                    b = P_np[j]
                    s = a + b
                    counts[s] += 1
            
            # Find the sum with max count
            if not counts:
                continue
                
            best_N = int(max(counts.keys(), key=lambda x: counts[x]))
            C = counts[best_N]
            
            # Calculate distinct pairs count
            P_set = set(P)
            pairs = []
            for a in P:
                b = best_N - a
                if b >= a and b in P_set:
                    pairs.append([int(a), int(b)])
            
            num_pairs = len(pairs)
            
            current_result = {'N': best_N, 'pairs': pairs}
            
            if best_result is None or num_pairs > len(best_result['pairs']):
                best_result = current_result
                
            # If we found a good number, we might break
            # But larger M is better, so continue
            # However, if we're running out of time, break
            if time.time() > deadline - 30:
                break

        except MemoryError:
            continue
        except Exception:
            continue

    # Fallback
    if best_result:
        return best_result
        
    try:
        P = generate_powerful(1000)
        P_np = np.array(P, dtype=np.int64)
        from collections import defaultdict
        counts = defaultdict(int)
        L = len(P_np)
        for i in range(L):
            a = P_np[i]
            for j in range(i, L):
                b = P_np[j]
                s = a + b
                counts[s] += 1
        best_N = int(max(counts.keys(), key=lambda x: counts[x]))
        P_set = set(P)
        pairs = []
        for a in P:
            b = best_N - a
            if b >= a and b in P_set:
                pairs.append([int(a), int(b)])
        return {'N': best_N, 'pairs': pairs}
    except:
        return {'N': 2, 'pairs': [[1, 1]]}
