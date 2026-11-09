import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def generate_square_free(limit):
        """Sieve for square-free numbers up to limit."""
        if limit < 1:
            return []
        is_sq_free = [True] * (limit + 1)
        is_sq_free[0] = False
        p = 2
        while p * p <= limit:
            if is_sq_free[p]:
                sq = p * p
                for j in range(sq, limit + 1, sq):
                    is_sq_free[j] = False
            p += 1
        return [i for i in range(1, limit + 1) if is_sq_free[i]]

    def generate_powerful(limit):
        """Generate all powerful numbers up to limit."""
        powerful = set()
        max_v = int(limit ** (1/3)) + 1
        sq_free_v = generate_square_free(max_v)
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

    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_score = 1

    # Structural change: use a two-phase approach
    # Phase 1: Generate powerful numbers up to M, find top candidate N values
    # Phase 2: For each top candidate, verify and collect pairs
    # Key: use int32 for counts to halve memory, and process sums in blocks
    
    M = 40_000_000
    P = generate_powerful(M)
    L = len(P)
    P_np = np.array(P, dtype=np.int64)
    P_set = set(P)

    try:
        # Compute all pair sums
        sums = np.add.outer(P_np, P_np)
        max_sum_val = int(sums.max())
        
        # Use int32 bincount to save memory
        counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1).astype(np.int32)
        
        # Find top 20 candidate N values
        top_k = 20
        # Get indices of top_k counts
        top_indices = np.argsort(counts)[-top_k:][::-1]
        
        for idx in top_indices:
            if time.time() > deadline - 15:
                break
            N = int(idx)
            if N > 10**18:
                continue
            
            # Collect pairs for this N
            pairs = []
            for a in P:
                b = N - a
                if b < a:
                    continue
                if b in P_set:
                    pairs.append([a, b])
            
            if len(pairs) > best_score:
                best_score = len(pairs)
                best_result = {'N': N, 'pairs': pairs}
        
        # If we have time, try a larger M
        if time.time() < deadline - 40:
            M2 = 60_000_000
            P2 = generate_powerful(M2)
            L2 = len(P2)
            P2_np = np.array(P2, dtype=np.int64)
            P2_set = set(P2)
            
            sums2 = np.add.outer(P2_np, P2_np)
            max_sum_val2 = int(sums2.max())
            counts2 = np.bincount(sums2.ravel(), minlength=max_sum_val2 + 1).astype(np.int32)
            
            top_indices2 = np.argsort(counts2)[-top_k:][::-1]
            
            for idx in top_indices2:
                if time.time() > deadline - 15:
                    break
                N = int(idx)
                if N > 10**18:
                    continue
                
                pairs = []
                for a in P2:
                    b = N - a
                    if b < a:
                        continue
                    if b in P2_set:
                        pairs.append([a, b])
                
                if len(pairs) > best_score:
                    best_score = len(pairs)
                    best_result = {'N': N, 'pairs': pairs}
            
    except MemoryError:
        # Fallback: use smaller M
        M3 = 20_000_000
        P3 = generate_powerful(M3)
        L3 = len(P3)
        P3_np = np.array(P3, dtype=np.int64)
        P3_set = set(P3)
        
        sums3 = np.add.outer(P3_np, P3_np)
        max_sum_val3 = int(sums3.max())
        counts3 = np.bincount(sums3.ravel(), minlength=max_sum_val3 + 1)
        
        max_count_idx3 = np.argmax(counts3)
        N3 = int(max_count_idx3)
        
        pairs3 = []
        for a in P3:
            b = N3 - a
            if b < a:
                continue
            if b in P3_set:
                pairs3.append([a, b])
        
        if len(pairs3) > best_score:
            best_score = len(pairs3)
            best_result = {'N': N3, 'pairs': pairs3}
    except Exception:
        pass

    return best_result
