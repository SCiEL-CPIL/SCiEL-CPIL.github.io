import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def sieve_square_free(n):
        """Return a boolean array where True means square-free."""
        is_sq_free = np.ones(n + 1, dtype=bool)
        is_sq_free[0] = False
        limit = int(math.isqrt(n))
        for p in range(2, limit + 1):
            p2 = p * p
            # Mark multiples of p^2 as not square-free
            is_sq_free[p2::p2] = False
        return is_sq_free

    def generate_powerful(limit):
        """
        Generate all powerful numbers up to limit.
        A powerful number is of the form a^2 * b^3 where b is square-free.
        """
        powerful = set()
        powerful.add(1)
        
        # Max b such that b^3 <= limit
        max_b = int(round(limit ** (1/3))) + 2
        
        # Use sieve for efficient square-free detection
        if max_b < 2:
            sq_free_mask = np.ones(max_b + 1, dtype=bool)
            if max_b >= 0:
                sq_free_mask[0] = False
        else:
            sq_free_mask = sieve_square_free(max_b)
        
        sq_free_v = np.where(sq_free_mask[1:])[0] + 1  # Start from 1
        
        for v in sq_free_v:
            v3 = int(v) ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        
        return sorted(powerful)

    # Try different limits, starting high to maximize the search space
    # M=5e8 is a good target if memory allows
    limits_to_try = [500_000_000, 300_000_000, 200_000_000, 100_000_000]
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    
    for M in limits_to_try:
        if time.time() > deadline - 15:
            break
            
        try:
            # Generate powerful numbers
            P = generate_powerful(M)
            L = len(P)
            if L < 100:
                continue
                
            P_np = np.array(P, dtype=np.int64)
            P_set = set(P)
            
            # Max sum is 2*M
            max_sum = 2 * M
            
            # We need to count sums a+b.
            # Use chunking to avoid creating a full LxL array.
            # Count array size is max_sum + 1.
            # For M=5e8, max_sum=1e9. int32 array is 4GB.
            # If MemoryError, we catch it and try smaller M.
            
            counts = np.zeros(max_sum + 1, dtype=np.int32)
            
            # Larger chunk size reduces Python loop overhead
            # 5000 * L * 8 bytes. If L=60k, 2.4GB. May need to reduce.
            # Start with 2000 as a safe bet, can increase if memory allows.
            CHUNK_SIZE = 2000
            
            for i in range(0, L, CHUNK_SIZE):
                if time.time() > deadline - 8:
                    break
                end = min(i + CHUNK_SIZE, L)
                P_chunk = P_np[i:end]
                
                # Compute sums for this chunk: P_chunk[:, None] + P_np[None, :]
                # This creates a (chunk_size, L) array
                sums = P_chunk[:, None] + P_np[None, :]
                
                # Flatten and count
                # Note: We count all pairs (a,b) where a is in chunk and b is in P.
                # This includes a > b and a < b and a == b.
                # The problem asks for distinct pairs a <= b.
                # Total count C for sum N includes:
                # - Pairs with a < b: counted twice (once as a+b, once as b+a)
                # - Pairs with a == b: counted once
                # We want the number of pairs with a <= b.
                # Let K be the number of pairs with a < b.
                # Let D be 1 if N/2 is powerful, else 0.
                # Total count C = 2*K + D.
                # Number of pairs a <= b is K + D = (C + D) // 2.
                
                chunk_counts = np.bincount(sums.ravel(), minlength=max_sum + 1)
                counts += chunk_counts.astype(np.int32)
            
            # Find the sum with max count
            # We must be careful: we are counting ALL ordered pairs (a,b) from P.
            # The score is the number of DISTINCT pairs a <= b.
            # So we need to find N that maximizes (counts[N] + D(N)) // 2.
            
            # Iterate through possible N? No, max_sum is 1e9.
            # We can find top candidates from counts.
            # But D(N) depends on N.
            # D(N) = 1 if N is even and N/2 is in P, else 0.
            
            # Let's find the top 200 sums by raw count, then evaluate their true score.
            # This is much faster than iterating 1e9.
            
            # Get top 200 indices
            # np.argpartition is O(N), but we need top K.
            # If max_sum is 1e9, argpartition is fast enough?
            # Or we can just use argmax if we assume the max raw count is likely the max true score?
            # D(N) is at most 1. So (C+1)//2 vs C//2.
            # If C is large, the difference is negligible.
            # If C is small, it matters.
            # Let's find the top 200 sums to be safe.
            
            top_k = 200
            # np.argsort is O(N log N). 1e9 log 1e9 is too slow.
            # np.argpartition is O(N).
            part_indices = np.argpartition(counts, -top_k)[-top_k:]
            
            best_N = 0
            best_score = 0
            best_pairs = []
            
            for idx in part_indices:
                N = int(idx)
                C = int(counts[N])
                if C == 0:
                    continue
                D = 1 if (N % 2 == 0 and (N // 2) in P_set) else 0
                score = (C + D) // 2
                
                if score > best_score:
                    best_score = score
                    best_N = N
                    # Recover pairs
                    pairs = []
                    for a in P:
                        b = N - a
                        if b < a:
                            continue
                        if b in P_set:
                            pairs.append([a, b])
                    best_pairs = pairs
            
            if best_score > 0:
                current_result = {'N': best_N, 'pairs': best_pairs}
                if len(best_pairs) > len(best_result['pairs']):
                    best_result = current_result
                
                # If we found a good result with the largest M, we can return it?
                # Usually larger M gives better results.
                # But if M=5e8 works, it's likely the best we can do in time.
                return current_result
                
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
        sums = np.add.outer(P_np, P_np)
        counts = np.bincount(sums.ravel(), minlength=sums.max() + 1)
        best_N = int(np.argmax(counts))
        C = int(counts[best_N])
        D = 1 if (best_N % 2 == 0 and best_N // 2 in P) else 0
        pairs = []
        P_set = set(P)
        for a in P:
            b = best_N - a
            if b >= a and b in P_set:
                pairs.append([a, b])
        return {'N': best_N, 'pairs': pairs}
    except:
        return {'N': 2, 'pairs': [[1, 1]]}
