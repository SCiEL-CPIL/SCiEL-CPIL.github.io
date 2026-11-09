import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_square_free(limit):
        """Generate all square-free numbers up to limit using sieve."""
        if limit < 1:
            return []
        sq_free = [True] * (limit + 1)
        sq_free[0] = False
        for p in range(2, int(math.isqrt(limit)) + 1):
            if sq_free[p]:
                p2 = p * p
                for multiple in range(p2, limit + 1, p2):
                    sq_free[multiple] = False
        return [i for i in range(1, limit + 1) if sq_free[i]]

    def generate_powerful(limit):
        """
        Generate all powerful numbers up to limit.
        A powerful number is of the form a^2 * b^3 where b is square-free.
        """
        powerful = set()
        powerful.add(1)
        
        # Max b such that b^3 <= limit
        max_b = int(round(limit ** (1/3))) + 2
        
        # Generate square-free numbers up to max_b using sieve
        sq_free_v = generate_square_free(max_b)
        
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

    def verify_N(N, P, P_set):
        """Verify N by finding all pairs a <= b with a+b=N where a,b are powerful."""
        pairs = []
        for a in P:
            b = N - a
            if b < a:
                continue
            if b > N:
                continue
            if b in P_set:
                pairs.append([a, b])
        return pairs

    def probabilistic_estimate(N, P, P_set, sample_size=5000):
        """
        Estimate the number of powerful representations of N using random sampling.
        For each powerful number a, check if N-a is powerful.
        """
        L = len(P)
        if L == 0:
            return 0
        
        # Sample from P
        if L <= sample_size:
            indices = list(range(L))
        else:
            indices = np.random.choice(L, size=sample_size, replace=False)
        
        count = 0
        for idx in indices:
            a = P[idx]
            b = N - a
            if b >= a and b <= 2 * max(P) and b in P_set:
                count += 1
        
        # Extrapolate: expected count = (count / sample_size) * L
        # But we only want a <= b, so we need to be careful.
        # For each valid pair (a,b) with a<=b, it's counted once if a is in sample.
        # Expected number of pairs where first element is in sample = (sample_size/L) * (total pairs)
        # So total pairs ≈ count * L / sample_size
        return int(count * L / sample_size)

    def find_best_N_probabilistic(P, P_set, M, time_budget):
        """
        Use probabilistic heuristic to find candidate N values, then verify them.
        """
        L = len(P)
        if L < 10:
            return 0, [], 0
        
        # The density of powerful numbers up to M is roughly |P| / M
        # For a random N in [2, 2M], expected number of representations:
        # E[# pairs] = sum over a in P of P(N-a is powerful)
        # ≈ L * (L / M) = L^2 / M (roughly)
        
        # We want to maximize this. The distribution of N that maximizes representations
        # tends to be around the "center" where there's more room for pairs.
        
        # Strategy: Try several candidate N values based on:
        # 1. N = 2 * median(P) - some offset
        # 2. N values that are sums of small powerful numbers (more likely to have many reps)
        
        candidates = set()
        
        # Candidate 1: N around 2 * median
        median_val = P[L // 2]
        for offset in range(-50, 51):
            N_cand = 2 * median_val + offset
            if 2 <= N_cand <= 2 * M:
                candidates.add(N_cand)
        
        # Candidate 2: N = a + b for small a, b (first few powerful numbers)
        small_P = P[:100]
        for a in small_P[:50]:
            for b in small_P[:50]:
                N_cand = a + b
                if 2 <= N_cand <= 2 * M:
                    candidates.add(N_cand)
        
        # Candidate 3: N values that are multiples of small powerful numbers
        for mult in range(1, 20):
            for a in small_P[:30]:
                N_cand = mult * a
                if 2 <= N_cand <= 2 * M:
                    candidates.add(N_cand)
        
        # Sort candidates by estimated probability (heuristic: prefer N in middle range)
        # N closer to M has more room for pairs
        candidates = sorted(candidates, key=lambda x: -min(x, 2*M - x))
        
        # Take top candidates within time budget
        max_candidates = min(len(candidates), 200)
        candidates = candidates[:max_candidates]
        
        best_score = 0
        best_N = 0
        best_pairs = []
        
        for N_cand in candidates:
            if time.time() > time_budget:
                break
            
            # Quick probabilistic estimate
            est = probabilistic_estimate(N_cand, P, P_set, sample_size=2000)
            
            # If estimate is promising, verify exactly
            if est >= best_score:
                pairs = verify_N(N_cand, P, P_set)
                actual_score = len(pairs)
                
                if actual_score > best_score:
                    best_score = actual_score
                    best_N = N_cand
                    best_pairs = pairs
        
        return best_N, best_pairs, best_score

    def full_pair_count(P, P_set, M, time_budget):
        """
        Full O(L^2) pair counting for smaller M values.
        """
        L = len(P)
        if L < 10:
            return 0, [], 0
        
        P_np = np.array(P, dtype=np.int64)
        max_sum = 2 * M
        
        # Use chunking
        CHUNK_SIZE = min(1000, L)
        
        # Only allocate if memory allows
        if max_sum > 500_000_000:
            # Too large for full array, use probabilistic method
            return find_best_N_probabilistic(P, P_set, M, time_budget)
        
        try:
            counts = np.zeros(max_sum + 1, dtype=np.int32)
        except MemoryError:
            return find_best_N_probabilistic(P, P_set, M, time_budget)
        
        for i in range(0, L, CHUNK_SIZE):
            if time.time() > time_budget:
                break
            end = min(i + CHUNK_SIZE, L)
            P_chunk = P_np[i:end]
            sums = P_chunk[:, None] + P_np[None, :]
            chunk_counts = np.bincount(sums.ravel(), minlength=max_sum + 1)
            counts += chunk_counts.astype(np.int32)
        
        # Find top candidates
        top_k = min(100, max_sum + 1)
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
                pairs = verify_N(N, P, P_set)
                best_pairs = pairs
        
        return best_N, best_pairs, best_score

    # Try different limits, starting high to maximize the search space
    limits_to_try = [1_000_000_000, 500_000_000, 300_000_000, 200_000_000, 100_000_000]
    
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
                
            P_set = set(P)
            
            # Use probabilistic method for large M, full count for smaller M
            if M >= 500_000_000:
                best_N, best_pairs, best_score = find_best_N_probabilistic(P, P_set, M, deadline - 10)
            else:
                best_N, best_pairs, best_score = full_pair_count(P, P_set, M, deadline - 10)
            
            if best_score > 0 and best_N > 0:
                current_result = {'N': best_N, 'pairs': best_pairs}
                if len(best_pairs) > len(best_result['pairs']):
                    best_result = current_result
                
                # If we found a good result, we can try to improve with more candidates
                # but return if we have a strong result
                if best_score >= 5:
                    return current_result
                
        except MemoryError:
            continue
        except Exception:
            continue

    # Fallback
    if best_result:
        return best_result
        
    try:
        P = generate_powerful(10000)
        P_set = set(P)
        best_N, best_pairs, best_score = find_best_N_probabilistic(P, P_set, 10000, deadline - 5)
        if best_score > 0:
            return {'N': best_N, 'pairs': best_pairs}
        
        # Final fallback
        P_np = np.array(P, dtype=np.int64)
        sums = np.add.outer(P_np, P_np)
        counts = np.bincount(sums.ravel(), minlength=sums.max() + 1)
        best_N = int(np.argmax(counts))
        C = int(counts[best_N])
        D = 1 if (best_N % 2 == 0 and best_N // 2 in P_set) else 0
        pairs = verify_N(best_N, P, P_set)
        return {'N': best_N, 'pairs': pairs}
    except:
        return {'N': 2, 'pairs': [[1, 1]]}
