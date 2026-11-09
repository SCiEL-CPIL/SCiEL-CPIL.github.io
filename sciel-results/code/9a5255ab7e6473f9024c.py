import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        """Generate all powerful numbers up to limit using the standard method."""
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

    def count_pairs_for_N(N, P_set, P_list):
        """Count pairs (a,b) with a<=b, a+b=N, both in P_set."""
        pairs = []
        for a in P_list:
            if a > N // 2:
                break
            b = N - a
            if b in P_set:
                pairs.append([a, b])
        return pairs

    def is_powerful(n):
        """Check if n is powerful by exact factorization."""
        if n < 1:
            return False
        if n == 1:
            return True
        temp = n
        for p in range(2, int(math.isqrt(temp)) + 1):
            if temp % (p * p) == 0:
                # p divides temp, check exponent
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                if exp == 1:
                    return False
            else:
                continue
        if temp > 1:
            # temp is prime, exponent is 1
            return False
        return True

    # Structural change: Instead of O(P^2) sum counting, use targeted candidate N generation
    # and O(P) verification per candidate.
    # 
    # Key insight: If we generate powerful numbers up to M, then for any N <= 2*M,
    # we can count representations in O(P) time using a set lookup.
    # 
    # To find good N candidates, we observe that N values which are sums of many 
    # small powerful numbers tend to have more representations. We generate candidate
    # N values as:
    #   1. Sums of pairs of small powerful numbers (dense region)
    #   2. Multiples of highly structured numbers
    #   3. Values near 2*sqrt(M) where the density of powerful numbers is highest
    #
    # Then we verify each candidate in O(P) time.
    
    # Generate powerful numbers up to a reasonable limit
    M = 50_000_000
    P = generate_powerful(M)
    P_set = set(P)
    P_list = P
    L = len(P)
    
    best_result = None
    best_count = 0
    
    # Generate candidate N values from dense regions
    # Use the first K powerful numbers to generate candidate sums
    K = min(2000, L)
    small_P = P_list[:K]
    
    # Generate candidate N values as sums of pairs from small_P
    # This gives us candidates in the dense region
    candidate_Ns = set()
    
    # Add sums of pairs from small powerful numbers
    for i in range(K):
        for j in range(i, K):
            candidate_Ns.add(small_P[i] + small_P[j])
    
    # Also add some structured candidates: multiples of small powerful numbers
    # and values near 2*M/3 (where both a and b can be up to M)
    for target in [M // 3, M // 2, 2 * M // 3, M - 1, M]:
        for mult in range(1, 5):
            candidate_Ns.add(target * mult)
    
    # Also add N = 2 * p for small powerful p (symmetric case)
    for p in small_P[:500]:
        candidate_Ns.add(2 * p)
    
    # Sort candidates and evaluate top ones
    # We want N <= 2*M so that both a and b can be in P
    max_N = 2 * M
    candidates = [n for n in candidate_Ns if 2 <= n <= max_N]
    
    # Sort by heuristic: prefer N values that are sums of more small powerful pairs
    # Count how many times each candidate appears as a sum of two small powerful numbers
    sum_counts = defaultdict(int)
    for i in range(K):
        for j in range(i, K):
            s = small_P[i] + small_P[j]
            if s <= max_N:
                sum_counts[s] += 1
    
    # Sort candidates by their sum count (descending)
    candidates.sort(key=lambda x: sum_counts.get(x, 0), reverse=True)
    
    # Evaluate top candidates
    max_candidates_to_try = 5000
    for N in candidates[:max_candidates_to_try]:
        if time.time() > deadline - 15:
            break
        
        pairs = count_pairs_for_N(N, P_set, P_list)
        c = len(pairs)
        
        if c > best_count:
            best_count = c
            best_result = {'N': N, 'pairs': pairs}
    
    # If we didn't find anything good, try a broader search
    if best_result is None or best_count < 5:
        # Try N values in a grid around dense regions
        step = max(1, M // 1000)
        for N in range(step, max_N + 1, step):
            if time.time() > deadline - 10:
                break
            pairs = count_pairs_for_N(N, P_set, P_list)
            c = len(pairs)
            if c > best_count:
                best_count = c
                best_result = {'N': N, 'pairs': pairs}
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a known valid result
    return {'N': 2, 'pairs': [[1, 1]]}
