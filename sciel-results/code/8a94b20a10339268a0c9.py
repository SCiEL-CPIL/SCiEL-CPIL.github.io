import time
import math
from bisect import bisect_left, bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer for safety
    
    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        powerful = set()
        # x = u^2 * v^3 where v is square-free
        max_v = int(round(limit ** (1/3))) + 2
        for v in range(1, max_v + 1):
            if not is_square_free(v):
                continue
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        return sorted(powerful)

    # Generate powerful numbers up to a reasonable limit
    # We need enough to find good N, but not so many that verification is slow
    M = 20000000  # 2 * 10^7
    
    P = generate_powerful(M)
    P_set = set(P)
    L = len(P)
    
    if time.time() > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Strategy: Generate candidate N values from dense subsets of small powerful numbers
    # Then verify each candidate by counting representations
    
    # Small powerful numbers are dense, so sums of many small ones create candidates
    # with potentially many representations
    
    # First, let's do a quick partial pair-sum to find promising N ranges
    # Use a limited window to find high-density sum regions
    
    # Take the first K powerful numbers for candidate generation
    K = min(L, 2000)
    small_P = P[:K]
    
    # Generate candidate N values as sums of pairs from small_P
    # Focus on sums in a range where density is high
    candidate_counts = {}
    for i in range(K):
        ai = small_P[i]
        for j in range(i, K):
            s = ai + small_P[j]
            candidate_counts[s] = candidate_counts.get(s, 0) + 1
    
    # Sort candidates by count to get top candidates
    if not candidate_counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    sorted_candidates = sorted(candidate_counts.items(), key=lambda x: -x[1])
    
    # Also add some larger N candidates from sums involving larger powerful numbers
    # N around 10^7 to 10^8 tend to have more representations
    # Add candidates from sums of one small and one medium powerful number
    K2 = min(L, 5000)
    for i in range(min(100, K2)):
        ai = P[i]
        for j in range(K2 // 2, K2):
            s = ai + P[j]
            if 5000000 <= s <= 50000000:
                candidate_counts[s] = candidate_counts.get(s, 0) + 1
    
    sorted_candidates = sorted(candidate_counts.items(), key=lambda x: -x[1])
    
    # Verify top candidates by counting actual representations
    best_N = 0
    best_pairs = []
    best_count = 0
    
    # Check top 50 candidates
    num_to_check = min(50, len(sorted_candidates))
    
    for N_candidate, _ in sorted_candidates[:num_to_check]:
        if time.time() > deadline - 5:
            break
        
        # Count representations of N_candidate as sum of two powerful numbers
        # a + b = N, a <= b, both powerful
        # Iterate over a in P where a <= N/2, check if N-a is in P_set
        pairs = []
        # Binary search for the range of a values: a <= N/2 and N-a <= M
        # a <= min(N//2, M)
        a_max = min(N_candidate // 2, M)
        
        # Find index range in P
        idx_start = bisect_left(P, 1)
        idx_end = bisect_right(P, a_max)
        
        for idx in range(idx_start, idx_end):
            a = P[idx]
            b = N_candidate - a
            if b < a:
                break
            if b in P_set:
                pairs.append([a, b])
        
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N_candidate
            best_pairs = pairs
    
    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': best_pairs}
