import math
from collections import defaultdict

def construct():
    L = 10**14
    
    # 1. Generate powerful numbers up to L
    powerful = set()
    max_b = int(L ** (1/3)) + 2
    
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = int(math.isqrt(L // b3))
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful.add(n)
            else:
                break
                
    powerful_list = sorted(powerful)
    powerful_set = powerful
    
    # 2. Select dense set of small powerful numbers
    M = 15000
    if len(powerful_list) < M:
        M = len(powerful_list)
    P_small = powerful_list[:M]
    
    # 3. Count sums of pairs in P_small
    sum_counts = defaultdict(int)
    for i in range(M):
        pi = P_small[i]
        for j in range(i, M):
            pj = P_small[j]
            N = pi + pj
            sum_counts[N] += 1
            
    # 4. Get top candidates
    sorted_candidates = sorted(sum_counts.items(), key=lambda x: x[1], reverse=True)
    
    best_N = 0
    best_pairs = []
    best_count = 0
    
    # Check top 20 candidates to be safe
    for N, count in sorted_candidates[:20]:
        pairs = []
        # Find all pairs a, b such that a+b=N, a<=b, a,b powerful
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                pairs.append([a, b])
                
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs
            
    return {'N': best_N, 'pairs': best_pairs}
