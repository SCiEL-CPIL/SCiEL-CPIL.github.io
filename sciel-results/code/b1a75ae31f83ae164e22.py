import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit

    # Generate all powerful numbers up to a limit
    # A powerful number n has all prime factors with exponent >= 2
    # So n = a^2 * b^3 where b is square-free
    
    # We want N up to 10^18, but we need both a and b to be powerful.
    # If a and b are both <= M, then N <= 2*M.
    # Number of powerfuls up to X is ~ c * X^(2/5).
    # For M = 10^9, X^(2/5) = (10^9)^0.4 = 10^3.6 ≈ 4000 powerfuls.
    # O(K^2) with K=4000 is ~8M operations, feasible.
    # For M = 10^10, X^(2/5) = (10^10)^0.4 = 10^4 ≈ 10000 powerfuls.
    # O(K^2) with K=10000 is ~50M operations, might be tight.
    # Let's try M = 10^9 first, see how long it takes.
    
    # Actually, let's think about this differently.
    # We want to maximize the number of pairs (a, b) with a + b = N, both powerful.
    # The expected number of representations for a random N is ~ N^(1/3) / (some constant).
    # But we can do better by choosing N strategically.
    
    # Key insight: If we choose N such that many powerful numbers a satisfy N - a being powerful,
    # we get many representations.
    
    # Strategy: Generate powerful numbers up to M, then for each pair (i, j) with i <= j,
    # count how many times N = p[i] + p[j] appears. Track the best N.
    
    # To go up to N = 10^18, we'd need powerfuls up to 10^18, which is too many.
    # But we can still find good N values with smaller M.
    
    # Let's use M = 10^9, which gives us powerfuls up to 10^9, and N up to 2*10^9.
    # That's well within 10^18.
    
    M = 1_000_000_000  # 10^9
    
    powerful = set()
    
    # For each square-free b, add b^3 * a^2 for all a such that b^3 * a^2 <= M
    # b^3 <= M => b <= M^(1/3) ≈ 1000 for M = 10^9
    
    max_b = int(M ** (1.0 / 3.0)) + 2
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        is_sf = True
        temp = b
        d = 2
        while d * d <= temp:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            if cnt >= 2:
                is_sf = False
                break
            d += 1 if d == 2 else 2
        if not is_sf:
            continue
        
        b_cubed = b ** 3
        if b_cubed > M:
            break
        
        # a^2 <= M / b_cubed => a <= sqrt(M / b_cubed)
        max_a = isqrt(M // b_cubed)
        for a in range(1, max_a + 1):
            powerful.add(a * a * b_cubed)
    
    powerful_list = sorted(powerful)
    powerful_set = powerful
    
    if time.time() > deadline:
        # Fallback: use a smaller set
        M2 = 100_000_000
        powerful2 = set()
        max_b2 = int(M2 ** (1.0 / 3.0)) + 2
        for b in range(1, max_b2 + 1):
            is_sf = True
            temp = b
            d = 2
            while d * d <= temp:
                cnt = 0
                while temp % d == 0:
                    temp //= d
                    cnt += 1
                if cnt >= 2:
                    is_sf = False
                    break
                d += 1 if d == 2 else 2
            if not is_sf:
                continue
            b_cubed = b ** 3
            if b_cubed > M2:
                break
            max_a = isqrt(M2 // b_cubed)
            for a in range(1, max_a + 1):
                powerful2.add(a * a * b_cubed)
        powerful_list = sorted(powerful2)
        powerful_set = powerful2
    
    K = len(powerful_list)
    
    # Count occurrences of each sum
    counts = defaultdict(int)
    
    for i in range(K):
        a = powerful_list[i]
        for j in range(i, K):
            b = powerful_list[j]
            N = a + b
            if N > 2 * M:
                break
            counts[N] += 1
    
    # Find the best N
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Rebuild pairs for best_N
    pairs = []
    for a in powerful_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerful_set and a <= b:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
