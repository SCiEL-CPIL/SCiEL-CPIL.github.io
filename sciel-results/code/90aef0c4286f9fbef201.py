import math
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    # 110s deadline to be safe against 120s limit
    deadline = start_time + 110.0
    
    # Generate powerful numbers up to L
    # Powerful numbers are of the form a^2 * b^3 where b is square-free.
    # We choose L = 10^12. This is large enough to have many powerful numbers (~10^7)
    # but small enough that O(M^2) operations or sorting might be heavy in pure Python.
    # However, we can use NumPy for the sum counting or just optimize the generation.
    # Actually, generating 10^7 numbers and sorting is fine.
    # The bottleneck is counting pairs a+b=N.
    
    # Strategy:
    # 1. Generate powerful numbers P up to L.
    # 2. Sort P.
    # 3. Use a two-pointer technique or dictionary to find N with max representations?
    #    No, finding N with max representations is equivalent to finding the most frequent sum.
    #    With M ~ 10^7, O(M^2) is 10^14, too slow.
    
    # Alternative:
    # The number of representations r(N) = sum_{a in P, a <= N/2} 1_{N-a in P}.
    # This is a convolution.
    # If we use FFT, we can compute all r(N) for N up to 2L.
    # L = 10^12 -> Array size 2*10^12. Too large for memory (2*10^12 * 8 bytes = 16 TB).
    
    # So we must restrict the range of N we search for, or use a different method.
    # The current program searches for N in [2, 2*10^6].
    # If we increase L to 10^12, the "hot" N will likely be much larger.
    # But we can't check all N up to 2*10^12.
    
    # Let's look at the density.
    # Number of powerful numbers up to X is ~ c * sqrt(X).
    # For X = 10^12, M ~ 10^6 * c.
    # If we pick a random N ~ 10^12, the expected number of pairs is roughly
    # (density of P at N/2) * (length of interval) ... 
    # Actually, r(N) is roughly proportional to N^{-1/2} * log(N) ? No.
    # The number of solutions to a+b=N with a,b powerful.
    # a = u^2 v^3, b = w^2 z^3.
    # This is related to the number of divisors or similar arithmetic functions.
    
    # Let's try a different approach:
    # Generate P up to L = 10^10. M ~ 10^5.
    # O(M^2) = 10^10. Still too slow in Python.
    
    # We need to find the N with the most representations.
    # Is there a way to sample?
    # Or maybe the maximum is attained at a specific structure?
    
    # Let's look at Inspiration 1 (score 2.86). It likely uses a better search.
    # The current program (2.26) uses L=10^6.
    # If I use L=10^9, M ~ 30,000.
    # O(M^2) = 9*10^8. This might take 10-30 seconds in Python if optimized?
    # Or use NumPy for the counting?
    
    # Let's try L = 10^9.
    # Generate P.
    # Count sums.
    # To count sums efficiently, we can use a dictionary.
    # For each pair (i, j) with i <= j, sum = P[i] + P[j].
    # This is still O(M^2).
    
    # Can we do better?
    # If we only care about the MAX, can we prune?
    # Unlikely.
    
    # Let's stick to Python loops but optimize the inner loop or use a smaller L if M^2 is too big.
    # M(10^9) ~ 30,000. 30,000^2 / 2 = 4.5 * 10^8.
    # In Python, 10^8 simple operations takes ~10-50s.
    # This is risky.
    
    # Let's try L = 10^8. M ~ 10,000. 10^8 / 2 = 5 * 10^7. This is fast (< 2s).
    # L = 10^9 might be too slow.
    
    # Let's try to optimize the generation and counting.
    
    def is_square_free(n):
        if n < 2:
            return True
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    def generate_powerful(limit):
        powerful = set()
        max_b = int(limit ** (1.0/3.0)) + 2
        for b in range(1, max_b + 1):
            if not is_square_free(b):
                continue
            b3 = b ** 3
            if b3 > limit:
                break
            max_a = int(math.isqrt(limit // b3))
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val <= limit:
                    powerful.add(val)
        return sorted(powerful)

    # Try L = 10^8 first to ensure it's fast, then maybe increase if time permits?
    # No, we need a single strategy.
    # Let's try L = 2 * 10^8.
    L = 2 * 10**8
    p_list = generate_powerful(L)
    M = len(p_list)
    
    # Count sums
    # Using a dictionary to count frequencies of sums
    # To speed up, we can iterate i from 0 to M-1
    # and j from i to M-1
    # sum = p[i] + p[j]
    # counts[sum] += 1
    
    # Optimization:
    # If M is large, this is slow.
    # Let's check M for L=2*10^8.
    # sqrt(2*10^8) ~ 14000.
    # M is roughly 1.4 * 10^4.
    # Pairs ~ 10^8.
    # Python loop 10^8 iterations ~ 10-20 seconds.
    # This fits in 110s.
    
    counts = defaultdict(int)
    
    # Use local variables for speed
    p = p_list
    m = M
    c = counts
    
    for i in range(m):
        ai = p[i]
        # If ai > L, break? No, ai <= L.
        # We need ai + aj <= 2L.
        # Since p is sorted, aj is increasing.
        # We can break inner loop if ai + p[j] > 2L?
        # But we want to count ALL sums to find the max.
        # The max sum is 2L.
        # So we don't need to break based on sum > 2L, as all sums are <= 2L.
        
        # However, we can break if time is running out.
        if time.time() > deadline - 10:
            break
            
        for j in range(i, m):
            s = ai + p[j]
            c[s] += 1
            
    # Find the N with max count
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Retrieve pairs for best_N
    p_set = set(p_list)
    pairs = []
    # To be efficient, iterate a in p_list
    # a <= best_N / 2
    # b = best_N - a
    # if b in p_set: pairs.append([a, b])
    
    # We can break early if a > best_N / 2
    half_N = best_N // 2
    for a in p_list:
        if a > half_N:
            break
        b = best_N - a
        if b in p_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}

if __name__ == '__main__':
    res = construct()
    print(res['N'], len(res['pairs']))
