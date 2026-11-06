import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def time_left():
        return deadline - time.time()

    # Generate all powerful numbers up to LIMIT
    LIMIT = 10**18

    def gen_powerful(limit):
        """Generate all powerful numbers up to limit."""
        pw = {1}
        
        # For each square-free part q, find all s such that q*s^2 <= limit
        # q must be square-free
        # s can be any integer >= 1
        
        # First, generate square-free numbers up to sqrt(limit) ~ 10^9
        # That's too many. Instead, iterate over possible s values.
        # For a given s, q*s^2 <= limit means q <= limit/s^2
        # q must be square-free.
        
        # Alternative: iterate over s from 1 to sqrt(limit)
        # For each s, the maximum q is limit // (s*s)
        # We need square-free q up to that value.
        
        # Better approach: 
        # A powerful number is of form q * s^2 where q is square-free.
        # Equivalently, for each prime p, the exponent in the factorization
        # is either 0 or >= 2.
        
        # Let's use the characterization: n is powerful iff for every prime p | n, p^2 | n.
        # So n = prod p_i^{e_i} where each e_i is 0 or >= 2.
        
        # Generate by iterating over "square parts" and "square-free parts"
        # n = s^2 * q where q is square-free
        # s^2 * q <= limit
        # q <= limit / s^2
        
        # For s from 1 to isqrt(limit):
        #   max_q = limit // (s*s)
        #   Find all square-free q <= max_q
        #   Add s^2 * q to set
        
        # But finding all square-free q up to max_q for large max_q is expensive.
        
        # Alternative: DFS over primes with exponent 0 or >= 2
        # This is what the original approach does but more carefully.
        
        # Let's use a different strategy:
        # Generate powerful numbers by their "core" structure.
        # A powerful number n has the form prod p_i^{e_i} where e_i >= 2 for all p_i | n.
        # Write n = m^2 * r where r is square-free (the "square-free part").
        # Then m^2 * r <= limit.
        
        # Iterate over m from 1 to isqrt(limit):
        #   For each m, we need square-free r <= limit // (m*m)
        #   But we can't enumerate all square-free r easily for large limits.
        
        # Better: use the fact that if n is powerful, then n = a^2 * b where b is square-free.
        # And a^2 * b <= 10^18.
        # a <= 10^9.
        # b <= 10^18 / a^2.
        
        # For small a, b can be large. For large a, b is small.
        
        # Let's split:
        # Case 1: a <= A (some threshold)
        #   For each a, enumerate square-free b <= limit // (a*a)
        # Case 2: a > A
        #   Then b <= limit // (A*A) which is small
        #   For each square-free b <= limit // (A*A), enumerate a from A+1 to isqrt(limit // b)
        
        # The number of square-free numbers up to X is ~ 6X/pi^2.
        # For X = 10^9, that's ~6*10^8 which is too many.
        
        # Let me think differently. The original approach with Q*M where Q is a square
        # and M has primes 1 mod 4 is actually quite good for finding many representations.
        # Let me stick with that but optimize the DFS and enumeration.
        
        # Actually, let me try a hybrid: generate powerful numbers up to a smaller limit
        # and use meet-in-the-middle on those.
        
        # For powerful numbers up to 10^12, we can generate them.
        # Then for candidate N, count pairs (a,b) from our list with a+b=N.
        
        # But N can be up to 10^18, so a and b can be up to 10^18.
        # We need powerful numbers up to 10^18.
        
        # Let me just improve the original approach with better time management.
        return None  # placeholder

    # Keep the original structural approach but optimize
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 713, 733, 757, 761, 769, 773, 797, 809, 811, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 991, 997]

    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

    best_N = 0
    best_pairs = []

    k_order = [4, 5, 6, 3, 7, 8, 2, 1, 9, 10, 11, 12]

    for k in k_order:
        if time_left() < 5:
            break
            
        S = 1
        for i in range(k):
            S *= primes_all[i]
        Q = S * S
        
        M_max = 10**18 // Q
        if M_max < 2:
            continue
        
        best_M = 1
        best_score = 0
        
        search_primes = p_1mod4[:40]
        
        def dfs(idx, current_M, current_score, prev_exp):
            nonlocal best_M, best_score
            
            if time_left() < 2:
                return
                
            if current_score > best_score:
                best_score = current_score
                best_M = current_M
                
            if idx == len(search_primes):
                return
            
            p = search_primes[idx]
            
            val = 1
            e = 0
            while current_M * val <= M_max and e <= prev_exp:
                next_M = current_M * val
                next_score = current_score * (e + 1)
                dfs(idx + 1, next_M, next_score, e)
                val *= p
                e += 1
                if val > M_max // max(current_M, 1):
                    break

        dfs(0, 1, 1, 200)
        
        M = best_M
        N = Q * M
        
        if N < 2 or N > 10**18:
            continue
        
        # Enumerate pairs using efficient approach
        limit_x = int(math.isqrt(M // 2)) + 1
        max_iters = min(limit_x, 3 * 10**7)
        
        pairs = []
        M_val = M
        Q_val = Q
        isqrt = math.isqrt
        
        for x in range(1, max_iters):
            x2 = x * x
            y2 = M_val - x2
            y = isqrt(y2)
            if y * y == y2 and x <= y:
                a = Q_val * x2
                b = Q_val * y2
                pairs.append([a, b])
            
            if x % 500000 == 0 and time_left() < 5:
                break

        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N

    # Fallback
    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
