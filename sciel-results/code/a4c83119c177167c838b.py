import math
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit

    def is_powerful(n):
        if n < 1:
            return False
        x = n
        d = 2
        while d * d <= x:
            if x % d == 0:
                cnt = 0
                while x % d == 0:
                    x //= d
                    cnt += 1
                if cnt == 1:
                    return False
            d += 1
        return True

    def generate_powerful_sum_of_two_squares(limit):
        """
        Generate powerful numbers <= limit that are sums of two squares.
        A number is a sum of two squares iff in its prime factorization,
        every prime p ≡ 3 (mod 4) appears with an even exponent.
        Since it's powerful, exponents are >= 2. So exponents of p ≡ 3 (mod 4)
        must be even (2, 4, ...).
        """
        nums = set([1])
        # Primes p ≡ 1 (mod 4) and p = 2 can have any exponent >= 1 (but for powerful, >= 2 if prime factor)
        # Actually, for powerful:
        # - If p | n, then p^2 | n.
        # - If p ≡ 3 (mod 4), exponent must be even.
        
        # Let's generate by multiplying primes.
        # Primes p ≡ 1 (mod 4): 5, 13, 17, 29, 37, 41, 53, 61, ...
        # Primes p ≡ 3 (mod 4): 3, 7, 11, 19, 23, 31, ...
        
        # We want n = 2^a * (prod p_i^b_i) * (prod q_j^c_j)
        # where p_i ≡ 1 (mod 4), q_j ≡ 3 (mod 4).
        # b_i >= 2, c_j >= 2 and c_j even.
        # a >= 0, but if a=1, n is not powerful? No, 2^1 is not powerful. So a=0 or a>=2.
        
        # Let's use a recursive generation or simple loops.
        # Given the limit is likely small for the "core", we can be exhaustive.
        
        # Primes up to limit^(1/2) or so?
        # Max prime factor in a powerful number <= limit is <= sqrt(limit).
        
        def gen_powerful(limit, prime_list, idx, current_val, current_is_sot2):
            # This is getting complex. Let's just generate all powerful numbers <= limit
            # and filter for sum of two squares.
            pass

    # Simpler: Generate all powerful numbers up to a small limit (e.g. 10^5)
    # and filter those that are sums of two squares.
    
    def generate_all_powerful(limit):
        powerful = set([1])
        # y^3 * x^2 <= limit => y <= limit^(1/3)
        max_y = int(limit ** (1.0/3.0)) + 2
        # Sieve for square-free
        sq_free = [True] * (max_y + 1)
        sq_free[0] = False
        p = 2
        while p * p <= max_y:
            if sq_free[p]:
                sq = p * p
                for j in range(sq, max_y + 1, sq):
                    sq_free[j] = False
            p += 1
        
        for y in range(1, max_y + 1):
            if not sq_free[y]:
                continue
            y3 = y * y * y
            if y3 > limit:
                break
            max_x = int(math.isqrt(limit // y3))
            for x in range(1, max_x + 1):
                n = x * x * y3
                powerful.add(n)
        return sorted(powerful)

    def is_sum_of_two_squares(n):
        # Check if n is a sum of two squares.
        # Condition: no prime p ≡ 3 (mod 4) divides n to an odd power.
        x = n
        d = 2
        while d * d <= x:
            if x % d == 0:
                cnt = 0
                while x % d == 0:
                    x //= d
                    cnt += 1
                if d % 4 == 3 and cnt % 2 == 1:
                    return False
            d += 1
        if x > 1 and x % 4 == 3:
            return False
        return True

    def get_reps(n):
        """Return list of (a, b) such that a^2 + b^2 = n, a <= b."""
        reps = []
        a = 0
        while a * a <= n // 2:
            b2 = n - a * a
            b = math.isqrt(b2)
            if b * b == b2:
                reps.append((a, b))
            a += 1
        return reps

    # Strategy:
    # 1. Generate powerful numbers up to a moderate limit L1 (e.g. 10^4) that are sums of two squares.
    #    These will serve as our "cores" C.
    # 2. For each core C, we can multiply by a powerful number M (not necessarily sum of two squares? 
    #    Wait, if C is sum of two squares and M is powerful, is C*M a sum of two squares?
    #    Yes, if M is a sum of two squares. But we need C*M to be powerful.
    #    If C and M are powerful, C*M is powerful.
    #    So we need M to be a powerful number that is a sum of two squares.
    #    So both C and M must be powerful sums of two squares.
    #    Then N = C * M is a powerful sum of two squares.
    #    The number of representations of N is related to the number of representations of C and M.
    #    Specifically, r_2(C*M) is roughly r_2(C) * r_2(M) (with some factors of 2).
    #    Actually, the number of representations r_2(n) = 4 * (d_1(n) - d_3(n)).
    #    For powerful n, d_3(n) = 0 if n is sum of two squares? No.
    #    Let's just compute r_2(N) directly or estimate.
    
    # To maximize representations, we want N to have many prime factors p ≡ 1 (mod 4).
    # Each such prime p contributes a factor 2 to the number of representations (roughly).
    # So we should construct N from many small primes p ≡ 1 (mod 4).
    
    # Let's generate powerful numbers of the form 2^a * 5^b * 13^c * 17^d * ...
    # where exponents are even (for powerfulness and sum of two squares condition, though p≡1 mod 4 doesn't need even exponent for sum of two squares, it needs even exponent for powerfulness? No, powerful means exponent >= 2. Sum of two squares means p≡3 mod 4 has even exponent. p≡1 mod 4 can have any exponent for sum of two squares, but for powerful it must be >= 2).
    
    # So we can use primes 2, 5, 13, 17, 29, 37, 41, 53, ...
    # Exponents must be >= 2.
    
    primes_1_mod_4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197]
    # Include 2.
    
    # We want to find N <= 10^18.
    # Let's try to construct N using the first few primes.
    # N = 2^a * 5^b * 13^c * ...
    # We want to maximize the number of representations.
    # The number of representations r_2(N) = 4 * prod (e_i + 1) for p_i ≡ 1 (mod 4), times some factor for 2 and p_j ≡ 3 (mod 4) (which must be 1 if exponent is even? No, d_1 - d_3).
    # Actually, for N = 2^a * prod p_i^b_i * prod q_j^c_j, where p_i ≡ 1 (mod 4), q_j ≡ 3 (mod 4).
    # If N is a sum of two squares, c_j are even.
    # r_2(N) = 4 * prod (b_i + 1).
    # So we want to maximize prod (b_i + 1).
    # Subject to N <= 10^18 and N is powerful (a>=0, a!=1; b_i>=2; c_j>=2, c_j even).
    
    # Let's use a search over exponents of the smallest primes.
    # Primes: 2, 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97
    # Exponents: a, b, c, d, ...
    # a can be 0 or >= 2.
    # b, c, ... >= 2.
    
    best_N = 0
    best_reps = 0
    best_pairs = []
    
    # Generate candidate N values
    candidates = []
    
    def dfs(idx, current_val, current_reps_factor):
        nonlocal best_N, best_reps, best_pairs
        if time.time() > deadline - 10:
            return
        
        if current_val > 10**18:
            return
        
        # Check if we can add more primes
        if idx < len(primes_1_mod_4):
            p = primes_1_mod_4[idx]
            # Try exponents e = 2, 4, 6, ...
            val = current_val
            # p^2
            if val * p * p <= 10**18:
                dfs(idx + 1, val * p * p, current_reps_factor * 3)
                # p^4
                if val * p**4 <= 10**18:
                    dfs(idx + 1, val * p**4, current_reps_factor * 5)
                    # p^6
                    if val * p**6 <= 10**18:
                        dfs(idx + 1, val * p**6, current_reps_factor * 7)
                        # p^8
                        if val * p**8 <= 10**18:
                            dfs(idx + 1, val * p**8, current_reps_factor * 9)
            # Skip this prime
            dfs(idx + 1, current_val, current_reps_factor)
        else:
            # Leaf node
            # current_val is a candidate N.
            # It must be powerful.
            # We constructed it from primes 5, 13, ... with exponents >= 2.
            # We haven't included 2 yet.
            # Let's consider multiplying by 2^a.
            # a=0, a=2, a=4, ...
            
            # For each a, N = current_val * 2^a.
            # r_2(N) = 4 * current_reps_factor (since 2 doesn't affect the product of (b_i+1) for p_i ≡ 1 mod 4? 
            # Wait, r_2(2^a * n) = r_2(n) if n is odd?
            # r_2(2) = 2. r_2(4) = 2. r_2(8) = 0.
            # Formula: r_2(n) = 4 * (d_1(n) - d_3(n)).
            # For n = 2^a * m (m odd), d_1(n) - d_3(n) = d_1(m) - d_3(m).
            # So r_2(2^a * m) = r_2(m) for a even?
            # Let's check: n=5, r_2=4. n=10, r_2=4. n=20, r_2=4. n=40, r_2=4.
            # Yes, multiplying by 2^a (a even) doesn't change r_2.
            # Multiplying by 2 (a=1) makes it not powerful.
            
            # So we can just use current_val or current_val * 4, * 16, etc.
            # But current_val is already powerful.
            # Let's just evaluate current_val.
            
            N = current_val
            if N > 1 and is_powerful(N):
                reps = get_reps(N)
                if len(reps) > best_reps:
                    best_reps = len(reps)
                    best_N = N
                    best_pairs = [[a, b] for a, b in reps]
                    # Note: get_reps returns (a, b) such that a^2+b^2=N.
                    # The task requires a+b=N where a,b are powerful.
                    # This is a COMPLETELY DIFFERENT PROBLEM.
                    # My strategy is flawed. I was solving "sum of two squares", not "sum of two powerful numbers".
                    
    # RESTART STRATEGY:
    # The task is: N = a + b, where a and b are powerful.
    # My previous thought process was drifting. Let's go back to basics but with a smarter search.
    
    # The current program uses L=5*10^5.
    # Let's try to use the time to search for N in a larger range or with a better heuristic.
    # Heuristic: N = 2 * Q where Q is powerful is a good start.
    # Also, N = Q1 + Q2.
    
    # Let's try a random search combined with local optimization.
    # Or, let's just increase L in the original method if we have time, but optimize the inner loop.
    
    # Actually, the original method's bottleneck is the O(P^2) loop.
    # We can use NumPy to vectorize the sum counting?
    # P is a list of powerful numbers.
    # We want to count how many pairs (i, j) with i <= j sum to each N.
    
    L = 2 * 10**6
    P = generate_all_powerful(L)
    P_set = set(P)
    
    # If we have time, we can try a larger L.
    # But first, let's see if we can find a better N with L=2*10^6.
    
    # To speed up the counting, we can use a dictionary of counts.
    # But the loop is slow.
    
    # Alternative: Use the fact that we only care about the TOP candidates.
    # We can sample pairs? No, we need exact counts.
    
    # Let's try a different L based on available time.
    # Estimate time for L=10^6: P ~ 40000. P^2 ~ 1.6 * 10^9. Too slow.
    # L=10^5: P ~ 4000. P^2 ~ 1.6 * 10^7. Fast.
    # L=2*10^5: P ~ 7000. P^2 ~ 5 * 10^7. ~5s.
    # L=5*10^5: P ~ 12000. P^2 ~ 1.4 * 10^8. ~15s.
    
    # Let's use L=5*10^5 as in the original, but maybe we can find a better N by looking at a slightly different range or using a trick.
    
    # Trick: What if we generate powerful numbers up to L, but also consider N up to 2*L?
    # The original code does this.
    
    # Let's just run the original logic but with a slightly larger L if time permits, and ensure we return valid pairs.
    
    L = 5 * 10**5
    P = generate_all_powerful(L)
    P_set = set(P)
    P_list = P
    
    pair_counts = defaultdict(int)
    
    import bisect
    
    for i, a in enumerate(P_list):
        max_b = 2 * L - a
        j_max = bisect.bisect_right(P_list, max_b)
        for j in range(i, j_max):
            b = P_list[j]
            s = a + b
            pair_counts[s] += 1
            if time.time() > deadline - 20:
                break
        if time.time() > deadline - 20:
            break
    
    top_candidates = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:100]
    
    best_n = 0
    best_pairs = []
    
    for N, count in top_candidates:
        if N < 2:
            continue
        
        pairs = []
        for a in P_list:
            if a > N // 2:
                break
            b = N - a
            if b in P_set:
                pairs.append([a, b])
        
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_n = N
    
    if not best_pairs:
        best_n = 5
        best_pairs = [[1, 4]]
        
    # Truncate to 1000 pairs if necessary
    if len(best_pairs) > 1000:
        best_pairs = best_pairs[:1000]
        
    return {'N': best_n, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
