import math
import time
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def time_left():
        return deadline - time.time()

    # Precompute powerful numbers up to a reasonable limit
    # A powerful number has all prime factors appearing with exponent >= 2
    # So powerful numbers are of the form prod(p_i^e_i) where e_i >= 2
    # Small powerful numbers: 1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 72, 81, 100, 108, 121, 125, 128, 144, 169, 196, 200, 216, 225, 243, 256, 288, 289, 324, 343, 361, 400, 432, 441, 484, 500, 512, 529, 576, 625, 648, 675, 676, 729, 784, 800, 841, 864, 900, 961, 968, 1000, ...

    def generate_powerful_numbers(limit):
        """Generate all powerful numbers up to limit."""
        powerful = set([1])
        # For each prime p, p^2, p^3, ... contribute
        # We need to enumerate products of prime powers
        # Use a recursive approach over small primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        
        def gen(idx, current):
            if current > limit:
                return
            if idx >= len(primes):
                if current > 1:
                    powerful.add(current)
                return
            p = primes[idx]
            # Try exponent 0 (skip this prime)
            gen(idx + 1, current)
            # Try exponents >= 2
            val = p * p
            while val <= limit:
                gen(idx + 1, current * val)
                val *= p
                if val > limit:
                    break
        
        gen(0, 1)
        powerful.add(1)
        return sorted(list(powerful))

    # We need powerful numbers up to 10^18, but that's too many to enumerate fully.
    # Instead, use the structure: N = Q * M where Q is a square and M is sum of two squares.
    # a = Q * x^2, b = Q * y^2, so a and b are powerful if Q * x^2 and Q * y^2 are powerful.
    # Q * x^2 is powerful iff for every prime p, the exponent in Q * x^2 is 0 or >= 2.
    # If Q = prod(q_i^2) (a perfect square), then Q * x^2 = prod(q_i^2) * x^2.
    # For this to be powerful, every prime factor of x must appear with even exponent in x (so x^2 has even exponents),
    # and primes in Q already have even exponents (2 * something).
    # Actually, if Q is a perfect square, then Q * x^2 is always a perfect square times x^2... 
    # Wait: Q * x^2 where Q is a perfect square. Let Q = s^2. Then Q * x^2 = (s*x)^2, which is a perfect square.
    # A perfect square is powerful iff every prime factor appears with even exponent, which is always true for a perfect square.
    # BUT: a perfect square is powerful only if every prime factor appears with exponent >= 2. 
    # A perfect square like 9 = 3^2 is powerful. 4 = 2^2 is powerful. 16 = 2^4 is powerful.
    # Actually, EVERY perfect square is powerful! Because if n = prod(p_i^{2e_i}), then each exponent 2e_i >= 2 (since e_i >= 1 for primes that appear).
    # Wait, what about 1? 1 is powerful by definition.
    # So if Q is a perfect square, then Q * x^2 = (sqrt(Q) * x)^2 is a perfect square, hence powerful.
    # Similarly Q * y^2 is powerful.
    # So the strategy is: pick Q = s^2, pick M = x^2 + y^2, then N = Q * M, and pairs are (Q*x^2, Q*y^2).
    # The number of pairs equals the number of representations of M as sum of two squares (with x <= y).

    # The number of representations of M as x^2 + y^2 (ordered, including signs and zero) is 4 * (d1(M) - d3(M)) where d1, d3 are divisors congruent to 1, 3 mod 4.
    # For unordered non-negative pairs with x <= y: it's roughly (d1 - d3) / 2, adjusted for cases where x=0 or x=y.
    # More precisely, if M > 0, the number of representations as sum of two squares (with x, y >= 0, x <= y) is:
    # Let r(M) = number of (x,y) with x^2+y^2=M, x,y integers (including negative and zero).
    # r(M) = 4 * prod_{p_i ≡ 1 (mod 4)} (e_i + 1) if all primes ≡ 3 (mod 4) have even exponent, else 0.
    # The number of unordered non-negative pairs (x,y) with x <= y is r(M)/8 if x != y and x,y != 0, with adjustments.
    # For our purposes, we want to maximize the number of distinct (a,b) pairs, which equals the number of distinct (x,y) with x <= y, x^2+y^2=M.

    # Strategy: Search over Q = s^2 for small s, and for each Q, find M <= 10^18/Q that maximizes representations.
    # M should be composed of primes ≡ 1 (mod 4) to have representations.
    # The number of representations (unordered, x<=y, x,y>=0) for M = prod(p_i^{e_i}) where all p_i ≡ 1 (mod 4) is:
    # prod(e_i + 1) / 2, roughly. More precisely, if M has k distinct prime factors ≡ 1 mod 4 with exponents e_1,...,e_k,
    # and no prime factors ≡ 3 mod 4 (or even exponents for those), then the number of representations as x^2+y^2 with x,y >= 0, x <= y is:
    # (prod(e_i+1) + 1) / 2 if M is a square, else prod(e_i+1) / 2.

    # Let's use a different approach: enumerate candidate M values that are products of small 1 mod 4 primes,
    # compute the number of representations, and pick the best.

    # Precompute small primes ≡ 1 mod 4
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461]

    def count_representations(M):
        """Count the number of representations of M as x^2 + y^2 with x <= y, x,y >= 0.
        Returns the count, or 0 if M has prime factors ≡ 3 mod 4 with odd exponent."""
        if M == 1:
            return 1  # 0^2 + 1^2
        m = M
        count = 1
        for p in primes_1mod4:
            if p * p > m:
                break
            if m % p == 0:
                e = 0
                while m % p == 0:
                    m //= p
                    e += 1
                count *= (e + 1)
        # Check remaining m
        if m > 1:
            # m is prime (or 1)
            if m % 4 == 3:
                # Odd exponent (1) of a 3 mod 4 prime -> no representations
                return 0
            else:
                # m ≡ 1 mod 4, exponent 1
                count *= 2
        # Now check if we removed all 3 mod 4 primes with even exponent
        # We need to verify that all 3 mod 4 primes in M have even exponent
        # Let's re-check by factoring M properly
        m2 = M
        for p in [3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83, 103, 107, 127, 131, 139, 151, 163, 167, 179, 191, 199, 211, 223, 227, 239, 251, 263, 269, 271, 283, 299]:
            if p * p > m2:
                break
            if m2 % p == 0:
                e = 0
                while m2 % p == 0:
                    m2 //= p
                    e += 1
                if e % 2 == 1:
                    return 0
        if m2 > 1 and m2 % 4 == 3:
            return 0
        # Number of unordered non-negative pairs
        # Total representations (ordered, signed) = 4 * count (if M > 0)
        # Unordered non-negative with x <= y: 
        # If M is not a square: count / 2
        # If M is a square: (count + 1) / 2
        if math.isqrt(M) ** 2 == M:
            return (count + 1) // 2
        else:
            return count // 2

    def find_best_M(M_max, max_primes=20):
        """Find M <= M_max that maximizes the number of sum-of-two-squares representations.
        M should be a product of primes ≡ 1 mod 4 (and possibly squares of primes ≡ 3 mod 4)."""
        best_M = 1
        best_count = 1  # M=1 gives 1 representation (0,1)
        
        # Try M = product of first k primes ≡ 1 mod 4, with various exponents
        # Use a greedy approach: start with M=1, multiply by smallest available 1 mod 4 prime
        # that keeps M <= M_max, and track the best.
        
        # Actually, let's enumerate combinations of small 1 mod 4 primes
        # For efficiency, limit to first ~15 primes and small exponents
        primes_to_use = primes_1mod4[:max_primes]
        
        # DFS over exponent vectors
        best = [1, 1]  # [M, count]
        
        def dfs(idx, current_M, current_count):
            if time_left() < 3:
                return
            if current_count > best[1]:
                best[1] = current_count
                best[0] = current_M
            if idx >= len(primes_to_use):
                return
            p = primes_to_use[idx]
            # Option: don't use this prime
            dfs(idx + 1, current_M, current_count)
            # Option: use this prime with exponent e >= 1
            val = p
            e = 1
            while current_M * val <= M_max:
                dfs(idx + 1, current_M * val, current_count * (e + 1))
                val *= p
                e += 1
                if val > M_max:
                    break
        
        dfs(0, 1, 1)
        return best[0], best[1]

    def find_pairs_for_M(M, Q):
        """Find all (x, y) with x <= y, x >= 0, x^2 + y^2 = M. Return [(Q*x^2, Q*y^2), ...]"""
        pairs = []
        limit_x = int(math.isqrt(M // 2)) + 1
        if limit_x > 5 * 10**7:
            # Too large for direct enumeration; use a different method
            # For now, skip if too large
            return []
        for x in range(0, limit_x + 1):
            x2 = x * x
            y2 = M - x2
            if y2 < 0:
                break
            y = math.isqrt(y2)
            if y * y == y2 and x <= y:
                a = Q * x2
                b = Q * y2
                if a + b == Q * M:
                    pairs.append([a, b])
        return pairs

    best_N = 0
    best_pairs = []

    # Search over different Q values (perfect squares)
    # Q = s^2 where s is a product of small primes
    primes_for_Q = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    # Try different combinations of primes for s
    # s can be 1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, ...
    # Q = s^2
    candidate_s_values = []
    # Generate s values: products of subsets of small primes
    def gen_s(idx, current):
        if idx >= len(primes_for_Q):
            candidate_s_values.append(current)
            return
        p = primes_for_Q[idx]
        # Don't use p
        gen_s(idx + 1, current)
        # Use p
        if current * p <= 10**9:  # s <= 10^9, Q <= 10^18
            gen_s(idx + 1, current * p)
    
    gen_s(0, 1)
    candidate_s_values = sorted(set(candidate_s_values))
    
    # Also include some larger s values
    for s in range(2, 200):
        candidate_s_values.append(s)
    candidate_s_values = sorted(set(candidate_s_values))
    
    # For each Q, find the best M
    # Sort by Q descending so we try larger Q first (which allow smaller M but more representations potentially)
    # Actually, smaller Q allows larger M, which can have more prime factors.
    # Let's try a range of Q values.
    
    for s in candidate_s_values:
        if time_left() < 10:
            break
        Q = s * s
        if Q > 10**18:
            continue
        M_max = 10**18 // Q
        if M_max < 2:
            continue
        
        # Find best M for this Q
        best_M, best_count = find_best_M(M_max, max_primes=18)
        
        if best_count < 2:
            continue
        
        M = best_M
        N = Q * M
        if N < 2 or N > 10**18:
            continue
        
        # Find actual pairs
        pairs = find_pairs_for_M(M, Q)
        
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N
    
    # Fallback
    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]

    return {'N': best_N, 'pairs': best_pairs}
