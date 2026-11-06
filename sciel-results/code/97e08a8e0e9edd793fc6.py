import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

    def time_left():
        return deadline - time.time()

    # Primes 1 mod 4 for building M
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 713, 733, 757, 761, 769, 773, 797, 809, 811, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 991, 997]
    
    # All primes for building Q (primorial)
    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

    def find_sqr_rep(p):
        """Find a+b^2 = p for prime p ≡ 1 (mod 4). Returns (a, b) with a²+b²=p."""
        # Find a quadratic non-residue mod p
        for g in range(2, p):
            if pow(g, (p - 1) // 2, p) == p - 1:
                break
        # Compute sqrt(-1) mod p using Tonelli-Shanks style
        # We need x such that x² ≡ -1 (mod p)
        # Since p ≡ 1 mod 4, -1 is a quadratic residue
        # Use: x = g^((p-1)/4) mod p
        x = pow(g, (p - 1) // 4, p)
        if (x * x) % p != p - 1:
            x = p - x
        # Now gcd(x + i, p) in Z[i] gives a factor
        # Use Euclidean algorithm in Z[i]
        a, b = x, 1
        while b != 0:
            q = a // b
            a, b = b, a - q * b
        # Now a is a Gaussian integer factor of p
        # a = a_re + a_im * i
        # But we need to extract real components
        # Actually, let's use a different approach:
        # We have x² ≡ -1 (mod p), so p | (x²+1) = (x+i)(x-i)
        # gcd(x+i, p) in Z[i]
        # Use the fact that if x² ≡ -1 mod p, then (x, 1) and (1, x) can help
        
        # Better approach: use the identity
        # If x² ≡ -1 (mod p), then p divides x²+1
        # We want to find a,b with a²+b²=p
        # Use: a = x mod p, but we need actual integers
        # Standard method: Euclidean algorithm on (x, p) in Z[i]
        
        # Let's use a simpler method for small p
        # Brute force for small p
        if p < 10**6:
            for a in range(1, int(math.isqrt(p)) + 1):
                b2 = p - a * a
                b = math.isqrt(b2)
                if b * b == b2:
                    return (a, b)
        
        # For larger p, use the Gaussian integer gcd
        # x² + 1 ≡ 0 (mod p) means p | (x+i)(x-i)
        # Compute gcd(x+i, x-i) in Z[i]
        # But they're not coprime in general
        
        # Alternative: use the formula
        # a = min(x, p-x), b = ?
        # Actually, if x² ≡ -1 mod p, then x² + 1 = kp for some k
        # And x² + 1 = (x+i)(x-i)
        # The norm of x+i is x²+1 = kp
        # So gcd(x+i, p) has norm p (or a factor)
        
        # Let's just do brute force for now since p is small in our use case
        for a in range(1, min(int(math.isqrt(p)) + 1, 10**5)):
            b2 = p - a * a
            b = math.isqrt(b2)
            if b * b == b2:
                return (a, b)
        
        # Fallback: should not happen for primes ≡ 1 mod 4
        return (1, math.isqrt(p - 1))

    def mul_gauss(z1, z2):
        """Multiply two Gaussian integers (a, b) * (c, d) = (ac-bd, ad+bc)."""
        a, b = z1
        c, d = z2
        return (a * c - b * d, a * d + b * c)

    def conj_gauss(z):
        """Conjugate of Gaussian integer (a, b) is (a, -b)."""
        return (z[0], -z[1])

    def norm_gauss(z):
        """Norm of Gaussian integer (a, b) is a² + b²."""
        return z[0] * z[0] + z[1] * z[1]

    def find_representations(M):
        """
        Find all pairs (x, y) with x <= y, x >= 0, x² + y² = M.
        Uses Gaussian integer factorization.
        """
        if M == 0:
            return [(0, 0)]
        if M == 1:
            return [(0, 1)]
        
        # Factor M into primes
        # M should be composed of 2 and primes ≡ 1 mod 4 (for our construction)
        # But handle general case
        
        # First, extract powers of 2
        # 2 = (1+i)(1-i) in Z[i]
        # 2^k has representations based on k
        
        # For now, let's use a hybrid approach:
        # If M is small enough, enumerate
        # If M is large, use Gaussian factorization
        
        limit_x = int(math.isqrt(M // 2)) + 1
        
        # If M is small, enumerate directly
        if limit_x <= 5 * 10**6:
            pairs = []
            for x in range(0, limit_x + 1):
                x2 = x * x
                y2 = M - x2
                y = math.isqrt(y2)
                if y * y == y2:
                    if x <= y:
                        pairs.append((x, y))
            return pairs
        
        # For large M, use Gaussian integer factorization
        # M = 2^a * prod p_i^e_i where p_i ≡ 1 mod 4
        # (assuming no primes ≡ 3 mod 4 with odd exponent)
        
        # Factor M
        m = M
        factors = {}
        
        # Extract 2
        c = 0
        while m % 2 == 0:
            m //= 2
            c += 1
        if c > 0:
            factors[2] = c
        
        # Extract odd primes
        p = 3
        while p * p <= m:
            if m % p == 0:
                c = 0
                while m % p == 0:
                    m //= p
                    c += 1
                factors[p] = c
            p += 2
        
        if m > 1:
            factors[m] = 1
        
        # Check if M can be written as sum of two squares
        # All primes ≡ 3 mod 4 must have even exponent
        for p, e in factors.items():
            if p % 4 == 3 and e % 2 == 1:
                return []  # No representations
        
        # Build Gaussian integer factors
        # 2 = (1+i)(1-i)
        # p ≡ 1 mod 4: p = (a+bi)(a-bi) where a²+b²=p
        
        gauss_factors = []
        
        # Handle 2
        if 2 in factors:
            e2 = factors[2]
            # 2^e = (1+i)^e * (1-i)^e
            # Each factor of 2 contributes one choice
            for _ in range(e2):
                gauss_factors.append([(1, 1), (1, -1)])
        
        # Handle primes ≡ 1 mod 4
        for p, e in factors.items():
            if p % 4 == 1:
                a, b = find_sqr_rep(p)
                # p = (a+bi)(a-bi)
                for _ in range(e):
                    gauss_factors.append([(a, b), (a, -b)])
        
        # Handle primes ≡ 3 mod 4 (must have even exponent)
        for p, e in factors.items():
            if p % 4 == 3:
                # p² = (p+0i)(p-0i) = p²
                # Each pair of factors contributes p
                for _ in range(e // 2):
                    gauss_factors.append([(p, 0), (p, 0)])
        
        # Now generate all combinations
        # Each gauss_factors[i] is a list of choices
        # We need to pick one from each, multiply them all, and get (x, y)
        
        # This can be exponential, so we need to be careful
        # But for our M, the number of factors is small
        
        # Start with z = (1, 0)
        # For each set of choices, multiply
        
        # To avoid exponential blowup, we'll use iterative combination
        # But we need to deduplicate at the end
        
        # Actually, the number of distinct representations is prod(e_i+1) for p_i ≡ 1 mod 4
        # which is what we're trying to maximize, so it's bounded
        
        # Let's generate all products
        results = set()
        
        # Use iterative multiplication
        current = [(1, 0)]
        
        for choices in gauss_factors:
            new_current = []
            for z in current:
                for choice in choices:
                    new_z = mul_gauss(z, choice)
                    new_current.append(new_z)
            current = new_current
        
        # Extract (x, y) from each Gaussian integer
        for z in current:
            x = abs(z[0])
            y = abs(z[1])
            # Ensure x <= y
            if x > y:
                x, y = y, x
            # Check x² + y² = M
            if x * x + y * y == M:
                results.add((x, y))
        
        return sorted(results)

    def dfs_exponents(idx, current_M, current_score, exponents, primes_list, M_max):
        best_M = [1]
        best_score = [0]
        
        def _dfs(i, cM, cS, exps):
            if time_left() < 5:
                return
            if cS > best_score[0]:
                best_score[0] = cS
                best_M[0] = cM
            if i == len(primes_list):
                return
            p = primes_list[i]
            val = 1
            e = 0
            while cM * val <= M_max:
                _dfs(i + 1, cM * val, cS * (e + 1), exps + [e])
                val *= p
                e += 1
                if val > M_max:
                    break
        
        _dfs(idx, current_M, current_score, exponents)
        return best_M[0], best_score[0]

    best_N = 0
    best_pairs = []

    # Iterate over possible Q values
    # Q = S^2, S = product of first k primes
    for k in range(1, 10):
        if time_left() < 15:
            break
        
        S = 1
        for i in range(k):
            S *= primes_all[i]
        Q = S * S
        
        M_max = 10**18 // Q
        if M_max < 2:
            continue
        
        # Find best M <= M_max
        # Use DFS to find optimal exponent vector
        best_M, best_score = dfs_exponents(0, 1, 1, [], p_1mod4[:15], M_max)
        
        if best_M < 1:
            continue
            
        M = best_M
        N = Q * M
        
        if N < 2 or N > 10**18:
            continue
        
        # Find all pairs (x, y) with x² + y² = M
        xy_pairs = find_representations(M)
        
        # Convert to (a, b) = (Q*x², Q*y²)
        pairs = []
        for x, y in xy_pairs:
            a = Q * x * x
            b = Q * y * y
            if a <= b:
                pairs.append([a, b])
        
        # Deduplicate
        pairs = [list(p) for p in set(tuple(p) for p in pairs)]
        pairs.sort()
        
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N

    # Fallback
    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]

    return {'N': best_N, 'pairs': best_pairs}
