"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from collections import defaultdict

def _is_powerful(n):
    """Check if n is a powerful number using exact factorization."""
    if n <= 1:
        return True
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp == 1:
                return False
        p += 1 if p == 2 else 2
    if temp > 1:
        return False
    return True

def _r2_count(n):
    """
    Compute the number of integer solutions (x,y) to x^2 + y^2 = n.
    Returns r_2(n) = 4 * prod_{p_i ≡ 1 mod 4} (e_i + 1) if all primes ≡ 3 mod 4
    have even exponents, else 0.
    """
    if n <= 0:
        return 0
    temp = n
    count = 4
    # Factor out 2
    while temp % 2 == 0:
        temp //= 2
    # Factor odd primes
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if p % 4 == 1:
                count *= (exp + 1)
            elif p % 4 == 3:
                if exp % 2 != 0:
                    return 0
        p += 2
    if temp > 1:
        if temp % 4 == 1:
            count *= 2
        elif temp % 4 == 3:
            return 0
    return count

def _sum_of_two_squares_pairs(n):
    """
    Find all pairs (u, v) with 0 <= u <= v, u^2 + v^2 = n.
    Returns list of [u, v].
    """
    if n < 0:
        return []
    pairs = []
    u = 0
    while u * u <= n:
        rem = n - u * u
        if rem < 0:
            break
        v = math.isqrt(rem)
        if v * v == rem and v >= u:
            pairs.append([u, v])
        u += 1
    return pairs

def construct():
    """
    Strategy: Pick a square-free integer Q (product of small primes ≡ 1 mod 4),
    then find N = Q * m where m has many representations as sum of two squares.
    For each representation m = u^2 + v^2, we get a = Q*u^2, b = Q*v^2 which are
    powerful (since Q is square-free, Q*u^2 has all prime exponents >= 2 except
    those in Q which appear once... wait, that's not powerful).
    
    Correction: If Q is square-free, Q*u^2 is NOT powerful unless Q=1.
    
    New strategy: Let a = x^2 * s1 and b = y^2 * s2 where s1, s2 are square-free.
    Then a+b = N. For a to be powerful, every prime in s1 must appear with even
    exponent in x^2*s1, meaning s1 must divide x^2, i.e., s1=1 (since s1 is square-free).
    
    So actually, powerful numbers are exactly a^2 * b^3 where b is square-free.
    
    Better approach: Use the fact that if we pick N such that N = k * Q where Q is
    a perfect square (not square-free!), and k has many sum-of-two-squares reps,
    then a = Q*u^2 and b = Q*v^2 are both powerful (since Q*u^2 = (sqrt(Q)*u)^2
    is a perfect square, hence powerful).
    
    So: pick Q to be a perfect square, N = Q * m, where m = u^2 + v^2.
    Then a = Q*u^2, b = Q*v^2 are both perfect squares, hence powerful.
    
    To maximize the number of pairs, we want m to have many representations as
    a sum of two squares. The number of unordered pairs (u,v) with u>=0, v>=0,
    u<=v, u^2+v^2=m is approximately r_2(m)/8.
    
    We want to maximize this subject to N = Q*m <= 10^18.
    
    Let's try Q = 1 (simplest), so N = m. We need m <= 10^18 with many sum-of-
    two-squares representations.
    
    m should be a product of primes ≡ 1 mod 4. Let's try m = 5^a * 13^b * 17^c * ...
    
    For m = 5^a * 13^b * 17^c, r_2(m) = 4*(a+1)*(b+1)*(c+1), and the number of
    unordered pairs is roughly r_2(m)/8.
    
    We want to maximize (a+1)*(b+1)*(c+1)*... subject to 5^a * 13^b * 17^c * ... <= 10^18.
    
    Let's search over exponent vectors.
    """
    # Primes ≡ 1 mod 4
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509]
    
    # We'll try to find m = product of primes_1mod4[i]^e[i] that maximizes prod(e[i]+1)
    # subject to m <= 10^18.
    
    # Use a greedy/beam search over exponent assignments
    # Start with all exponents 0, then increment exponents of smaller primes first
    
    best_score = 0
    best_exponents = {}
    best_m = 1
    
    # Try various combinations
    # First, try using just the first few primes
    for num_primes in range(1, min(15, len(primes_1mod4))):
        selected = primes_1mod4[:num_primes]
        
        # For a fixed set of primes, find exponents that maximize prod(e_i+1)
        # subject to prod(p_i^e_i) <= 10^18
        # This is a knapsack-like problem. Use dynamic programming or greedy.
        
        # Greedy: start with e_i = 0, then increment the exponent of the prime
        # that gives the best ratio of score increase to log cost
        exponents = [0] * num_primes
        m_val = 1
        score = 1
        
        # Try to improve by incrementing exponents
        for _ in range(50):  # max iterations
            best_gain = -1
            best_idx = -1
            for i in range(num_primes):
                new_m = m_val * selected[i]
                if new_m > 10**18:
                    continue
                new_score = score * (exponents[i] + 2) // (exponents[i] + 1)
                gain = new_score - score
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            exponents[best_idx] += 1
            m_val *= selected[best_idx]
            score = 1
            for e in exponents:
                score *= (e + 1)
        
        if m_val <= 10**18 and score > best_score:
            best_score = score
            best_exponents = {selected[i]: exponents[i] for i in range(num_primes)}
            best_m = m_val
    
    # Now find all pairs (u, v) with u^2 + v^2 = best_m
    pairs = _sum_of_two_squares_pairs(best_m)
    
    # Filter: u >= 1 (since a = u^2 should be >= 1, and 1 is powerful)
    # Actually, 0^2 = 0 is not a positive integer, but the problem says powerful numbers
    # and 1 counts. Let's check: a must be >= 1? The problem says a <= b, a+b=N,
    # and a,b are powerful. 0 is not typically considered, but let's be safe and
    # require a >= 1.
    
    valid_pairs = []
    for u, v in pairs:
        if u >= 1 and v >= 1:
            a = u * u
            b = v * v
            if a <= b and a + b == best_m:
                if _is_powerful(a) and _is_powerful(b):
                    valid_pairs.append([a, b])
    
    # Remove duplicates
    seen = set()
    unique_pairs = []
    for a, b in valid_pairs:
        key = (a, b)
        if key not in seen:
            seen.add(key)
            unique_pairs.append([a, b])
    
    # Sort for consistency
    unique_pairs.sort()
    
    if not unique_pairs:
        # Fallback: try a simpler construction
        # N = 65 = 1^2 + 8^2 = 4^2 + 7^2
        # Both 1 and 64 are powerful (perfect squares)
        # Also 16 and 49 are powerful
        N = 65
        test_pairs = []
        for u in range(1, int(math.isqrt(N)) + 1):
            rem = N - u*u
            v = math.isqrt(rem)
            if v*v == rem and v >= u:
                a = u*u
                b = v*v
                if _is_powerful(a) and _is_powerful(b):
                    test_pairs.append([a, b])
        return {'N': N, 'pairs': test_pairs}
    
    return {'N': best_m, 'pairs': unique_pairs}
# EVOLVE-BLOCK-END
