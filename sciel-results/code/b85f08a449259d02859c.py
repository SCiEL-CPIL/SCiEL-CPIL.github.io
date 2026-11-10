"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n via trial division."""
    if n <= 0:
        return 0
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            sp = 1
            pk = 1
            while temp % p == 0:
                temp //= p
                pk *= p
                sp += pk
            result *= sp
        p += 1
    if temp > 1:
        result *= (1 + temp)
    return result


def _find_solutions_for_M(M, max_k):
    """Find all k <= max_k such that k * sigma(k) == M."""
    sols = []
    for k in range(1, max_k + 1):
        if k * _sigma(k) == M:
            sols.append(k)
    return sols


def _verify_solution(M, k):
    """Verify that k * sigma(k) == M and k <= isqrt(M)."""
    if k < 1 or k > math.isqrt(M):
        return False
    return k * _sigma(k) == M


def construct():
    """
    Search for M <= 10^18 with many distinct k <= isqrt(M) satisfying k*sigma(k) = M.
    Strategy:
      1. Precompute a candidate M from known constructions (the seed M=333312 with 3 solutions).
      2. Use a systematic search over structured M values (highly composite-like) to find M with more solutions.
      3. For each candidate M, find all k <= isqrt(M) with k*sigma(k)=M.
      4. Return the M with the most solutions found, within time budget.
    """
    deadline = time.time() + 145.0  # safety margin for 150s limit

    # Known good starting point
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = len(best_k)

    # Strategy: Search over M values that are products of many small primes raised to powers,
    # which tend to have many divisors and thus more k with k*sigma(k)=M.
    # We'll search over M = prod p_i^a_i where a_i are small, and check how many k satisfy the equation.

    # Precompute sigma for k up to some limit for candidate M values
    # For M up to 10^18, isqrt(M) up to 10^9, so we can't iterate all k.
    # Instead, we factor M and enumerate k such that k | M and sigma(k) = M/k.

    def _divisors_of(n):
        """Generate all divisors of n."""
        divs = [1]
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                new_divs = []
                for d in divs:
                    for e in range(exp + 1):
                        new_divs.append(d * (p ** e))
                divs = new_divs
            p += 1
        if temp > 1:
            new_divs = []
            for d in divs:
                new_divs.append(d)
                new_divs.append(d * temp)
            divs = new_divs
        return divs

    def _sigma_from_factors(factors):
        """Compute sigma from prime factorization [(p, a), ...]."""
        result = 1
        for p, a in factors:
            result *= (p ** (a + 1) - 1) // (p - 1)
        return result

    def _factorize(n):
        """Factorize n into list of (prime, exponent) pairs."""
        factors = []
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                a = 0
                while temp % p == 0:
                    temp //= p
                    a += 1
                factors.append((p, a))
            p += 1
        if temp > 1:
            factors.append((temp, 1))
        return factors

    # For a given M, find k such that k | M and sigma(k) = M/k
    # Equivalently, k is a divisor of M and k * sigma(k) = M
    def _solutions_for_M(M):
        """Find all k <= isqrt(M) with k*sigma(k) = M."""
        if M <= 0:
            return []
        max_k = math.isqrt(M)
        divs = _divisors_of(M)
        sols = []
        for k in divs:
            if k > max_k:
                continue
            if k * _sigma(k) == M:
                sols.append(k)
        return sorted(sols)

    # Generate candidate M values: products of small primes with small exponents
    # We want M <= 10^18
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    # Generate M as products of powers of these primes
    # Use a priority queue or iterative approach
    candidates = set()
    candidates.add(333312)  # seed

    # Generate more candidates: M = 2^a * 3^b * 5^c * 7^d * ... with a,b,c,d <= 10
    # and M <= 10^18
    def _gen_candidates(primes, max_exp, max_val):
        """Generate products of prime powers up to max_val."""
        cands = [1]
        for p in primes:
            new_cands = []
            for c in cands:
                v = c
                for e in range(0, max_exp + 1):
                    if v > max_val:
                        break
                    new_cands.append(v)
                    v *= p
            cands = new_cands
        return cands

    # Try with first 6 primes, exponents up to 8
    # This could be large, so be careful
    # 2^8 * 3^8 * 5^8 * 7^8 * 11^8 * 13^8 is way too big
    # Let's be more selective: generate all products of 2^a * 3^b * 5^c * 7^d * 11^e * 13^f
    # with a,b,c,d,e,f in [0..6] and product <= 10^18

    # Better approach: just try a range of structured M values
    # M = lcm(1..n) for various n, or M = product of first few primes with small exponents

    # Let's try M values that are highly composite numbers near various magnitudes
    # Or simply: for each candidate M from a structured generation, compute solutions

    # Generate candidates more carefully
    # Use the fact that k*sigma(k) = M means M/k = sigma(k), so M/k >= k (since sigma(k) >= k+1 for k>=1)
    # Thus k <= sqrt(M), which is our constraint.

    # Search strategy: iterate over possible k structures
    # k = prod p_i^a_i, sigma(k) = prod (p_i^(a_i+1)-1)/(p_i-1)
    # M = k * sigma(k)
    # We want many distinct k giving the same M.

    # Alternative: For a given set of primes, vary exponents in k and compute M = k*sigma(k)
    # Group by M and count distinct k.

    # This is a number-theoretic search. Let's do it for small prime sets.

    # For each combination of exponents for primes [2,3,5,7,11], compute k and M=k*sigma(k)
    # Collect M -> set of k

    from collections import defaultdict

    M_to_k = defaultdict(set)

    # Exponent ranges: keep k <= 10^9 (since M <= 10^18, k <= 10^9)
    # For k = 2^a * 3^b * 5^c * 7^d * 11^e:
    # a in [0..29], b in [0..18], c in [0..12], d in [0..9], e in [0..7]
    # But k must be <= 10^9, so we can't iterate all.

    # Better: iterate k up to some limit, compute M = k*sigma(k), and group
    # But k up to 10^6 is feasible for one pass. Let's do k up to 200000.

    K_LIMIT = 200000

    for k in range(1, K_LIMIT + 1):
        if time.time() > deadline:
            break
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            M_to_k[M].add(k)

    # Find the M with the most k values
    # But also verify k <= isqrt(M)
    best_M2 = best_M
    best_k2 = best_k
    best_count2 = best_count

    for M, kset in M_to_k.items():
        if time.time() > deadline:
            break
        max_k = math.isqrt(M)
        valid_k = [k for k in kset if k <= max_k and k * _sigma(k) == M]
        if len(valid_k) > best_count2:
            best_count2 = len(valid_k)
            best_M2 = M
            best_k2 = sorted(valid_k)

    # If we found something better, use it
    if best_count2 > best_count:
        best_M = best_M2
        best_k = best_k2
        best_count = best_count2

    # Verify final result
    final_k = []
    for k in best_k:
        if _verify_solution(best_M, k):
            final_k.append(k)

    if len(final_k) < best_count:
        # Fallback to seed
        best_M = 333312
        final_k = [336, 372, 434]

    return {'M': int(best_M), 'k': list(final_k)}

# EVOLVE-BLOCK-END
