"""Targeted multi-prime construction: build N with many powerful representations
by exploiting the algebraic structure of powerful numbers directly."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _is_powerful(n):
    """Check if n is powerful by exact factorization.
    A number is powerful if every prime factor appears with exponent >= 2.
    1 is powerful.
    """
    if n <= 1:
        return True
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            exp = 0
            while m % p == 0:
                m //= p
                exp += 1
            if exp == 1:
                return False
        p += 1 if p == 2 else 2
    # remaining m is prime (or 1)
    if m > 1:
        # m appears to the first power
        return False
    return True

def _generate_powerfuls_up_to(limit):
    """Generate all powerful numbers <= limit using the a^2 * b^3 form."""
    powerfuls = set()
    max_b = int(limit ** (1/3)) + 2
    
    # Sieve for square-free numbers up to max_b
    is_sq_free = [True] * (max_b + 1)
    is_sq_free[0] = False
    if max_b >= 1:
        is_sq_free[1] = True
    # Mark non-square-free
    p = 2
    while p * p <= max_b:
        if is_sq_free[p]:
            p2 = p * p
            for k in range(p2, max_b + 1, p2):
                is_sq_free[k] = False
        p += 1
    
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerfuls.add(val)
    return powerfuls

def _build_n_from_primes(prime_set, max_exponent=6):
    """Given a set of primes, build powerful numbers as products of p^e where e>=2.
    Return a list of powerful numbers that are products of these primes."""
    results = [1]
    for p in prime_set:
        new_results = []
        for base in results:
            # Multiply by p^2, p^3, ..., p^max_exponent
            pe = p * p
            while base * pe <= 10**18:
                new_results.append(base * pe)
                pe *= p
        results.extend(new_results)
    return results

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Strategy: Use the algebraic structure to find N with many representations.
    # 
    # Key insight: If N = sum of powerful numbers, and we can express many powerful
    # numbers as N - a where a is also powerful, we get many pairs.
    #
    # Better approach: Generate powerful numbers up to a reasonable limit,
    # then use a smarter counting method.
    #
    # The current program's bottleneck is the O(n^2) pairwise counting.
    # Let's try a different approach:
    # 1. Generate powerful numbers up to L (vectorized)
    # 2. For candidate N values (found heuristically), count representations directly
    # 3. Use the fact that powerful numbers have structure to narrow candidates
    
    # Generate powerful numbers up to L
    L = 50_000_000  # 5 * 10^7
    
    if time.time() > deadline - 80:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
    
    powerfuls = _generate_powerfuls_up_to(L)
    powerfuls_sorted = sorted(powerfuls)
    powerfuls_set = set(powerfuls)
    n = len(powerfuls_sorted)
    
    if time.time() > deadline - 60:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Now find the best N.
    # Approach: Use a frequency-based method.
    # For each powerful a, the possible sums a+b where b is powerful and b>=a
    # are a + powerfuls_sorted[i:] for i where powerfuls_sorted[i] >= a.
    # 
    # Instead of O(n^2), use the following:
    # - The most likely candidates for high-count N are around 2*median(powerfuls)
    # - Use a sampling + refinement approach
    
    # First, get a rough estimate by sampling
    import random
    random.seed(42)
    
    # Sample pairs to find candidate N values
    sample_size = min(500000, n * n // 4)
    freq = defaultdict(int)
    
    for _ in range(sample_size):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i > j:
            i, j = j, i
        a = powerfuls_sorted[i]
        b = powerfuls_sorted[j]
        s = a + b
        freq[s] += 1
    
    # Get top candidates from sampling
    top_candidates = sorted(freq.items(), key=lambda x: -x[1])[:50]
    
    # Also add some structural candidates
    # Powerful numbers include 1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, ...
    # Sums of two powerfuls: 2, 5, 9, 10, 12, 13, 17, 18, 20, 21, 25, ...
    # Add multiples of small powerful numbers
    for m in range(2, 200):
        for pm in [1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, 100, 121, 125, 144, 169, 196]:
            cand = m * pm
            if 2 <= cand <= 2 * L:
                top_candidates.append((cand, 0))
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for cand, _ in top_candidates:
        if cand not in seen and 2 <= cand <= 2 * L:
            seen.add(cand)
            unique_candidates.append(cand)
    
    # Also add candidates near the sampled top
    for cand, _ in top_candidates[:20]:
        for delta in range(-10, 11):
            c = cand + delta
            if 2 <= c <= 2 * L and c not in seen:
                seen.add(c)
                unique_candidates.append(c)
    
    # Count representations for each candidate
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # Sort candidates by expected count (from sampling) descending
    cand_order = {c: i for i, c in enumerate(seen)}
    unique_candidates.sort(key=lambda c: -freq.get(c, 0))
    
    for cand_N in unique_candidates:
        if time.time() > deadline - 15:
            break
        
        # Count pairs (a, b) with a <= b, a + b = cand_N, both powerful
        pairs = []
        for a in powerfuls_sorted:
            if a > cand_N // 2:
                break
            b = cand_N - a
            if b in powerfuls_set:
                pairs.append([a, b])
        
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = cand_N
            best_pairs = pairs
    
    # Verify the best result
    if best_count == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Final verification: check each pair
    verified_pairs = []
    for a, b in best_pairs:
        if a + b == best_N and a <= b:
            if _is_powerful(a) and _is_powerful(b):
                verified_pairs.append([a, b])
    
    if not verified_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': verified_pairs}

# EVOLVE-BLOCK-END
