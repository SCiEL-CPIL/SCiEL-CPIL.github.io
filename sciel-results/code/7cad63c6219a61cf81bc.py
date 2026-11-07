"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _squarefree_part(n):
    """Return the squarefree part of n (product of primes with odd exponent)."""
    result = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            if exp % 2 == 1:
                result *= d
        d += 1
    if n > 1:
        result *= n
    return result

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _verify_solution(selected):
    """Verify that product of selected * 210 is a perfect square."""
    prod = 210
    for x in selected:
        prod *= x
    return _is_perfect_square(prod)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = P['n']
    
    # Precompute squarefree parts for numbers from 211 to some upper bound
    # We'll search incrementally
    max_M = 310  # Start with a reasonable bound
    
    # We need: product(selected) * 210 is a perfect square
    # Equivalently: squarefree_part(product(selected) * 210) == 1
    # Which means: squarefree_part(product(selected)) == squarefree_part(210)
    
    # 210 = 2 * 3 * 5 * 7, so squarefree_part(210) = 210
    sf_210 = _squarefree_part(210)  # = 210
    
    best_raw = 10000
    best_solution = [224, 240]  # Known good: 224*240*210 = 3360^2, raw=30
    
    # Precompute sf for numbers 211..max_M
    sf_map = {}
    for i in range(211, max_M + 1):
        sf_map[i] = _squarefree_part(i)
    
    # Phase 1: Search for 2-element solutions
    # Need sf(a) * sf(b) ≡ 210 (mod squares)
    # i.e., squarefree_part(sf(a) * sf(b)) == 210
    # Since sf values are squarefree, we need sf(a) * sf(b) to have squarefree part 210
    # This means sf(a) * sf(b) = 210 * k^2 for some integer k
    # Or equivalently, sf(a) * sf(b) / 210 must be a perfect square (when divisible)
    # Better: just check if _squarefree_part(sf(a) * sf(b)) == 210
    
    for a in range(211, max_M + 1):
        if time.time() > deadline - 10:
            break
        for b in range(a + 1, max_M + 1):
            combined_sf = _squarefree_part(sf_map[a] * sf_map[b])
            if combined_sf == sf_210:
                # Verify directly
                if _verify_solution([a, b]):
                    raw = max(a, b) - 210
                    if raw < best_raw:
                        best_raw = raw
                        best_solution = [a, b]
    
    # Phase 2: Search for 3-element solutions with smaller max
    # We want max(selected) - 210 < best_raw
    # So max(selected) < 210 + best_raw
    upper_bound = 210 + best_raw - 1
    if upper_bound > max_M:
        upper_bound = max_M
    
    # Build a map from sf value to list of numbers with that sf
    sf_to_numbers = defaultdict(list)
    for i in range(211, min(upper_bound, max_M) + 1):
        sf_to_numbers[sf_map[i]].append(i)
    
    # For 3 elements a, b, c: need sf(a)*sf(b)*sf(c) to have squarefree part 210
    # i.e., _squarefree_part(sf(a)*sf(b)*sf(c)) == 210
    # Since all sf are squarefree, let's think of it as: the product sf(a)*sf(b)*sf(c)
    # has each prime appearing with odd total exponent iff it's in 210.
    
    # Iterate over pairs (a, b) and find c
    for a in range(211, min(upper_bound, max_M) + 1):
        if time.time() > deadline - 5:
            break
        for b in range(a + 1, min(upper_bound, max_M) + 1):
            if time.time() > deadline - 5:
                break
            # We need c such that _squarefree_part(sf_map[a] * sf_map[b] * sf_map[c]) == 210
            # Let sf_ab = sf_map[a] * sf_map[b]
            # We need _squarefree_part(sf_ab * sf_map[c]) == 210
            # This is equivalent to: sf_map[c] must satisfy a specific condition
            # Let's just iterate c > b for small ranges
            # For efficiency, limit c search
            max_c = min(upper_bound, max_M)
            # Only search if upper_bound is small enough for O(n^3)
            if max_c - b > 200:
                # Too many candidates, skip this pair or use smarter method
                continue
            for c in range(b + 1, max_c + 1):
                combined = _squarefree_part(sf_map[a] * sf_map[b] * sf_map[c])
                if combined == sf_210:
                    if _verify_solution([a, b, c]):
                        raw = max(a, b, c) - 210
                        if raw < best_raw:
                            best_raw = raw
                            best_solution = [a, b, c]
    
    # Phase 3: Try extending the search range if we have time
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Expand max_M and recheck
        extended_max = min(400, 210 + best_raw - 1)
        if extended_max > max_M:
            for i in range(max_M + 1, extended_max + 1):
                sf_map[i] = _squarefree_part(i)
            # Quick recheck for 2-element
            for a in range(max_M + 1, extended_max + 1):
                if time.time() > deadline - 2:
                    break
                for b in range(a + 1, extended_max + 1):
                    combined_sf = _squarefree_part(sf_map[a] * sf_map[b])
                    if combined_sf == sf_210:
                        if _verify_solution([a, b]):
                            raw = max(a, b) - 210
                            if raw < best_raw:
                                best_raw = raw
                                best_solution = [a, b]
    
    # Final verification
    if not _verify_solution(best_solution):
        best_solution = [224, 240]
        best_raw = 30
    
    # Ensure types
    result = [int(x) for x in best_solution]
    return result

# EVOLVE-BLOCK-END
