"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import heapq

def _square_free_kernel(n, primes):
    """Compute the square-free kernel of n with respect to the given primes."""
    kernel = 1
    for p in primes:
        count = 0
        while n % p == 0:
            n //= p
            count += 1
        if count % 2 == 1:
            kernel *= p
    return kernel

def _is_perfect_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    root = math.isqrt(n)
    return root * root == n

def construct():
    # The primes involved in 210 are 2, 3, 5, 7
    base_primes = [2, 3, 5, 7]
    
    # We need to find a set of distinct integers S, all > 210 and <= 10210,
    # such that 210 * prod(S) is a perfect square.
    # Let P = prod(S). Then 210 * P = 2 * 3 * 5 * 7 * P must be a square.
    # This implies that for each prime p, the exponent of p in 210*P must be even.
    # Let v_p(n) be the exponent of p in n.
    # v_p(210) + v_p(P) must be even for all p.
    # v_p(210) is 1 for p in {2,3,5,7} and 0 otherwise.
    # So for p in {2,3,5,7}, v_p(P) must be odd.
    # For p not in {2,3,5,7}, v_p(P) must be even.
    
    # We want to minimize max(S) - 210.
    # This is equivalent to minimizing max(S).
    
    # Strategy:
    # 1. Generate candidate numbers in the range (210, 10210].
    # 2. For each candidate, compute its "signature" relative to the condition.
    #    Specifically, we care about the parity of the exponents of primes in the number.
    #    Let sig(n) be a bitmask where bit i is 1 if the exponent of the i-th prime in n is odd.
    #    The primes are [2, 3, 5, 7, ...]. But we only care about primes that appear in 210 or in the candidates.
    #    Actually, any prime factor of a candidate contributes to the product.
    #    If a prime q > 7 appears in a candidate, its exponent in the product must be even.
    #    This means that if a candidate has a prime factor q > 7 to an odd power, we must include another candidate
    #    that also has q to an odd power (or multiple such candidates).
    #    
    #    Let's refine the state. The state is the set of primes with odd exponent in the current product P.
    #    We start with state = {2, 3, 5, 7} because we need to "fix" the odd exponents from 210.
    #    Wait, the condition is v_p(210) + v_p(P) is even.
    #    So v_p(P) must have the same parity as v_p(210).
    #    So the target state for P is exactly the set of primes where v_p(210) is odd, which is {2, 3, 5, 7}.
    #    So we need to select numbers such that the XOR (symmetric difference) of their prime exponent parity sets
    #    equals {2, 3, 5, 7}.
    
    # Let's represent the state as a frozenset of primes.
    # But the set of primes can be large. However, numbers are <= 10210.
    # Primes > 10210 cannot appear.
    # Primes > sqrt(10210) ~ 101 can appear at most once (to the power 1).
    # If a prime q > 101 appears in a number, it must appear in an even number of selected numbers.
    # This makes the search space complex.
    
    # Alternative approach:
    # Since we want to minimize the maximum value, we can try to find a small set of numbers
    # with small values.
    
    # Let's try to find a pair (a, b) such that 210 * a * b is a square.
    # This requires a * b = 210 * k^2 for some integer k? No.
    # 210 * a * b = m^2.
    # So a * b must be 210 * (square).
    # Let a = 210 * x * y^2 and b = 210 * x * z^2? No.
    # Let's use the kernel.
    # kernel(210 * a * b) = 1.
    # kernel(210) * kernel(a) * kernel(b) = 1 (mod squares).
    # kernel(210) = 210 (since 210 is square-free).
    # So kernel(a) * kernel(b) must be 210 (mod squares).
    # i.e., kernel(a) * kernel(b) / 210 is a square.
    
    # Let's generate all numbers in (210, 10210] and compute their kernels.
    # Then we look for pairs (a, b) with a < b such that kernel(a) * kernel(b) is 210 * square.
    # This is equivalent to kernel(a) * kernel(b) / 210 being a perfect square.
    # Let K_a = kernel(a), K_b = kernel(b).
    # We need K_a * K_b = 210 * s^2.
    # This implies K_a * K_b is divisible by 210, and the quotient is a square.
    
    # Let's iterate over possible max values M from 211 upwards.
    # For each M, consider all numbers a, b in (210, M] with a <= b.
    # Check if 210 * a * b is a square.
    # This is O(N^2) per M, which might be too slow if N is large.
    # But we stop as soon as we find a solution.
    # The current best is 240. So we only need to check M up to 239 to improve.
    # If we don't find a solution for M < 240, we return the known solution.
    
    # Let's optimize:
    # Precompute kernel(n) for all n in (210, 240].
    # Then check all pairs.
    
    # Actually, we can do better.
    # We want to minimize max(a, b).
    # Let's just check all pairs (a, b) with 210 < a <= b < 240.
    # If none works, return [224, 240].
    
    # Let's also check triples? A triple might have a smaller max than any pair.
    # E.g., max(triple) < min(max(pair)).
    # But pairs are simpler. Let's start with pairs.
    
    # Generate candidates up to 239.
    candidates = list(range(211, 240))
    
    # Precompute kernels
    kernels = {}
    for n in candidates:
        kernels[n] = _square_free_kernel(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    
    # Check pairs
    n_cand = len(candidates)
    for i in range(n_cand):
        a = candidates[i]
        for j in range(i, n_cand):
            b = candidates[j]
            # Check if 210 * a * b is a square
            prod = 210 * a * b
            if _is_perfect_square(prod):
                return [a, b]
    
    # If no pair found with max < 240, check triples?
    # Or just return the known best.
    # Let's quickly check if any triple has max < 240.
    # This is O(N^3) = 29^3 ~ 24000, which is fast.
    for i in range(n_cand):
        a = candidates[i]
        for j in range(i, n_cand):
            b = candidates[j]
            for k in range(j, n_cand):
                c = candidates[k]
                prod = 210 * a * b * c
                if _is_perfect_square(prod):
                    return [a, b, c]
    
    # If nothing found, return the known solution
    return [224, 240]
# EVOLVE-BLOCK-END
