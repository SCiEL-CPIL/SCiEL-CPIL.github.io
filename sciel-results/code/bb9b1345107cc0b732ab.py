# EVOLVE-BLOCK-START
import time
from math import isqrt

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _squarefree_part(n):
    """Compute the squarefree part of n using trial division."""
    if n <= 1:
        return n
    sf = 1
    p = 2
    while p * p <= n:
        cnt = 0
        while n % p == 0:
            n //= p
            cnt ^= 1
        if cnt:
            sf *= p
        p += 1
    if n > 1:
        sf *= n
    return sf

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 211  # minimum value
    MAX_VAL = 10210
    MAX_SIZE = 100
    
    best_result = None
    best_max = MAX_VAL + 1
    
    # Strategy: iterate M (the maximum element) from 211 upward.
    # For each M, check if there exists a subset S of {211, ..., M} with max(S) = M
    # (or <= M, but we want to minimize max so we check M first) such that
    # product(S) * 210 is a perfect square.
    #
    # Key insight: product(S) * 210 is a square iff the squarefree part of product(S)
    # equals the squarefree part of 210 (which is 210 itself since 210 = 2*3*5*7 is squarefree).
    # Wait, that's not quite right. Let's think in terms of parity vectors.
    #
    # For a number x, define its parity vector v(x) as the vector of exponents mod 2
    # in the prime factorization. Then product(S) * 210 is a square iff:
    # XOR of v(x) for x in S equals v(210).
    #
    # Since 210 = 2*3*5*7, v(210) has 1s in positions 2,3,5,7 and 0 elsewhere.
    #
    # For a single element M: v(M) = v(210), i.e., M*210 is a square.
    # For a pair (a, M): v(a) XOR v(M) = v(210), i.e., v(a) = v(M) XOR v(210).
    #   Equivalently, a*M*210 is a square.
    #
    # Efficient approach for pairs:
    # For each M, compute sf_M = squarefree_part(M). We need a in (210, M) such that
    # squarefree_part(a) = squarefree_part(M * 210) / squarefree_part(M) ... 
    # Actually simpler: a*M*210 is square iff squarefree_part(a*M*210) = 1.
    # Let sf_total = squarefree_part(M * 210). Then we need squarefree_part(a) such that
    # sf(a) * sf_total is a square, i.e., sf(a) = sf_total (since both are squarefree).
    # Wait: sf(a*M*210) = sf(a) XOR sf(M*210) in the sense of symmetric difference of prime sets.
    # For the product to be a square, sf(a) must equal sf(M*210).
    #
    # So: for each M, compute sf_target = squarefree_part(M * 210).
    # Then find a in (210, M) with squarefree_part(a) == sf_target.
    # a = sf_target * t^2 for some t, and 210 < a < M.
    
    for M in range(211, MAX_VAL + 1):
        if time.time() > deadline - 5.0:
            break
        if M >= best_max:
            break
        
        # Check single element: M*210 is a perfect square
        if _is_perfect_square(M * 210):
            if M < best_max:
                best_max = M
                best_result = [M]
            continue
        
        # Check pairs (a, M): need a*M*210 to be a perfect square
        # a must have squarefree part = squarefree_part(M * 210)
        sf_target = _squarefree_part(M * 210)
        
        # a = sf_target * t^2, 210 < a < M, a != M
        if sf_target > 0:
            t_max = isqrt((M - 1) // sf_target)
            for t in range(1, t_max + 1):
                a = sf_target * t * t
                if a <= 210 or a >= M:
                    if a > M:
                        break
                    continue
                # Verify a*M*210 is a perfect square
                if _is_perfect_square(a * M * 210):
                    if M < best_max:
                        best_max = M
                        best_result = [a, M]
                    break
        
        # If we found a solution with this M, we can stop since we're iterating M upward
        if best_result is not None and best_max == M:
            break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # If no pair found, try triples for small M
    # For triples (a, b, M): need a*b*M*210 to be a square
    # a*b must have squarefree part = squarefree_part(M*210)
    # So a*b*sf(M*210) must be a square... wait.
    # sf(a*b*M*210) = 1 iff sf(a*b) = sf(M*210)
    # sf(a*b) = sf(a) XOR sf(b) (symmetric difference)
    # We need sf(a) XOR sf(b) = sf(M*210)
    # Equivalently, sf(a) * sf(b) * sf(M*210) is a square (product of three squarefree numbers
    # is a square iff they are all equal or two are equal and one is 1... no)
    # Actually: sf(a*b) = sf(M*210) iff a*b*sf(M*210) is a perfect square? No.
    # sf(x) = sf(y) iff x*y is a perfect square (for squarefree x,y).
    # So sf(a*b) = sf(M*210) iff (a*b) * sf(M*210) is a perfect square? 
    # Not exactly, because a*b may not be squarefree.
    # Better: a*b*M*210 is a square iff a*b has the same squarefree part as M*210.
    # Let sf_M210 = squarefree_part(M*210). Then a*b*M*210 is square iff 
    # squarefree_part(a*b) == sf_M210.
    # And squarefree_part(a*b) == sf_M210 iff a*b*sf_M210 is a perfect square? 
    # Let's verify: if sf(a*b) = s, then a*b = s * k^2 for some k.
    # a*b*sf_M210 = s * k^2 * sf_M210. This is a square iff s * sf_M210 is a square.
    # Since s and sf_M210 are squarefree, s * sf_M210 is a square iff s = sf_M210.
    # So yes: a*b*M*210 is a square iff a*b*sf_M210 is a perfect square.
    
    for M in range(211, 500):
        if time.time() > deadline - 3.0:
            break
        if M >= best_max:
            break
        
        sf_M210 = _squarefree_part(M * 210)
        
        # Find pairs (a, b) with 211 <= a < b < M such that a*b*sf_M210 is a perfect square
        for a in range(211, M - 1):
            if time.time() > deadline - 2.0:
                break
            for b in range(a + 1, M):
                if _is_perfect_square(a * b * sf_M210):
                    if M < best_max:
                        best_max = M
                        best_result = [a, b, M]
                    break
            if best_result is not None and best_max == M:
                break
        
        if best_result is not None and best_max == M:
            break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Fallback: known solution 840 (since 840*210 = 176400 = 420^2)
    return [840]

# EVOLVE-BLOCK-END
