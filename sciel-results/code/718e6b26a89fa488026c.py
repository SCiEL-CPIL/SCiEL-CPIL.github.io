"""Rewritten program: exhaustive search for max distinct pairwise-coprime sumset.

Key structural insight (parity):
  Pairwise-coprime sums => at most one even number in the whole distinct sumset.
  For a 3x3 grid a_i+b_j, the number of even sums is k*m + (3-k)*(3-m)
  where k=#even in A, m=#even in B. This equals 1 for NO (k,m) in [0,3]^2.
  Hence we must have ZERO even sums, forcing (A all odd & B all even) or
  (A all even & B all odd). Every sum is ODD; only odd primes matter.

We precompute prime-factor bitmasks for odd primes up to 255, then enumerate
all valid (A,B) pairs with A odd, B even (and the symmetric case), checking
that the 9 distinct sums are pairwise coprime. We maximize the number of
distinct sums.
"""
import time
from itertools import combinations

def _prime_masks():
    """Return a list of bitmasks for numbers 0..256.
    For odd n, mask has bits set for each odd prime factor.
    For even n or n<2, mask = 0 (we won't use these since sums are odd)."""
    MAX = 256
    # Sieve for primes up to MAX
    is_prime = [True] * (MAX + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX + 1, i):
                is_prime[j] = False
    
    # Map odd primes to bit positions
    odd_primes = [p for p in range(3, MAX + 1) if is_prime[p]]
    prime_to_bit = {p: i for i, p in enumerate(odd_primes)}
    
    masks = [0] * (MAX + 1)
    for n in range(1, MAX + 1):
        if n % 2 == 0:
            continue
        mask = 0
        for p in odd_primes:
            if p > n:
                break
            if n % p == 0:
                mask |= (1 << prime_to_bit[p])
        masks[n] = mask
    return masks

def _pairwise_coprime_masks(masks_list):
    """Check if all pairs of masks in the list are disjoint (gcd=1)."""
    n = len(masks_list)
    for i in range(n):
        for j in range(i+1, n):
            if masks_list[i] & masks_list[j]:
                return False
    return True

def construct():
    deadline = time.time() + 60  # 60s deadline with safety margin
    
    masks = _prime_masks()
    
    odds = list(range(1, 129, 2))   # odd numbers in [1, 128]
    evens = list(range(2, 129, 2))  # even numbers in [1, 128]
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Case 1: A all odd, B all even
    # Case 2: A all even, B all odd (symmetric, will find same count)
    
    # For efficiency, iterate over combinations of 3 from odds for A,
    # and 3 from evens for B.
    
    odds_combos = list(combinations(odds, 3))
    evens_combos = list(combinations(evens, 3))
    
    for A in odds_combos:
        if time.time() > deadline - 5:
            break
        for B in evens_combos:
            if time.time() > deadline - 2:
                break
            
            # Compute all 9 sums
            sums = [a + b for a in A for b in B]
            distinct_sums = list(set(sums))
            count = len(distinct_sums)
            
            if count <= best_count:
                continue
            
            # Check pairwise coprime
            sum_masks = [masks[s] for s in distinct_sums]
            if _pairwise_coprime_masks(sum_masks):
                best_count = count
                best_A = list(A)
                best_B = list(B)
    
    # Also check the symmetric case: A even, B odd
    # This will yield the same maximum count, but let's verify
    for A in evens_combos:
        if time.time() > deadline - 5:
            break
        for B in odds_combos:
            if time.time() > deadline - 2:
                break
            
            sums = [a + b for a in A for b in B]
            distinct_sums = list(set(sums))
            count = len(distinct_sums)
            
            if count <= best_count:
                continue
            
            sum_masks = [masks[s] for s in distinct_sums]
            if _pairwise_coprime_masks(sum_masks):
                best_count = count
                best_A = list(A)
                best_B = list(B)
    
    if best_A is None:
        # Fallback to seed
        return [[1, 7, 13], [4, 10, 16]]
    
    return [best_A, best_B]
