"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations

def _is_perfect_square(n):
    """Check if n is a perfect square using integer square root."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _prime_factorization(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _squarefree_kernel(n):
    """Return the squarefree kernel of n (product of primes with odd exponent)."""
    factors = _prime_factorization(n)
    sf = 1
    for p, e in factors.items():
        if e % 2 == 1:
            sf *= p
    return sf

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # Target: product of selected * 210 must be a perfect square.
    # 210 = 2*3*5*7. The XOR (symmetric difference) of the prime-parity vectors
    # of all selected numbers must equal the vector for 210.
    
    # BFS approach: iterate M upward, for each M collect candidates in (210, M],
    # then check if any subset has XOR equal to target.
    
    # Target signature: primes 2,3,5,7 each with odd exponent
    target_primes = frozenset([2, 3, 5, 7])
    
    def signature(n):
        """Return frozenset of primes dividing n to an odd power."""
        factors = _prime_factorization(n)
        return frozenset(p for p, e in factors.items() if e % 2 == 1)
    
    # Precompute signatures for numbers 211 to some upper bound
    # We'll iterate M and maintain a set of reachable XOR states
    
    best_max = 10211
    best_set = None
    
    # First, quick check: single number with sf = 210
    # n = 210 * k^2 > 210 => k >= 2 => n = 840
    n_single = 840
    if n_single <= 10210:
        best_max = n_single
        best_set = [n_single]
    
    # Now iterate M from 211 upward, checking for valid subsets
    # For each M, we have candidates = [211, ..., M]
    # We need a subset whose signature XOR = target_primes
    
    # Use DP: reachable states after considering some candidates
    # State = frozenset of primes with odd count so far
    # We want to reach target_primes
    
    # For efficiency, we'll try small subset sizes first
    # k=1: already done (840)
    # k=2: check all pairs
    # k=3: check all triples (may be expensive)
    
    # Let's do a BFS over M, and for each M try to find a valid subset
    # by checking subsets of increasing size
    
    # Precompute signatures up to a reasonable limit
    max_check = min(10210, 500)  # start with smaller range
    sigs = {}
    for n in range(211, max_check + 1):
        sigs[n] = signature(n)
    
    # For each M from 211 to max_check, check if any subset of (211..M) works
    for M in range(211, max_check + 1):
        if time.time() > deadline:
            break
        
        candidates = list(range(211, M + 1))
        cand_sigs = [(n, sigs[n]) for n in candidates]
        
        # Try k=2 first (fast)
        found = False
        for i in range(len(cand_sigs)):
            n1, s1 = cand_sigs[i]
            for j in range(i + 1, len(cand_sigs)):
                n2, s2 = cand_sigs[j]
                # XOR of s1 and s2
                xor_set = s1.symmetric_difference(s2)
                if xor_set == target_primes:
                    if M < best_max:
                        best_max = M
                        best_set = [n1, n2]
                    found = True
                    break
            if found:
                break
        
        if found:
            continue
        
        # Try k=3
        if len(cand_sigs) <= 50:  # limit for performance
            found = False
            for i in range(len(cand_sigs)):
                n1, s1 = cand_sigs[i]
                for j in range(i + 1, len(cand_sigs)):
                    n2, s2 = cand_sigs[j]
                    xor12 = s1.symmetric_difference(s2)
                    for k in range(j + 1, len(cand_sigs)):
                        n3, s3 = cand_sigs[k]
                        xor123 = xor12.symmetric_difference(s3)
                        if xor123 == target_primes:
                            if M < best_max:
                                best_max = M
                                best_set = [n1, n2, n3]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                continue
        
        # If we found a solution at this M, we can stop since we're going upward
        if best_set is not None and best_max <= M:
            break
    
    # Also try the partition-based approach for completeness
    primes_target = [2, 3, 5, 7]
    
    def get_partitions(elements):
        if not elements:
            yield []
            return
        first = elements[0]
        for partition in get_partitions(elements[1:]):
            for i in range(len(partition)):
                new_partition = partition[:i] + [partition[i] + [first]] + partition[i+1:]
                yield new_partition
            yield partition + [[first]]
    
    def min_num_for_sf(d, exclude=None):
        if exclude is None:
            exclude = set()
        k = math.isqrt(210 // d) + 1
        while d * k * k <= 210:
            k += 1
        while d * k * k <= 10210:
            n = d * k * k
            if n not in exclude:
                return n
            k += 1
        return None
    
    for partition in get_partitions(primes_target):
        sf_values = []
        for subset in partition:
            prod = 1
            for p in subset:
                prod *= p
            sf_values.append(prod)
        
        numbers = []
        used = set()
        valid = True
        for sf in sf_values:
            n = min_num_for_sf(sf, used)
            if n is None:
                valid = False
                break
            numbers.append(n)
            used.add(n)
        
        if valid and len(numbers) >= 1:
            max_val = max(numbers)
            if max_val < best_max:
                best_max = max_val
                best_set = numbers[:]
        
        if time.time() > deadline:
            break
    
    # Fallback
    if best_set is None:
        best_set = [840]
        best_max = 840
    
    # Verify
    product = 210
    for x in best_set:
        product *= x
    if not _is_perfect_square(product):
        best_set = [840]
        best_max = 840
    
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-END
