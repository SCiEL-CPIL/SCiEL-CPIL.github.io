"""Square-product extension for 210: find distinct ints in (210,10210]
whose product * 210 is a perfect square, minimizing max - 210."""

import math
import time
from collections import defaultdict

N = 210
MAX_N = 10210
MAX_K = 100


def _is_square(n):
    r = math.isqrt(n)
    return r * r == n


def _sieve_primes(limit):
    """Generate all primes up to limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def _squarefree_kernel(n, primes):
    """Compute the squarefree kernel of n (product of primes with odd exponent)."""
    kernel = 1
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e & 1:
                kernel *= p
    if n > 1:
        kernel *= n
    return kernel


def construct():
    t_start = time.time()
    DEADLINE = t_start + 60.0  # safety margin under 90s kill

    # Generate primes up to MAX_N for trial division
    primes = _sieve_primes(MAX_N)

    # Direct construction:
    # We need product * 210 to be a square.
    # 210 = 2 * 3 * 5 * 7.
    # If we choose numbers of the form k_i * 210, then:
    # product = (210^m) * (prod k_i)
    # product * 210 = 210^(m+1) * (prod k_i)
    # For this to be a square, the parity of exponents of 2,3,5,7 must match.
    # 210^(m+1) has exponents (m+1) for 2,3,5,7.
    # So we need prod k_i to have parity (m+1) mod 2 for 2,3,5,7.
    # Also, all other prime factors must have even total exponent.
    
    # Let's simplify: choose k_i such that their squarefree kernels multiply to a square.
    # If we pick k_i, let sf(k_i) be its squarefree kernel.
    # Then prod sf(k_i) must be a square (i.e., every prime appears an even number of times).
    # This is equivalent to the XOR of the prime sets of sf(k_i) being empty.
    
    # We want to minimize max(k_i * 210) - 210 = 210 * max(k_i) - 210.
    # So we want to minimize max(k_i).
    # k_i must be such that k_i * 210 > 210 => k_i > 1.
    # And k_i * 210 <= 10210 => k_i <= 48.
    
    # So k_i in [2, 48].
    # We need a set of distinct k_i in [2, 48] such that the XOR of their squarefree prime sets is empty.
    # And we want to minimize max(k_i).
    
    # Precompute squarefree kernels for k in [2, 48]
    # Actually, we only care about the set of primes in the squarefree kernel.
    # Let's represent each k by a frozenset of primes in its squarefree kernel.
    
    k_to_primes = {}
    for k in range(2, 49):
        sf = _squarefree_kernel(k, primes)
        # Factor sf to get the prime set
        prime_set = set()
        temp = sf
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                prime_set.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            prime_set.add(temp)
        k_to_primes[k] = frozenset(prime_set)
    
    # Now we need to find a subset of distinct k's from [2, 48] such that the XOR of their prime sets is empty.
    # And we want to minimize the maximum k in the subset.
    # We can iterate over possible max_k from 2 to 48.
    # For a given max_k, we consider k in [2, max_k].
    # We need to find a non-empty subset S of [2, max_k] such that XOR_{k in S} prime_set(k) = empty.
    # This is a linear algebra problem over GF(2).
    # Each k corresponds to a vector in GF(2)^P, where P is the set of primes up to 48.
    # We need a non-trivial linear combination that sums to 0.
    # This is equivalent to finding a linear dependency among the vectors.
    # We can use Gaussian elimination to find a dependency.
    
    # Primes up to 48: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    # Let's map primes to indices.
    primes_up_to_48 = [p for p in primes if p <= 48]
    prime_to_idx = {p: i for i, p in enumerate(primes_up_to_48)}
    dim = len(primes_up_to_48)
    
    # Represent each k as a bitmask of primes in its squarefree kernel.
    k_to_mask = {}
    for k in range(2, 49):
        mask = 0
        for p in k_to_primes[k]:
            mask |= (1 << prime_to_idx[p])
        k_to_mask[k] = mask
    
    # For each max_k, find a dependency among masks of k in [2, max_k].
    # A dependency is a subset S such that XOR of masks is 0.
    # We can find this by building a matrix and finding the null space.
    # Since dim is small (15), we can do this efficiently.
    
    best_max_k = None
    best_set = None
    
    for max_k in range(2, 49):
        if time.time() > DEADLINE:
            break
        
        # Get masks for k in [2, max_k]
        masks = [k_to_mask[k] for k in range(2, max_k + 1)]
        
        # Find a non-trivial subset that XORs to 0.
        # We can do this by keeping track of the basis and the combination.
        # basis[i] = mask with highest bit i
        # combo[i] = bitmask of indices in [0, max_k-1] that make up basis[i]
        basis = {}
        combo = {}
        
        for i, m in enumerate(masks):
            cur_m = m
            cur_c = 1 << i
            # Reduce cur_m by existing basis
            while cur_m:
                high_bit = cur_m.bit_length() - 1
                if high_bit in basis:
                    cur_m ^= basis[high_bit]
                    cur_c ^= combo[high_bit]
                else:
                    break
            
            if cur_m == 0:
                # Found a dependency: cur_c is a non-zero combination that XORs to 0
                # Extract the set of k's
                selected_k = []
                for j in range(max_k - 1):
                    if cur_c & (1 << j):
                        selected_k.append(j + 2)  # k = j + 2
                if selected_k:
                    # This is a valid set. Check if it's distinct (it is, by construction)
                    # We found a set with max_k = max_k.
                    # Since we iterate max_k in increasing order, the first one we find is minimal.
                    best_max_k = max_k
                    best_set = selected_k
                    break
            else:
                high_bit = cur_m.bit_length() - 1
                basis[high_bit] = cur_m
                combo[high_bit] = cur_c
        
        if best_set is not None:
            break
    
    if best_set is not None:
        # Convert k's to actual numbers: n_i = k_i * 210
        numbers = [k * 210 for k in best_set]
        
        # Verify
        prod = N
        for x in numbers:
            prod *= x
        if _is_square(prod) and len(set(numbers)) == len(numbers) and all(
            isinstance(x, int) and not isinstance(x, bool) and 210 < x <= MAX_N for x in numbers
        ):
            return numbers
    
    # Fallback: 840 = 4*210, so 210*840 = 210^2*4 = (210*2)^2 is a perfect square
    return [840]


# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result: {result}")
    print(f"Max - 210: {max(result) - 210}")
    prod = 210
    for x in result:
        prod *= x
    print(f"Product is square: {_is_square(prod)}")
    print(f"Time: {t1 - t0:.2f}s")
