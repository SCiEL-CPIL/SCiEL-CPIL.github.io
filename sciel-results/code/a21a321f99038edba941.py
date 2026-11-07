"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import deque

# EVOLVE-BLOCK-END

def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    
    def squarefree_kernel(n):
        """Return the squarefree kernel of n."""
        res = 1
        i = 2
        while i * i <= n:
            cnt = 0
            while n % i == 0:
                n //= i
                cnt += 1
            if cnt % 2 == 1:
                res *= i
            i += 1
        if n > 1:
            res *= n
        return res

    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def kernel_xor(a, b):
        """Compute sf(a * b) for squarefree a and b."""
        # For squarefree numbers, sf(a*b) = a*b/gcd(a,b)^2
        g = math.gcd(a, b)
        return (a // g) * (b // g)

    # 210 = 2 * 3 * 5 * 7 (squarefree)
    # We need: sf(210 * prod) = 1, i.e., sf(prod) = 210
    # Equivalently, the XOR of prime signatures must equal {2,3,5,7}
    
    target_kernel = 210  # sf(210) = 210
    
    # BFS over XOR states:
    # State: current squarefree kernel
    # Start: kernel = 1 (empty product)
    # Goal: kernel = 210
    # Edge: multiply by candidate n, new_kernel = kernel_xor(current, sf(n))
    # Cost: max value used so far
    # We want minimum max value
    
    # Precompute squarefree kernels for all candidates up to some limit
    max_candidate = 10210
    
    # For efficiency, we'll do a BFS where we track:
    # - The set of kernels reachable with max value <= M
    # - For each kernel, the minimum max value to reach it
    # - The path (which numbers were used)
    
    # Actually, let's use a different approach:
    # Iterate M from 211 upward
    # Maintain a dictionary: kernel -> (max_value, path)
    # When we reach kernel 210, we have our answer
    
    # But the state space could be large. Let's bound it.
    
    # Alternative: Use the fact that we want distinct integers.
    # For each M, candidates are {211, ..., M}
    # We need a subset whose XOR of kernels = 210
    
    # This is a subset XOR problem. We can use linear algebra over GF(2).
    # Represent each kernel as a bit vector over primes.
    # Find the smallest M such that 210 is in the span of {sf(n) : 211 <= n <= M}
    
    # Let's collect primes up to 10210
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve_primes(max_candidate)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def kernel_to_bits(n):
        """Convert squarefree kernel to bit tuple."""
        bits = set()
        for p in primes:
            if p > n:
                break
            if n % p == 0:
                bits.add(p)
                n //= p
        if n > 1:
            bits.add(n)
        return frozenset(bits)
    
    # For each candidate, compute its kernel bits
    # Then use Gaussian elimination over GF(2) to find when 210 becomes reachable
    
    target_bits = kernel_to_bits(210)
    
    # Build basis incrementally
    # basis: list of (bitset, candidate_value)
    # We add candidates in order and maintain a basis
    
    basis = []  # list of frozensets
    basis_values = []  # corresponding candidate values (for path reconstruction)
    basis_origins = []  # which candidate introduced this basis element
    
    def try_add_to_basis(bitset, value, origin):
        """Try to add bitset to basis. Returns True if added."""
        b = bitset
        for i, existing in enumerate(basis):
            if not b:
                break
            # XOR with existing if it reduces
            if len(b) > len(existing):
                b = b.symmetric_difference(existing)
            elif len(b) == len(existing):
                # Use the larger prime to decide
                max_b = max(b) if b else 0
                max_e = max(existing) if existing else 0
                if max_b > max_e:
                    b = b.symmetric_difference(existing)
                elif max_b < max_e:
                    # Swap: this one is better
                    basis[i] = b
                    basis_values[i] = value
                    basis_origins[i] = origin
                    return True
                else:
                    b = b.symmetric_difference(existing)
            else:
                b = b.symmetric_difference(existing)
        
        if b:
            # Find insertion point
            for i in range(len(basis)):
                max_b = max(b) if b else 0
                max_e = max(basis[i]) if basis[i] else 0
                if max_b > max_e:
                    basis.insert(i, b)
                    basis_values.insert(i, value)
                    basis_origins.insert(i, origin)
                    return True
            basis.append(b)
            basis_values.append(value)
            basis_origins.append(origin)
            return True
        return False
    
    def check_in_span(bitset):
        """Check if bitset is in the span of basis."""
        b = set(bitset)
        for existing in basis:
            if not b:
                break
            max_b = max(b) if b else 0
            max_e = max(existing) if existing else 0
            if max_b >= max_e:
                b = b.symmetric_difference(existing)
        return len(b) == 0
    
    # Iterate M upward
    best_result = None
    best_max = float('inf')
    
    # First, check size-1 solutions
    for k in range(1, 100):
        n = 210 * k * k
        if n > 211 and n <= 10210:
            if n < best_max:
                best_max = n
                best_result = [n]
        elif n > 10210:
            break
    
    # Now BFS: add candidates one by one
    # For each new candidate, check if target is now in span
    # If yes, reconstruct the path
    
    # We need to track not just the basis, but also how to reconstruct
    # Let's use a different approach: maintain for each basis vector,
    # the set of original candidates that XOR to it
    
    # Simpler: just track the basis and when target becomes reachable,
    # do a backtracking search
    
    # Actually, let's just do a direct search for small sizes
    # since the answer is likely small
    
    # Precompute kernels for a range
    max_search = 500
    kernels = {}
    for n in range(211, max_search + 1):
        kernels[n] = squarefree_kernel(n)
    
    # Size 1
    for n in range(211, max_search + 1):
        if kernels[n] == target_kernel:
            if n < best_max:
                best_max = n
                best_result = [n]
    
    # Size 2
    cands = list(range(211, max_search + 1))
    n_cands = len(cands)
    for i in range(n_cands):
        if time.time() > deadline:
            break
        a = cands[i]
        ka = kernels[a]
        if a >= best_max:
            continue
        for j in range(i + 1, n_cands):
            b = cands[j]
            if b >= best_max:
                break
            kb = kernels[b]
            if kernel_xor(ka, kb) == target_kernel:
                best_max = b
                best_result = [a, b]
                break
    
    # Size 3
    for i in range(n_cands):
        if time.time() > deadline:
            break
        a = cands[i]
        ka = kernels[a]
        if a >= best_max:
            continue
        for j in range(i + 1, n_cands):
            if time.time() > deadline:
                break
            b = cands[j]
            if b >= best_max:
                break
            kb = kernels[b]
            kab = kernel_xor(ka, kb)
            for k in range(j + 1, n_cands):
                c = cands[k]
                if c >= best_max:
                    break
                kc = kernels[c]
                if kernel_xor(kab, kc) == target_kernel:
                    best_max = c
                    best_result = [a, b, c]
                    break
            if best_result and best_result[-1] == c:
                break
        if best_result and len(best_result) == 3 and best_result[-1] <= cands[j]:
            break
    
    # Size 4
    for i in range(n_cands):
        if time.time() > deadline:
            break
        a = cands[i]
        ka = kernels[a]
        if a >= best_max:
            continue
        for j in range(i + 1, n_cands):
            if time.time() > deadline:
                break
            b = cands[j]
            if b >= best_max:
                break
            kb = kernels[b]
            kab = kernel_xor(ka, kb)
            for k in range(j + 1, n_cands):
                if time.time() > deadline:
                    break
                c = cands[k]
                if c >= best_max:
                    break
                kc = kernels[c]
                kabc = kernel_xor(kab, kc)
                for l in range(k + 1, n_cands):
                    d = cands[l]
                    if d >= best_max:
                        break
                    kd = kernels[d]
                    if kernel_xor(kabc, kd) == target_kernel:
                        best_max = d
                        best_result = [a, b, c, d]
                        break
                if best_result and len(best_result) == 4 and best_result[-1] <= cands[l]:
                    break
            if best_result and len(best_result) == 4 and best_result[-1] <= cands[k]:
                break
        if best_result and len(best_result) == 4 and best_result[-1] <= cands[j]:
            break
    
    if best_result is None:
        # Fallback: known valid solution
        best_result = [224, 240]
    
    # Verify the result
    prod = 210
    for x in best_result:
        prod *= x
    if not is_perfect_square(prod):
        # Should not happen, but fallback
        best_result = [224, 240]
    
    # Ensure types are correct
    return [int(x) for x in best_result]
