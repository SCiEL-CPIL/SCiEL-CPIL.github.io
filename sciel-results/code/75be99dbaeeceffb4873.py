"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _factorize(n):
    """Return list of primes (with multiplicity) of n."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append(n)
    return factors

def _squarefree_kernel(n):
    """Return the squarefree kernel (product of primes with odd exponent)."""
    factors = _factorize(n)
    # Count exponents
    from collections import Counter
    c = Counter(factors)
    kernel = 1
    for p, e in c.items():
        if e % 2 == 1:
            kernel *= p
    return kernel

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n_base = P['n']  # 210
    max_val = P['maximum_window']  # 10210
    max_size = P['maximum_size']  # 100
    
    # The product of selected integers multiplied by 210 must be a perfect square.
    # This means: 210 * prod(selected) = square
    # In terms of squarefree kernels: kernel(210) * XOR-of-kernels(selected) must be 1 (i.e., all exponents even)
    # Actually, we need the total product to be a square.
    # Let's think in terms of prime parity vectors over GF(2).
    
    # 210 = 2 * 3 * 5 * 7
    # So kernel(210) = 210 (since all exponents are 1)
    
    # We need to select distinct integers > 210 and <= 10210 such that:
    # kernel(210) XOR kernel(x1) XOR kernel(x2) ... XOR kernel(xk) = 1 (mod 2 in each prime)
    # i.e., the XOR of all squarefree kernels (including 210's) must equal 1.
    # Wait, more precisely: the product 210 * x1 * x2 * ... * xk must be a perfect square.
    # This is equivalent to: the squarefree part of (210 * x1 * ... * xk) must be 1.
    # The squarefree part is determined by the parity of exponents of each prime.
    
    # Let's represent each number by its "parity vector" over relevant primes.
    # Relevant primes: all primes up to 10210 that can appear in factorizations.
    
    # Strategy: 
    # 1. For each candidate number x in (210, 10210], compute its squarefree kernel.
    # 2. We need a subset S of these numbers such that:
    #    - |S| between 1 and 100
    #    - All distinct
    #    - kernel(210) XOR (XOR of kernels in S) == 1 (meaning the combined product is a square)
    #    - Minimize max(S) - 210
    
    # Since we want to minimize the maximum, we can try increasing upper bounds.
    # For a given upper bound M, consider all numbers in (210, M].
    # We need to find if there's a subset with XOR property.
    
    # The XOR of kernels: since kernels are squarefree, their XOR (in the multiplicative sense)
    # is the squarefree part of the product. We need this to be 1.
    
    # Actually, let's use a different approach. Represent each number by a bitmask of primes
    # with odd exponent. Then we need the XOR (bitwise) of all masks (including 210's mask) to be 0.
    
    # Let's find all relevant primes up to 10210.
    # Primes up to 10210.
    
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    primes = sieve_primes(max_val)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Compute parity bitmask for 210
    def parity_mask(n):
        mask = 0
        d = 2
        temp = n
        while d * d <= temp:
            count = 0
            while temp % d == 0:
                count += 1
                temp //= d
            if count % 2 == 1:
                if d in prime_to_idx:
                    mask |= (1 << prime_to_idx[d])
            d += 1 if d == 2 else 2
        if temp > 1:
            if temp in prime_to_idx:
                mask |= (1 << prime_to_idx[temp])
        return mask
    
    mask_210 = parity_mask(210)
    
    # For each candidate number, compute its parity mask
    # We'll try to find the smallest M such that there exists a valid subset from (210, M]
    
    # Use linear algebra over GF(2): we have vectors v_i for each candidate.
    # We need a subset whose XOR equals mask_210 (so that total XOR is 0).
    
    # To minimize max, we try M = 211, 212, ... and for each, check if mask_210 is in the span.
    # But we also need the subset size <= 100 and all elements distinct (which they are by construction).
    
    # Actually, we just need ANY subset (size 1 to 100) whose XOR = mask_210.
    # The minimum subset can be found via Gaussian elimination.
    
    # Let's precompute masks for all numbers from 211 to 10210.
    masks = {}
    for x in range(211, max_val + 1):
        masks[x] = parity_mask(x)
    
    # Now, try increasing M
    best_result = None
    best_M = None
    
    # For efficiency, we'll do Gaussian elimination incrementally or just for each M.
    # Since max_val is 10210 and we have ~10000 candidates, and num_primes ~ 1270,
    # doing full Gaussian elimination for each M might be slow.
    
    # Alternative: just find the minimal M by checking if mask_210 is representable.
    # We can maintain a basis incrementally.
    
    # Let's use a simpler approach: for each M from 211 to 10210, maintain a list of vectors
    # and check if mask_210 is in their span. We can do this with a basis.
    
    # Build basis incrementally
    basis = {}  # leading bit -> vector
    vectors = []  # list of (value, mask)
    
    def add_to_basis(mask):
        """Add mask to basis, return True if it's new (linearly independent)."""
        m = mask
        while m != 0:
            lead = m.bit_length() - 1
            if lead in basis:
                m ^= basis[lead]
            else:
                basis[lead] = m
                return True
        return False
    
    def can_represent(target):
        """Check if target is in the span of the basis."""
        m = target
        while m != 0:
            lead = m.bit_length() - 1
            if lead in basis:
                m ^= basis[lead]
            else:
                return False
        return True
    
    # We need to track which vectors were added to be able to find the actual subset.
    # For now, let's just find the minimal M, then reconstruct the subset.
    
    # Actually, let's just iterate M and check.
    # But we need to be able to reconstruct the subset too.
    
    # Let's store all vectors and do a full elimination when we find the minimal M.
    
    # First, find minimal M:
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
        v = masks[M]
        add_to_basis(v)
        if can_represent(mask_210):
            best_M = M
            break
    
    if best_M is None:
        # Fallback: use the original simple solution
        return [4 * P['n']]
    
    # Now reconstruct the subset from (211, best_M] that XORs to mask_210.
    # We'll collect all vectors and do Gaussian elimination with backtracking.
    
    candidates = [(x, masks[x]) for x in range(211, best_M + 1)]
    
    # Perform Gaussian elimination to find a subset
    # We'll build a basis with tracking of which original vectors were used.
    
    # Each basis vector will be (mask, set_of_indices)
    # where indices refer to the candidates list
    
    basis2 = {}  # leading bit -> (mask, frozenset of indices)
    index_to_basis = {}  # for backtracking
    
    for idx, (val, mask) in enumerate(candidates):
        m = mask
        used = {idx}
        while m != 0:
            lead = m.bit_length() - 1
            if lead in basis2:
                bm, bused = basis2[lead]
                m ^= bm
                used = used.symmetric_difference(bused)
            else:
                basis2[lead] = (m, used)
                break
        if m == 0:
            # This vector is dependent, ignore
            pass
    
    # Now check if mask_210 is representable
    m = mask_210
    result_indices = set()
    while m != 0:
        lead = m.bit_length() - 1
        if lead in basis2:
            bm, bused = basis2[lead]
            m ^= bm
            result_indices = result_indices.symmetric_difference(bused)
        else:
            # Should not happen since we checked
            break
    
    if m != 0:
        # Fallback
        return [4 * P['n']]
    
    # Convert indices back to values
    selected = [candidates[i][0] for i in sorted(result_indices)]
    
    # Verify
    if not selected:
        return [4 * P['n']]
    
    # Check size constraint
    if len(selected) > max_size:
        # This shouldn't happen typically, but fallback
        return [4 * P['n']]
    
    # Verify the square condition
    product = 210
    for x in selected:
        product *= x
    
    if not _is_perfect_square(product):
        return [4 * P['n']]
    
    # Verify all constraints
    for x in selected:
        if not isinstance(x, int) or x <= 210 or x > 10210:
            return [4 * P['n']]
    
    if len(set(selected)) != len(selected):
        return [4 * P['n']]
    
    return selected
# EVOLVE-BLOCK-END
