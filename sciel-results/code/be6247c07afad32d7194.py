import math
import time

# EVOLVE-BLOCK-START
N = 210
MAX_VAL = 10210
MAX_SIZE = 100
DEADLINE = 85.0

def get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

primes = get_primes(MAX_VAL)
prime_to_idx = {p: i for i, p in enumerate(primes)}

def compute_squarefree_kernels(limit):
    sf = list(range(limit + 1))
    for p in primes:
        p2 = p * p
        if p2 > limit:
            break
        for j in range(p2, limit + 1, p2):
            sf[j] = sf[j // p2]
    return sf

sf_kernels = compute_squarefree_kernels(MAX_VAL)

def kernel_to_mask(kernel):
    mask = 0
    temp = kernel
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            mask |= (1 << prime_to_idx[p])
            temp //= p
    if temp > 1:
        mask |= (1 << prime_to_idx[temp])
    return mask

def is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Precompute vectors
    vecs = {}
    for i in range(N + 1, MAX_VAL + 1):
        k = sf_kernels[i]
        vecs[i] = kernel_to_mask(k)
    
    target_vec = kernel_to_mask(sf_kernels[N])
    
    # Build basis with dependencies (each basis vector tracks which original vectors were used to form it)
    basis = []  # list of (pivot_bit, reduced_vec, dep_mask)
    
    def reduce_vec(v):
        res = v
        for pivot, bvec, _ in basis:
            if (res >> pivot) & 1:
                res ^= bvec
        return res
    
    def add_to_basis(v, orig_mask):
        # Try to reduce v using existing basis
        for pivot, bvec, bmask in basis:
            if (v >> pivot) & 1:
                v ^= bvec
                orig_mask ^= bmask
        if v != 0:
            pivot = (v & -v).bit_length() - 1
            basis.append((pivot, v, orig_mask))
        return orig_mask
    
    # We want to find the minimum M such that target_vec is in the span of {vecs[i] for i in N+1..M}
    # And the number of selected integers <= 100
    
    best_M = None
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
        
        v_M = vecs[M]
        orig_mask = 1 << (M - (N + 1))
        add_to_basis(v_M, orig_mask)
        
        # Check if target_vec is in the span
        t = target_vec
        for pivot, bvec, _ in basis:
            if (t >> pivot) & 1:
                t ^= bvec
        if t == 0:
            best_M = M
            break
    
    if best_M is None:
        return [840]
    
    # Rebuild basis from scratch up to best_M to get clean dependency masks
    basis2 = []
    
    def add_to_basis2(v, orig_mask):
        for pivot, bvec, bmask in basis2:
            if (v >> pivot) & 1:
                v ^= bvec
                orig_mask ^= bmask
        if v != 0:
            pivot = (v & -v).bit_length() - 1
            basis2.append((pivot, v, orig_mask))
    
    for i in range(N + 1, best_M + 1):
        v = vecs[i]
        mask = 1 << (i - (N + 1))
        add_to_basis2(v, mask)
    
    # Solve for target_vec
    t = target_vec
    sol_mask = 0
    for pivot, bvec, bmask in basis2:
        if (t >> pivot) & 1:
            t ^= bvec
            sol_mask ^= bmask
    
    if t != 0:
        return [840]
    
    selected = []
    for i in range(N + 1, best_M + 1):
        if (sol_mask >> (i - (N + 1))) & 1:
            selected.append(i)
    
    if len(selected) > MAX_SIZE:
        # Try to find a solution with fewer elements by using a different approach
        # For now, return fallback
        return [840]
    
    product = N
    for x in selected:
        product *= x
        
    if is_perfect_square(product):
        return [int(x) for x in selected]
    else:
        return [840]

# EVOLVE-BLOCK-END
