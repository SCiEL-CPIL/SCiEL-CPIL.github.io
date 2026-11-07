import math
import time
# EVOLVE-BLOCK-START
N = 210
MAX_VAL = 10210
MAX_SIZE = 100
DEADLINE = 88.0
SAFETY_MARGIN = 2.0

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
    """
    Compute squarefree kernel for all numbers up to limit using a sieve.
    sf[n] will hold the squarefree part of n.
    """
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
    """Convert a squarefree integer to a bitmask."""
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
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    # Precompute vectors using the fast kernel method
    vecs = {}
    for i in range(N + 1, MAX_VAL + 1):
        k = sf_kernels[i]
        vecs[i] = kernel_to_mask(k)
        
    target_vec = kernel_to_mask(sf_kernels[N])
    
    basis = []
    
    def reduce_vec(v):
        res = v
        for pivot, bvec in basis:
            if (res >> pivot) & 1:
                res ^= bvec
        return res

    def add_to_basis(v):
        for pivot, bvec in basis:
            if (v >> pivot) & 1:
                v ^= bvec
        if v != 0:
            pivot = (v & -v).bit_length() - 1
            basis.append((pivot, v))

    best_M = None
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
            
        v_M = vecs[M]
        add_to_basis(v_M)
        
        t = target_vec
        for pivot, bvec in basis:
            if (t >> pivot) & 1:
                t ^= bvec
        if t == 0:
            best_M = M
            break

    if best_M is None:
        return [840]

    # Rebuild with dependencies
    basis_deps = []
    
    def add_to_basis_dep(v, mask):
        for pivot, bvec, bmask in basis_deps:
            if (v >> pivot) & 1:
                v ^= bvec
                mask ^= bmask
        if v != 0:
            pivot = (v & -v).bit_length() - 1
            basis_deps.append((pivot, v, mask))
        return v, mask

    for i in range(N + 1, best_M + 1):
        v = vecs[i]
        mask = 1 << (i - (N + 1))
        add_to_basis_dep(v, mask)
        
    t = target_vec
    sol_mask = 0
    for pivot, bvec, bmask in basis_deps:
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
        return [840]
        
    product = N
    for x in selected:
        product *= x
        
    if is_perfect_square(product):
        return [int(x) for x in selected]
    else:
        return [840]

# EVOLVE-BLOCK-END
