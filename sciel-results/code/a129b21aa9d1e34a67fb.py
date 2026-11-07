"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _isqrt(n):
    return math.isqrt(n)

def _factor_parity(n, primes):
    """Return frozenset of primes dividing n an odd number of times."""
    result = set()
    temp = n
    for p in primes:
        if p * p > temp:
            break
        count = 0
        while temp % p == 0:
            count += 1
            temp //= p
        if count % 2 == 1:
            result.add(p)
    if temp > 1:
        result.add(temp)
    return frozenset(result)

def construct():
    n_base = 210
    max_size = 100
    max_value = 10210
    
    # Generate primes up to max_value
    limit = max_value
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    # Target parity: 210 = 2 * 3 * 5 * 7
    target = frozenset([2, 3, 5, 7])
    
    # Precompute parity vectors
    parity_map = {}
    for n in range(211, max_value + 1):
        parity_map[n] = _factor_parity(n, primes)
    
    # Incremental search for size 1 or 2 solutions
    best_result = None
    parity_to_min = {}  # parity -> smallest number with that parity
    
    for m in range(211, max_value + 1):
        pm = parity_map[m]
        
        # Check if m alone works (size 1)
        if pm == target:
            prod = m * 210
            s = _isqrt(prod)
            if s * s == prod:
                best_result = [m]
                break
        
        # Check if there's a previous number x such that parity(x) XOR pm = target (size 2)
        needed = target ^ pm
        if needed in parity_to_min:
            x = parity_to_min[needed]
            prod = x * m * 210
            s = _isqrt(prod)
            if s * s == prod:
                best_result = sorted([x, m])
                break
        
        # Add m to the map if not already present
        if pm not in parity_to_min:
            parity_to_min[pm] = m
    
    if best_result is not None:
        return list(best_result)
    
    # If no size 1 or 2 solution, use Gaussian elimination for the full range
    candidates = list(range(211, max_value + 1))
    n_cands = len(candidates)
    
    all_primes = set()
    for x in candidates:
        all_primes.update(parity_map[x])
    all_primes.update(target)
    
    prime_list = sorted(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(prime_list)}
    num_primes = len(prime_list)
    
    def to_bitset(fset):
        bits = 0
        for p in fset:
            if p in prime_to_idx:
                bits |= (1 << prime_to_idx[p])
        return bits
    
    target_bits = to_bitset(target)
    
    mat = [to_bitset(parity_map[x]) for x in candidates]
    comb = [1 << i for i in range(n_cands)]
    mat.append(target_bits)
    comb.append(0)
    
    n_rows = n_cands + 1
    rank = 0
    
    for col in range(num_primes):
        pivot = -1
        for r in range(rank, n_rows):
            if (mat[r] >> col) & 1:
                pivot = r
                break
        if pivot == -1:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        comb[rank], comb[pivot] = comb[pivot], comb[rank]
        for r in range(n_rows):
            if r != rank and ((mat[r] >> col) & 1):
                mat[r] ^= mat[rank]
                comb[r] ^= comb[rank]
        rank += 1
    
    if mat[-1] == 0:
        c = comb[-1]
        subset_indices = []
        for i in range(n_cands):
            if (c >> i) & 1:
                subset_indices.append(i)
        
        if 0 < len(subset_indices) <= max_size:
            subset = [candidates[i] for i in subset_indices]
            prod = 1
            for x in subset:
                prod *= x
            prod *= 210
            s = _isqrt(prod)
            if s * s == prod:
                best_result = subset
    
    if best_result is not None:
        return list(best_result)
    
    # Final fallback
    return [840]
# EVOLVE-BLOCK-END
