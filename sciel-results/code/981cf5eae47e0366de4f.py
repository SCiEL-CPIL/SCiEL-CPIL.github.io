import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N_MIN = 211
    N_MAX = 10210
    
    # Target: sf(210) = {2, 3, 5, 7} (primes with odd exponent in 210)
    # 210 = 2 * 3 * 5 * 7
    
    # Sieve primes up to N_MAX
    is_prime = [True] * (N_MAX + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N_MAX)) + 1):
        if is_prime[i]:
            for j in range(i*i, N_MAX + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Compute parity bitset for 210 (target)
    def parity_bitset(x):
        temp = x
        b = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    b |= (1 << prime_to_idx[p])
        if temp > 1:
            if temp in prime_to_idx:
                b |= (1 << prime_to_idx[temp])
            else:
                # temp is a prime > sqrt(N_MAX), should be in our prime list if <= N_MAX
                # but if temp > N_MAX... no, temp <= x <= N_MAX
                b |= (1 << prime_to_idx[temp])
        return b
    
    target = parity_bitset(210)
    
    # Gaussian elimination over GF(2)
    # We want to find the minimal M such that target is in span of {parity(x) : x in [211, M]}
    # and extract the subset.
    
    # basis: list of (vector, mask) where mask is a bitmask over positions 0..(M-211)
    # indicating which elements are in the combination
    
    basis = []  # list of (vec, mask)
    
    # For each M from 211 to N_MAX, add parity(M) to the system
    # and check if target is now in the span
    
    for M in range(N_MIN, N_MAX + 1):
        if time.time() > deadline - 3:
            break
        
        v = parity_bitset(M)
        pos = M - N_MIN  # position index
        m = 1 << pos
        
        # Add v to basis with mask m
        current_v = v
        current_m = m
        
        for bv, bm in basis:
            # Find pivot bit of bv (highest set bit)
            if bv == 0:
                continue
            pivot_bit = 1 << (bv.bit_length() - 1)
            if current_v & pivot_bit:
                current_v ^= bv
                current_m ^= bm
        
        if current_v != 0:
            # Add to basis and maintain sorted order by pivot bit
            basis.append((current_v, current_m))
            # Sort by pivot bit (highest set bit) descending
            basis.sort(key=lambda x: x[0].bit_length(), reverse=True)
        
        # Check if target is in span
        w = target
        result_mask = 0
        for bv, bm in basis:
            if bv == 0:
                continue
            pivot_bit = 1 << (bv.bit_length() - 1)
            if w & pivot_bit:
                w ^= bv
                result_mask ^= bm
        
        if w == 0:
            # target is in span, extract solution
            selected = []
            for i in range(M - N_MIN + 1):
                if (result_mask >> i) & 1:
                    selected.append(N_MIN + i)
            
            if not selected:
                # Empty set: product is 1, 210*1 = 210, not a square
                # So this shouldn't happen for our target
                continue
            
            # Verify
            prod = 210
            for x in selected:
                prod *= x
            s = math.isqrt(prod)
            if s * s == prod:
                # All selected are distinct (by construction), > 210, <= N_MAX
                sel_max = max(selected)
                raw = sel_max - 210
                return list(selected)
            else:
                # Shouldn't happen, but skip
                continue
    
    # Fallback: known valid solution
    # 840 = 2^3 * 3 * 5 * 7
    # 210 * 840 = 210^2 * 4 = (210*2)^2 = 420^2
    return [840]
