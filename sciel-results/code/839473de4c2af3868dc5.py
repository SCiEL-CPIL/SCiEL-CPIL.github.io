import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N_MIN = 211
    N_MAX = 10210
    
    # Sieve primes up to N_MAX
    is_prime = [True] * (N_MAX + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N_MAX)) + 1):
        if is_prime[i]:
            for j in range(i*i, N_MAX + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Pre-compute parity bitsets for all numbers up to N_MAX
    parities = [0] * (N_MAX + 1)
    for x in range(2, N_MAX + 1):
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
        parities[x] = b
    
    target = parities[210]
    
    # Strategy 1: Find minimal pair (a, b) with a < b, a >= N_MIN
    # such that parity(a) XOR parity(b) == target
    seen = {}  # parity -> smallest number with that parity
    for M in range(N_MIN, N_MAX + 1):
        if time.time() > deadline - 10:
            break
        v = parities[M]
        complement = v ^ target
        if complement in seen:
            a = seen[complement]
            prod = 210 * a * M
            s = math.isqrt(prod)
            if s * s == prod:
                return [a, M]
        if v not in seen:
            seen[v] = M
    
    # Strategy 2: Gaussian elimination for larger sets
    # We track the basis and also try to minimize the max element in the solution
    basis = {}  # pivot_bit -> (vec, mask)
    best_solution = None
    best_max = N_MAX + 1
    
    for M in range(N_MIN, N_MAX + 1):
        if time.time() > deadline - 3:
            break
        
        v = parities[M]
        pos = M - N_MIN
        m = 1 << pos
        
        current_v = v
        current_m = m
        
        while current_v != 0:
            pivot_bit = 1 << (current_v.bit_length() - 1)
            if pivot_bit in basis:
                bv, bm = basis[pivot_bit]
                current_v ^= bv
                current_m ^= bm
            else:
                break
        
        if current_v != 0:
            pivot_bit = 1 << (current_v.bit_length() - 1)
            basis[pivot_bit] = (current_v, current_m)
        
        # Check if target is in the span
        w = target
        result_mask = 0
        for pivot_bit in sorted(basis.keys(), reverse=True):
            bv, bm = basis[pivot_bit]
            if w & pivot_bit:
                w ^= bv
                result_mask ^= bm
        
        if w == 0:
            # Found a solution - extract it
            selected = []
            for i in range(M - N_MIN + 1):
                if (result_mask >> i) & 1:
                    selected.append(N_MIN + i)
            
            if not selected:
                continue
            
            # Verify
            prod = 210
            for x in selected:
                prod *= x
            s = math.isqrt(prod)
            if s * s == prod:
                max_sel = max(selected)
                if max_sel < best_max:
                    best_max = max_sel
                    best_solution = list(selected)
                
                # Since we're scanning M in increasing order and this is the first time
                # target is in the span, any valid solution must have max >= some threshold.
                # But the found solution might not use M, so let's try to improve.
                # Try removing the largest element and see if we can compensate
                if len(selected) > 1:
                    # Try to find a solution with smaller max by checking subsets
                    # that exclude the largest few elements
                    selected_sorted = sorted(selected, reverse=True)
                    # Try excluding the largest element
                    for exclude_idx in range(min(3, len(selected_sorted))):
                        exclude_elem = selected_sorted[exclude_idx]
                        remaining = [x for x in selected if x != exclude_elem]
                        if not remaining:
                            continue
                        # Check if target is in span of remaining elements' parities
                        # by trying to find a subset of remaining that XORs to target
                        # This is exponential, so only do it for small sets
                        if len(remaining) <= 20:
                            # Brute force subset search
                            n_rem = len(remaining)
                            found = False
                            for mask in range(1, 1 << n_rem):
                                xor_val = 0
                                for j in range(n_rem):
                                    if (mask >> j) & 1:
                                        xor_val ^= parities[remaining[j]]
                                if xor_val == target:
                                    subset = [remaining[j] for j in range(n_rem) if (mask >> j) & 1]
                                    if subset:
                                        max_subset = max(subset)
                                        if max_subset < best_max:
                                            # Verify
                                            prod2 = 210
                                            for x in subset:
                                                prod2 *= x
                                            s2 = math.isqrt(prod2)
                                            if s2 * s2 == prod2:
                                                best_max = max_subset
                                                best_solution = list(subset)
                                                found = True
                                    break
                            if found:
                                break
                
                # Return immediately if we found a good solution
                if best_solution is not None:
                    return best_solution
    
    if best_solution is not None:
        return best_solution
    
    # Fallback
    return [840]
