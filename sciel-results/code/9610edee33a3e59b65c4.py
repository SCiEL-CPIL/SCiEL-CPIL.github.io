import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 210
    max_val = 10210
    target_primes = [2, 3, 5, 7]

    def sieve(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit + 1) if is_prime[i]]

    primes_up_to = sieve(max_val)
    prime_to_idx = {p: i for i, p in enumerate(primes_up_to)}

    def get_bitmask(num):
        mask = 0
        temp = num
        for p in primes_up_to:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    mask |= (1 << prime_to_idx[p])
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        return mask

    number_bitmask = {}
    for num in range(211, max_val + 1):
        number_bitmask[num] = get_bitmask(num)

    target_mask = 0
    for p in target_primes:
        target_mask |= (1 << prime_to_idx[p])

    def gf2_solve(vectors, target):
        if time.time() > deadline - 3:
            return None
        
        n = len(vectors)
        if n == 0:
            return [] if target == 0 else None
        
        basis = {}
        
        for i, v in enumerate(vectors):
            if time.time() > deadline - 3:
                return None
            
            current = v
            current_idx = [i]
            
            while current != 0:
                hb = current.bit_length() - 1
                pivot_bit = 1 << hb
                if pivot_bit in basis:
                    bvec, bidx = basis[pivot_bit]
                    current ^= bvec
                    current_idx = list(set(current_idx) ^ set(bidx))
                else:
                    basis[pivot_bit] = (current, current_idx)
                    break
        
        reduced = target
        solution = []
        for pivot_bit, (bvec, bidx) in basis.items():
            if reduced & pivot_bit:
                reduced ^= bvec
                solution = list(set(solution) ^ set(bidx))
        
        if reduced == 0:
            return solution
        else:
            return None

    best_answer = None
    best_raw = float('inf')
    
    # Single element
    for k in range(2, 100):
        x = 210 * k * k
        if x > max_val:
            break
        if x - 210 < best_raw:
            best_answer = [x]
            best_raw = x - 210

    # Two elements
    for k in range(1, 200):
        if time.time() > deadline - 15:
            break
        val = 210 * k * k
        if val > max_val * max_val:
            break
        sqrt_val = math.isqrt(val)
        lo_x = max(211, math.ceil(val / max_val))
        hi_x = min(sqrt_val, max_val)
        for x in range(lo_x, hi_x + 1):
            if val % x == 0:
                y = val // x
                if y > 210 and y <= max_val and x != y:
                    m = max(x, y)
                    if m - 210 < best_raw:
                        prod = 210 * x * y
                        r = math.isqrt(prod)
                        if r * r == prod:
                            best_answer = [int(x), int(y)]
                            best_raw = m - 210

    # Binary search on M for 3+ elements
    def check_exists(M):
        if time.time() > deadline - 5:
            return None
        vectors = [number_bitmask[num] for num in range(211, M + 1)]
        return gf2_solve(vectors, target_mask)

    lo = 211
    hi = min(best_raw + 210, max_val) if best_raw < float('inf') else max_val
    found_solution = None
    found_M = None
    
    while lo <= hi and time.time() < deadline - 10:
        mid = (lo + hi) // 2
        sol = check_exists(mid)
        if sol is not None:
            found_M = mid
            found_solution = sol
            hi = mid - 1
        else:
            lo = mid + 1

    if found_solution is not None and found_M is not None:
        numbers_in_range = list(range(211, found_M + 1))
        subset = [numbers_in_range[i] for i in found_solution]
        prod = 210
        for x in subset:
            prod *= x
        r = math.isqrt(prod)
        if r * r == prod:
            if all(210 < x <= max_val for x in subset):
                if len(set(subset)) == len(subset):
                    if 1 <= len(subset) <= 100:
                        raw = max(subset) - 210
                        if raw < best_raw:
                            best_answer = [int(x) for x in subset]
                            best_raw = raw

    if best_answer:
        return best_answer
    
    return [840]
