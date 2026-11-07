"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
N = 210
MAX_VAL = 10210
MAX_SIZE = 100
DEADLINE = 90.0
SAFETY_MARGIN = 2.0

def is_perfect_square(n):
    """Check if n is a perfect square using integer square root."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def get_primes_up_to(n):
    """Sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Precompute primes up to MAX_VAL
    primes = get_primes_up_to(MAX_VAL)
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    def kernel_to_mask(n):
        """Convert n to its squarefree kernel bitmask."""
        mask = 0
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    mask |= (1 << prime_to_bit[p])
        if temp > 1:
            if temp in prime_to_bit:
                mask |= (1 << prime_to_bit[temp])
        return mask
    
    target_mask = kernel_to_mask(N)
    
    # Precompute masks for all numbers in (210, MAX_VAL]
    masks = {}
    for val in range(211, MAX_VAL + 1):
        masks[val] = kernel_to_mask(val)
    
    def find_subset(max_val):
        """Find subset of {211,...,max_val} with XOR = target_mask, size <= MAX_SIZE."""
        # Gaussian elimination to check if target is in span
        # Then extract solution
        basis = {}  # leading_bit -> mask
        origin = {}  # leading_bit -> list of values
        
        for v in range(211, max_val + 1):
            m = masks[v]
            if m == 0:
                continue
            cur = m
            cur_origin = [v]
            while cur != 0:
                lb = cur.bit_length() - 1
                if lb in basis:
                    cur ^= basis[lb]
                    # Combine origins
                    new_origin = cur_origin + origin[lb]
                    if len(new_origin) > MAX_SIZE:
                        break
                    cur_origin = new_origin
                else:
                    basis[lb] = cur
                    origin[lb] = cur_origin
                    break
            # If cur == 0, redundant vector (in span)
        
        # Reduce target
        t = target_mask
        solution = []
        while t != 0:
            lb = t.bit_length() - 1
            if lb in basis:
                t ^= basis[lb]
                solution.extend(origin[lb])
            else:
                return None
        
        if len(solution) > MAX_SIZE:
            return None
        
        # Verify distinctness
        if len(set(solution)) != len(solution):
            return None
        
        return solution
    
    # Fallback: single number 840
    best_solution = [840]
    best_max = 840
    
    # Try binary search for smallest max_val with valid subset
    lo = 211
    hi = best_max
    
    # Check if hi works with better max
    sol_hi = find_subset(hi)
    if sol_hi is not None:
        m = max(sol_hi)
        if m < best_max:
            best_solution = sol_hi
            best_max = m
    
    # Binary search
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        sol_mid = find_subset(mid)
        if sol_mid is not None:
            m = max(sol_mid)
            if m < best_max:
                best_solution = sol_mid
                best_max = m
            hi = mid
        else:
            lo = mid + 1
    
    # Verify final solution
    if best_solution:
        if not all(isinstance(x, int) for x in best_solution):
            best_solution = [840]
        if not (1 <= len(best_solution) <= MAX_SIZE):
            best_solution = [840]
        if not all(210 < x <= MAX_VAL for x in best_solution):
            best_solution = [840]
        if len(set(best_solution)) != len(best_solution):
            best_solution = [840]
        
        prod = N
        for x in best_solution:
            prod *= x
        if not is_perfect_square(prod):
            best_solution = [840]
    
    return list(best_solution)
# EVOLVE-BLOCK-END
