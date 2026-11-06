"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(is_prime[start:limit+1:step]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def compute_phi_segment(start, length, primes):
    """
    Compute exact phi(n) for n in [start, start+length-1].
    Returns list of phi values.
    """
    phi_val = list(range(start, start + length))
    rem = list(range(start, start + length))
    
    for p in primes:
        if p * p > start + length - 1:
            break
        
        # Find first multiple of p in [start, start+length-1]
        r = start % p
        if r == 0:
            first = start
        else:
            first = start + (p - r)
        
        if first > start + length - 1:
            continue
        
        for multiple in range(first, start + length, p):
            idx = multiple - start
            phi_val[idx] = phi_val[idx] // p * (p - 1)
            while rem[idx] % p == 0:
                rem[idx] //= p
                
    for i in range(length):
        if rem[i] > 1:
            phi_val[i] = phi_val[i] // rem[i] * (rem[i] - 1)
            
    return phi_val

def find_longest_distinct_run(phis):
    """Find the longest contiguous subarray with all distinct values.
    Returns (length, start_index)."""
    n = len(phis)
    if n == 0:
        return 0, 0
    best = 1
    best_start = 0
    left = 0
    seen = {}
    for right, val in enumerate(phis):
        if val in seen and seen[val] >= left:
            left = seen[val] + 1
        seen[val] = right
        current_len = right - left + 1
        if current_len > best:
            best = current_len
            best_start = left
    return best, best_start

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin
    
    # Generate primes up to 10^6 (sufficient for n up to 10^12)
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    best_k = 1
    best_start = 1
    
    random.seed(42)
    
    # Precompute useful structures for faster search
    # Strategy: Use a multi-phase approach with increasing sophistication
    
    # Phase 1: Quick random sampling to find any good run
    # Try many random starts, but focus on different magnitude ranges
    num_trials_phase1 = 200
    for i in range(num_trials_phase1):
        if time.time() > deadline - 60:
            break
        # Spread across different magnitudes
        magnitude = random.choice([4, 6, 8, 10, 12])
        lo = 10**(magnitude - 1)
        hi = min(10**magnitude - max_k, max_n - max_k)
        if hi <= lo:
            continue
        s = random.randint(lo, hi)
        phis = compute_phi_segment(s, max_k, primes)
        if len(set(phis)) == max_k:
            return {'start': s, 'length': max_k}
        run_len, run_start = find_longest_distinct_run(phis)
        if run_len > best_k:
            best_k = run_len
            best_start = s + run_start
    
    # Phase 2: If we found a decent run, try to extend it locally
    if best_k >= 5 and time.time() < deadline - 30:
        # Try extending around the best found run
        search_radius = min(500, max(50, best_k * 10))
        for delta in range(-search_radius, search_radius + 1):
            if time.time() > deadline - 25:
                break
            s = best_start + delta
            if s < 1 or s + max_k - 1 > max_n:
                continue
            phis = compute_phi_segment(s, max_k, primes)
            if len(set(phis)) == max_k:
                return {'start': s, 'length': max_k}
            run_len, run_start = find_longest_distinct_run(phis)
            if run_len > best_k:
                best_k = run_len
                best_start = s + run_start
    
    # Phase 3: Structured search near primorials and their multiples
    if time.time() < deadline - 20:
        primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, 6469693230, 200862915270, 69720375229710]
        for pm in primorials:
            if time.time() > deadline - 15:
                break
            if pm > max_n:
                break
            # Try various multiples
            max_mult = min(1000, max(1, int((max_n - max_k) / pm)))
            num_mults = min(100, max_mult)
            for mult in random.sample(range(1, num_mults + 1), num_mults):
                if time.time() > deadline - 12:
                    break
                base = pm * mult
                if base < max_k or base + max_k > max_n:
                    continue
                # Try offsets around this base
                for offset in range(-100, 101):
                    if time.time() > deadline - 10:
                        break
                    s = base + offset
                    if s < 1 or s + max_k - 1 > max_n:
                        continue
                    phis = compute_phi_segment(s, max_k, primes)
                    if len(set(phis)) == max_k:
                        return {'start': s, 'length': max_k}
                    run_len, run_start = find_longest_distinct_run(phis)
                    if run_len > best_k:
                        best_k = run_len
                        best_start = s + run_start
    
    # Phase 4: Systematic sweep in promising ranges
    if time.time() < deadline - 5:
        # Focus on ranges where we've had success
        if best_start > 1:
            # Try around the best found location with wider spacing
            base_lo = max(1, best_start - 100000)
            base_hi = min(max_n - max_k + 1, best_start + 100000)
            step = max(1, (base_hi - base_lo) // 200)
            for s in range(base_lo, base_hi, step):
                if time.time() > deadline - 3:
                    break
                phis = compute_phi_segment(s, max_k, primes)
                if len(set(phis)) == max_k:
                    return {'start': s, 'length': max_k}
                run_len, run_start = find_longest_distinct_run(phis)
                if run_len > best_k:
                    best_k = run_len
                    best_start = s + run_start
    
    return {'start': best_start, 'length': best_k}

# EVOLVE-BLOCK-END
