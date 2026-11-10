"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def sieve_primes(limit):
        """Generate list of primes up to limit."""
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        primes = []
        for i in range(2, limit + 1):
            if is_prime[i]:
                primes.append(i)
                if i * i <= limit:
                    for j in range(i * i, limit + 1, i):
                        is_prime[j] = 0
        return primes

    # We need to factor numbers up to 10^12, so primes up to 10^6
    # 10^6 is small enough to sieve quickly
    primes = sieve_primes(10**6)

    def compute_phi_segmented(lo, hi):
        """
        Compute phi for numbers in [lo, hi] using segmented approach.
        Returns a list of phi values.
        """
        n = hi - lo + 1
        if n <= 0:
            return []
        
        # phi[i] stores the current value for number lo+i
        phi = list(range(lo, hi + 1))
        # remaining[i] stores the part of lo+i not yet divided by small primes
        remaining = list(range(lo, hi + 1))
        
        for p in primes:
            if p * p > hi:
                break
            # Find the first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            for j in range(start, hi + 1, p):
                idx = j - lo
                if remaining[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle any remaining prime factors (which must be > 10^6, so at most one per number)
        for i in range(n):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]
        
        return phi

    def find_longest_distinct_run(phi_arr, offset, max_len=200):
        """Find the longest run of distinct phi values."""
        n = len(phi_arr)
        best_k = 0
        best_s = offset
        i = 0
        while i < n:
            seen = set()
            k = 0
            # Expand window as long as values are distinct
            while i + k < n and k < max_len and phi_arr[i + k] not in seen:
                seen.add(phi_arr[i + k])
                k += 1
            if k > best_k:
                best_k = k
                best_s = offset + i
            if k < max_len:
                i += 1
            else:
                # If we hit max_len, we can't do better, but we should still check 
                # if there's a longer run starting later? No, max_len is the cap.
                # But we should move i forward to see if another run of max_len exists?
                # Actually, if we found max_len, that's the best possible for this block.
                # We can return early if we just want ANY max_len, but we want the best.
                # Since max_len is the global cap, finding one is enough to return 200.
                # But to be safe, let's just continue or break if we found 200.
                if k == 200:
                    return best_s, best_k
                i += k
        return best_s, best_k

    best_s, best_k = 1, 1
    
    # Strategy:
    # 1. Check a few large ranges where collisions are negligible.
    # 2. If we find a run of 200, we are done.
    # 3. If not, fall back to smaller ranges.
    
    # Candidate ranges: [start, start + window_size]
    # We want window_size to be at least 200. Let's use 1000 to be safe.
    window_size = 1000
    
    candidates = [
        10**12 - window_size - 10,
        10**11 - window_size - 10,
        10**10 - window_size - 10,
        5 * 10**11 - window_size - 10,
        10**9 - window_size - 10,
    ]
    
    # Add a few random large candidates
    random.seed(42)
    for _ in range(5):
        scale = random.choice([10**9, 10**10, 10**11, 10**12])
        lo = random.randint(1, max(1, 10**12 - scale - window_size - 100))
        candidates.append(lo)

    for lo in candidates:
        if time_left() < 5.0:
            break
        
        hi = lo + window_size - 1
        if hi > 10**12:
            hi = 10**12
        if lo < 1:
            lo = 1
        
        try:
            phi_arr = compute_phi_segmented(lo, hi)
            if time_left() < 2.0:
                break
            
            s, k = find_longest_distinct_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
                
            if best_k == 200:
                break
        except Exception:
            continue

    # If we haven't found 200, try a dense search in smaller ranges
    # This is a fallback, but unlikely to be needed
    if best_k < 200 and time_left() > 10.0:
        # Try ranges around powers of 10
        for base in [10**6, 10**7, 10**8, 10**9]:
            if time_left() < 5.0:
                break
            # Check a few spots around base
            for offset in [0, base//10, base//2, base]:
                lo = base + offset
                hi = min(10**12, lo + window_size)
                if lo < 1:
                    continue
                try:
                    phi_arr = compute_phi_segmented(lo, hi)
                    s, k = find_longest_distinct_run(phi_arr, lo, max_len=200)
                    if k > best_k:
                        best_k = k
                        best_s = s
                    if best_k == 200:
                        break
                except Exception:
                    continue
            if best_k == 200:
                break

    # Final safety checks
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
        best_k = 1
        
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
