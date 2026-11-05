import time
import numpy as np

def construct():
    """
    Finds integer n in [10000, 1000000] minimizing the largest prime factor of n(n+1).
    Uses a bitmask to mark y-smooth numbers and scans for consecutive runs.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    N = 1000000
    MIN_N = 10000
    
    # Generate small primes up to a reasonable bound
    # We'll try increasing y until we find consecutive y-smooth numbers
    # Start with primes up to 1000, which should cover most smooth numbers in range
    
    def generate_primes(limit):
        """Generate primes up to limit using simple sieve."""
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0:2] = b'\x00\x00'
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    # Generate primes up to 1000 (should be sufficient for smooth numbers in [10000, 1000000])
    primes = generate_primes(1000)
    
    # Create a boolean array marking y-smooth numbers up to N
    smooth_mask = np.zeros(N + 1, dtype=bool)
    
    # Start with 1 as smooth
    smooth_mask[1] = True
    
    # Multiply combinations of small primes to find all y-smooth numbers
    # Use iterative approach: for each prime, multiply existing smooth numbers
    for p in primes:
        # For each existing smooth number, multiply by p^k until we exceed N
        current_smooth = smooth_mask.copy()
        for k in range(1, 20):  # Max exponent (2^20 > 10^6)
            power = p ** k
            if power > N:
                break
            # Multiply existing smooth numbers by power
            indices = np.where(current_smooth)[0]
            for idx in indices:
                new_val = idx * power
                if new_val <= N:
                    smooth_mask[new_val] = True
                else:
                    break
            current_smooth = smooth_mask.copy()
    
    # Check time
    if time.time() > deadline:
        return 10000
    
    # Find the best n where both n and n+1 are y-smooth
    # This minimizes the largest prime factor to at most the largest prime in our list
    
    best_n = None
    best_max_lpf = float('inf')
    
    # Scan through the range looking for consecutive smooth numbers
    for n in range(MIN_N, N):
        if smooth_mask[n] and smooth_mask[n+1]:
            # Both n and n+1 are y-smooth, so largest prime factor <= max(primes)
            max_lpf = max(primes) if primes else 0
            if max_lpf < best_max_lpf:
                best_max_lpf = max_lpf
                best_n = n
                # If we found a very small max_lpf, we can stop early
                if max_lpf <= 13:  # Very smooth, likely optimal
                    break
    
    if best_n is None:
        # Fallback: use the original sieve approach
        lpf = np.zeros(N + 1, dtype=np.int64)
        for i in range(2, N + 1):
            if lpf[i] == 0:
                lpf[i::i] = i
        
        if time.time() > deadline:
            return 10000
        
        n_range = np.arange(MIN_N, N)
        lpf_n = lpf[n_range]
        lpf_n1 = lpf[n_range + 1]
        max_lpf = np.maximum(lpf_n, lpf_n1)
        min_idx = np.argmin(max_lpf)
        best_n = n_range[min_idx]
    
    # Time check
    if time.time() > deadline:
        return int(best_n)
    
    return int(best_n)
