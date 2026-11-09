import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    def compute_primes_up_to(n):
        """Compute all primes up to n using simple sieve."""
        if n < 2:
            return []
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    def compute_phi_segment(start, end, primes):
        """
        Compute phi for all numbers from start to end using segmented sieve.
        Returns a list of phi values for [start, end].
        """
        length = end - start + 1
        # Initialize: phi[i] = i - start + 1 (the number itself)
        # Using lists of integers
        phi = list(range(start, end + 1))
        rem = list(range(start, end + 1))
        
        # Pre-compute primes squared to break early
        # We only need primes p where p*p <= end
        # But for segmented sieve, we need primes p <= sqrt(end)
        # Actually, we need to process all primes p such that there is a multiple of p in [start, end]
        # AND p is prime.
        # If p > sqrt(end), it can divide at most one number in the segment (if that number is a multiple of p).
        # But we handle "remaining factors" at the end.
        # So we only iterate primes p <= sqrt(end) in the main loop.
        
        limit = int(math.isqrt(end))
        
        for p in primes:
            if p > limit:
                break
            
            # Find the first multiple of p in [start, end]
            # first = ceil(start / p) * p
            first = ((start + p - 1) // p) * p
            
            # Iterate through multiples of p
            # Optimization: local variables for speed
            phi_local = phi
            rem_local = rem
            
            for j in range(first, end + 1, p):
                idx = j - start
                if rem_local[idx] % p == 0:
                    phi_local[idx] = (phi_local[idx] // p) * (p - 1)
                    # Remove all factors of p from rem
                    while rem_local[idx] % p == 0:
                        rem_local[idx] //= p
        
        # Handle remaining prime factors > sqrt(end)
        # If rem[i] > 1, it is a prime factor
        for i in range(length):
            if rem[i] > 1:
                phi[i] = (phi[i] // rem[i]) * (rem[i] - 1)
        
        return phi

    def scan_for_best_run(phi_vals, offset, max_k=200):
        """
        Scan phi_vals for the longest run of distinct values.
        phi_vals[i] corresponds to number offset + i.
        Uses a dictionary to track last seen position for O(N) scanning.
        Returns (best_k, best_start_number)
        """
        best_k = 1
        best_s = offset
        n = len(phi_vals)
        
        # last_seen[v] = index of last occurrence of value v
        last_seen = {}
        left = 0
        for right in range(n):
            v = phi_vals[right]
            if v in last_seen:
                if last_seen[v] >= left:
                    left = last_seen[v] + 1
            last_seen[v] = right
            
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = offset + left
                if best_k >= max_k:
                    break
        
        return best_k, best_s

    best_k = 1
    best_s = 1
    
    # Precompute primes up to 10^6
    max_primes_limit = 1000000
    primes = compute_primes_up_to(max_primes_limit)
    
    # Strategy: Scan contiguous blocks from 1 upwards.
    # Block size W.
    # Estimate time for one block.
    # If W = 10^6, and we have 78k primes, but most primes don't have multiples in a small window?
    # No, for p < W, there is at least one multiple.
    # For p > W, there is at most one multiple.
    
    # Let's start with W = 10^6.
    W = 1000000
    
    # We will scan until time runs out.
    # To be safe, we'll process blocks and check time.
    
    current_start = 1
    # Upper bound for start: 10^12 - W
    # But we won't reach 10^12.
    
    # Estimate time for first block to calibrate?
    # Or just use a conservative estimate.
    
    # Let's try to process as many blocks as possible.
    # We'll use a loop.
    
    # Note: The "remaining factor" loop is O(W).
    # The sieve loop is O(W log log W) roughly.
    
    while current_start <= 10**12 - W:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Check if we have enough time for the next block
        # Assume next block takes similar time to last, or estimate.
        # For now, just break if elapsed > deadline - 5.
        if elapsed > deadline - 5:
            break
            
        end = current_start + W - 1
        if end > 10**12:
            end = 10**12
        
        phi_vals = compute_phi_segment(current_start, end, primes)
        k_found, s_found = scan_for_best_run(phi_vals, current_start, max_k=200)
        
        if k_found > best_k:
            best_k = k_found
            best_s = s_found
            
        if best_k >= 200:
            break
            
        current_start = end + 1
        
    return {'start': best_s, 'length': best_k}
