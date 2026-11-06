"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # Precompute phi for 1..N using sieve
    N = 2_000_000
    phi_arr = np.arange(1, N + 1, dtype=np.int64)
    
    # Sieve for phi
    for i in range(2, N + 1):
        if phi_arr[i - 1] == i:  # phi[i] == i means i is prime
            for j in range(i, N + 1, i):
                phi_arr[j - 1] -= phi_arr[j - 1] // i

    # Find the longest run of consecutive distinct phi values
    # Use a sliding window approach with a hash map to track last seen positions
    best_len = 1
    best_start = 1
    
    # We'll scan through and for each position, track the longest window ending there
    # where all values are distinct. Use a dict to store last occurrence of each phi value.
    last_occurrence = {}
    window_start = 1  # 1-indexed
    
    for i in range(1, N + 1):
        val = int(phi_arr[i - 1])
        if val in last_occurrence:
            # Move window start past the previous occurrence
            if last_occurrence[val] >= window_start:
                window_start = last_occurrence[val] + 1
        last_occurrence[val] = i
        current_len = i - window_start + 1
        if current_len > best_len:
            best_len = current_len
            best_start = window_start
    
    # Now we have a good candidate from small numbers. Let's also try to find
    # longer runs by checking larger ranges if time permits.
    # The key insight: for large n, phi(n) is often even, so distinctness is easier
    # but we need to verify. Let's do targeted searches in larger ranges.
    
    # Strategy: For larger ranges, phi values tend to be more spread out.
    # Let's try to find runs in ranges like [M, M+200] for various M.
    # We'll use a segmented approach: compute phi for a window and check distinctness.
    
    def compute_phi_segment(lo, hi):
        """Compute phi for all numbers in [lo, hi] using segmented sieve."""
        size = hi - lo + 1
        phi_seg = np.arange(lo, hi + 1, dtype=np.int64).copy()
        
        # For each prime p, multiply phi by (1 - 1/p) for multiples of p in range
        # We need primes up to sqrt(hi)
        limit = int(np.sqrt(hi)) + 1
        
        # Generate primes up to limit using simple sieve
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = False
        if limit >= 1:
            is_prime[1] = False
        for p in range(2, limit + 1):
            if is_prime[p]:
                for multiple in range(p * p, limit + 1, p):
                    is_prime[multiple] = False
        
        primes = [p for p in range(2, limit + 1) if is_prime[p]]
        
        for p in primes:
            # Find first multiple of p in [lo, hi]
            first = ((lo + p - 1) // p) * p
            for j in range(first, hi + 1, p):
                idx = j - lo
                phi_seg[idx] -= phi_seg[idx] // p
        
        return phi_seg

    # Try searching in larger ranges
    # Based on theory, long runs should exist. Let's try several ranges.
    # We'll prioritize ranges where we expect good density of distinct values.
    
    search_ranges = []
    # Try ranges around powers of 2, factorials, and other structured points
    # Also try random-ish ranges
    
    # First, let's try to extend our best found run by checking nearby
    # and then do broader searches
    
    # Generate candidate start points for larger searches
    candidates = set()
    
    # Add some structured candidates
    for base in [10**6, 2*10**6, 5*10**6, 10**7, 2*10**7, 5*10**7, 10**8, 2*10**8, 5*10**8, 10**9, 2*10**9, 5*10**9, 10**10, 2*10**10, 5*10**10, 10**11, 2*10**11, 5*10**11]:
        if base <= 10**12 - 200:
            candidates.add(base)
            candidates.add(base - 100)
            candidates.add(base + 100)
    
    # Add some random-ish candidates
    np.random.seed(42)
    for _ in range(500):
        c = int(np.random.randint(1, 10**12 - 200))
        candidates.add(c)
    
    # Sort candidates and try them, prioritizing smaller ones first (faster to verify)
    candidates = sorted(candidates)
    
    for s in candidates:
        if time_remaining() < 5.0:
            break
        if s < 1 or s + 200 > 10**12:
            continue
        
        # Compute phi for [s, s+200]
        phi_seg = compute_phi_segment(s, s + 200)
        
        # Find longest run of distinct values
        last_occ = {}
        win_start = 0
        for i in range(201):
            val = int(phi_seg[i])
            if val in last_occ:
                if last_occ[val] >= win_start:
                    win_start = last_occ[val] + 1
            last_occ[val] = i
            cur_len = i - win_start + 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = s + win_start
    
    # Verify our best answer
    if best_start + best_len - 1 > 10**12:
        best_len = 1
        best_start = 1
    
    # Final verification: make sure the phi values are indeed distinct
    if best_len > 1:
        phi_check = compute_phi_segment(best_start, best_start + best_len - 1)
        vals = [int(v) for v in phi_check]
        if len(set(vals)) != len(vals):
            # Fallback to our small-range answer
            # Re-find in small range
            best_len = 1
            best_start = 1
            last_occurrence2 = {}
            window_start2 = 1
            for i in range(1, min(N, 200000) + 1):
                val = int(phi_arr[i - 1])
                if val in last_occurrence2:
                    if last_occurrence2[val] >= window_start2:
                        window_start2 = last_occurrence2[val] + 1
                last_occurrence2[val] = i
                current_len2 = i - window_start2 + 1
                if current_len2 > best_len:
                    best_len = current_len2
                    best_start = window_start2
    
    return {'start': best_start, 'length': best_len}
# EVOLVE-BLOCK-END
