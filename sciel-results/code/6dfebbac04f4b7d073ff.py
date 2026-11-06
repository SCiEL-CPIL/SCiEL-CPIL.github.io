import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Pre-compute primes up to sqrt(10^12) = 10^6
    # This is fast and allows efficient phi computation for large numbers
    max_prime_limit = 10**6
    is_prime = [True] * (max_prime_limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_prime_limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, max_prime_limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_prime_limit + 1) if is_prime[i]]
    
    def compute_phi_segment(start, end):
        """
        Compute phi(n) for n in [start, end] using the pre-computed primes.
        Returns a list of phi values.
        """
        length = end - start + 1
        if length <= 0:
            return []
        
        # phi[n] will hold the current value for phi(n)
        # remaining[n] will hold the remaining factor of n after removing small primes
        phi = list(range(start, end + 1))
        remaining = list(range(start, end + 1))
        
        # For each prime, update phi and remaining for multiples of p
        for p in primes:
            if p * p > end:
                break
            
            # Find the first multiple of p in [start, end]
            first = ((start + p - 1) // p) * p
            
            # Iterate through multiples of p in the segment
            # We only need to process if p is a factor of the original number
            for m in range(first, end + 1, p):
                idx = m - start
                if remaining[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    # Remove all factors of p from remaining
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # After processing all small primes, if remaining > 1, it's a prime factor > sqrt(end)
        # There can be at most one such factor for any n <= 10^12
        for i in range(length):
            if remaining[i] > 1:
                p = remaining[i]
                phi[i] -= phi[i] // p
        
        return phi

    def find_max_distinct_run(phi_values, offset):
        """
        Find the longest subarray with all distinct values.
        Returns (length, start_index_absolute).
        """
        if not phi_values:
            return 0, offset
            
        seen = {}
        left = 0
        max_len = 0
        max_start = offset
        
        for right, val in enumerate(phi_values):
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                max_start = offset + left
        
        return max_len, max_start

    best_length = 1
    best_start = 1
    
    # Search Strategy:
    # 1. Check small ranges [1, N] to find any initial patterns.
    # 2. Check random ranges to avoid missing isolated long runs.
    # 3. If time permits, check larger chunks in specific interesting areas.
    
    # Define search windows. We want to balance coverage and computation cost.
    # A segment of size K takes roughly K * (number of primes <= sqrt(end)/K) operations?
    # Actually, it's sum over primes p of (K/p). So cost is ~ K * log log K.
    # 10^5 size is very fast. 10^6 is moderate. 10^7 might be slow in pure Python.
    
    # Let's start with a fixed set of strategic windows.
    # Note: The maximum possible length is bounded by the density of collisions.
    # Collisions are rare for large n, so long runs are possible.
    # However, we must verify distinctness exactly.
    
    search_windows = []
    
    # Small numbers: dense, many collisions expected, but let's check just in case.
    # Actually, for n=1, phi=1. n=2, phi=1. Collision immediately.
    # So small numbers won't yield long runs.
    # Skip very small numbers.
    
    # Medium numbers
    # We'll do a few random samples and some structured ones.
    
    # Structured:
    # Around powers of 2?
    # Around 10^k?
    
    # Let's generate a list of (start, end) to process.
    # We will process them in order of increasing "interest" or just iterate through.
    
    # 1. Random samples in [10^6, 10^12]
    # 2. Specific regions
    
    # To maximize chance of finding a long run, we should look in regions where n is large.
    # Why? Because phi(n) grows with n. Distinctness is easier to maintain in log scale?
    # Actually, the probability of phi(a) = phi(b) is roughly 1/log(n) or similar.
    # Larger n means larger phi values, but the relative density of collisions might change.
    # Generally, larger n allows for longer runs before a collision is likely, simply because the range of phi values is larger?
    # No, the collision probability depends on the ratio.
    # But empirically, long runs are often found in mid-to-high ranges.
    
    # Let's use a mix.
    
    # Generate candidates for search
    candidates = []
    
    # 1. A few fixed checkpoints
    checkpoints = [
        (10**6, 10**6 + 100000),
        (10**7, 10**7 + 100000),
        (10**8, 10**8 + 100000),
        (10**9, 10**9 + 100000),
        (10**10, 10**10 + 100000),
        (10**11, 10**11 + 100000),
        (10**12 - 100000, 10**12),
    ]
    candidates.extend(checkpoints)
    
    # 2. Random samples
    # Number of random samples depends on time. Let's plan for ~50 samples of size 50k.
    # 50 * 50,000 = 2.5M operations * overhead. Should be fast.
    random.seed(42)
    for _ in range(100):
        # Pick a random start
        # Avoid very small numbers where collisions are frequent?
        # Min start 10^5
        s = random.randint(10**5, 10**12 - 100000)
        e = s + 100000
        candidates.append((s, e))
    
    # Sort candidates? Maybe by start? Or just process.
    # Processing in random order might be better to ensure coverage if we run out of time.
    # But structured checkpoints are high value.
    # Let's interleave or just process structured first, then random.
    
    # Shuffle random ones to break correlation
    random.shuffle(candidates[7:])
    
    for s, e in candidates:
        if time.time() > deadline:
            break
            
        if e > 10**12:
            e = 10**12
        if s < 1:
            s = 1
        if e < s:
            continue
            
        # Compute phi
        # Limit segment size to avoid timeout if something goes wrong with large primes?
        # 100k is safe.
        if e - s > 200000: 
            # If a window is too big, skip or resize? 
            # Our candidates are 100k, so this shouldn't happen.
            continue
        
        try:
            phi_vals = compute_phi_segment(s, e)
            length, start_idx = find_max_distinct_run(phi_vals, s)
            
            if length > best_length:
                best_length = length
                best_start = start_idx
        except Exception:
            # In case of any error, skip this window
            continue
            
        # Early exit optimization? 
        # If we find length 200 (max allowed), we can stop? 
        # No, we must verify if we can find > 200? 
        # Task says k <= 200. "Maximize k". 
        # If we find 200, we can't do better within the constraint. 
        # But wait, "1 <= k <= 200". 
        # If we find a run of length 200, that is the maximum possible score.
        # We should return immediately.
        if best_length >= 200:
            break

    # Ensure constraints
    if best_start < 1:
        best_start = 1
    if best_start + best_length - 1 > 10**12:
        best_length = 10**12 - best_start + 1
    if best_length < 1:
        best_length = 1
    if best_length > 200:
        best_length = 200
        
    return {'start': best_start, 'length': best_length}

# Helper for testing or manual invocation if needed, but construct() is the interface.
if __name__ == "__main__":
    res = construct()
    print(res)
