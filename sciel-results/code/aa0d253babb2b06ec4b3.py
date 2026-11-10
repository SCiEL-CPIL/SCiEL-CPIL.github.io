import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def sieve_primes(limit):
        """Sieve of Eratosthenes to find all primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray([1]) * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_numpy(lo, hi, primes):
        """
        Compute phi(n) for all n in [lo, hi] using NumPy.
        """
        n = hi - lo + 1
        if n <= 0:
            return np.array([])
        
        # Initialize phi and remaining arrays with numbers themselves
        # Using int64 to handle values up to 10^12
        phi = np.arange(lo, hi + 1, dtype=np.int64)
        remaining = np.arange(lo, hi + 1, dtype=np.int64)
        
        # Handle phi(1) = 1 if lo == 1
        if lo == 1:
            phi[0] = 1
            remaining[0] = 1
            
        # For each prime p, update phi and remaining for multiples of p
        for p in primes:
            if p > hi:
                break
            
            # Find the first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            if start > hi:
                continue
                
            # Indices in the array corresponding to multiples of p
            # We need to iterate over indices or use vectorized slicing
            # Slicing with step is efficient in NumPy
            indices = np.arange(start, hi + 1, p)
            idx_arr = indices - lo
            
            # Check which of these remaining values are divisible by p
            # remaining[idx_arr] % p == 0
            mask = (remaining[idx_arr] % p == 0)
            
            if not np.any(mask):
                continue
            
            # Get the actual indices where the condition holds
            valid_indices = idx_arr[mask]
            
            # Update phi: phi[n] = phi[n] * (p-1) / p
            # To avoid floating point issues, do integer division carefully
            # phi = phi - phi // p
            phi[valid_indices] -= phi[valid_indices] // p
            
            # Update remaining: divide by p as many times as possible
            # This is harder to vectorize perfectly without a loop, 
            # but we can do it in a loop over the valid indices if the count is small,
            # or use a while loop on the array.
            # However, for a segmented sieve, the number of multiples is N/p.
            # For small p, this is large.
            # A pure Python loop over 10^6 elements is slow.
            # But we can do:
            # while any(remaining[valid_indices] % p == 0):
            #    mask2 = remaining[valid_indices] % p == 0
            #    remaining[valid_indices[mask2]] //= p
            # This is still potentially slow if p is small and we have many multiples.
            
            # Alternative: Just do the division once in the vectorized step?
            # No, we must remove all factors of p.
            
            # Let's optimize the "remaining" update.
            # For small p, the number of multiples is large.
            # We can use a while loop on the subset.
            # Actually, for p=2, there are N/2 multiples. 
            # The inner while loop for each element is fast if unrolled? No.
            
            # Let's stick to a simple loop for the "remaining" part if it's not too slow,
            # or realize that we only need `remaining` to be 1 or a prime > sqrt(hi).
            # We can just divide by p once in the vectorized step, and then 
            # handle the rest? No, that's incorrect.
            
            # Better approach for NumPy:
            # We can just do the division in a loop over the valid indices if the count is manageable.
            # For p=2, count is 500,000. A Python loop of 500k is ~0.05s.
            # For p=3, count is 333,333.
            # Sum over primes 1/p is log log N.
            # Total operations ~ N log log N.
            # For N=10^6, this is ~ 10^6 * 3 = 3 * 10^6 operations.
            # In Python, this might take 1-2 seconds.
            
            # Let's try to vectorize the "remove all factors" part.
            # It's tricky. 
            # However, note that we only care about phi.
            # The `remaining` array is only used to detect if a prime factor > sqrt(hi) remains.
            # We can simplify:
            # 1. Update phi for all multiples of p (vectorized).
            # 2. Divide remaining by p for all multiples of p (vectorized, once).
            # 3. Repeat? No.
            
            # Actually, the standard segmented sieve in NumPy often just does:
            # phi[idx] -= phi[idx] // p
            # remaining[idx] //= p
            # and then checks if remaining[idx] > 1 at the end.
            # BUT, if remaining[idx] is divisible by p again, we haven't removed it.
            # However, if we iterate through primes in increasing order, 
            # and we divide `remaining` by p as many times as possible, 
            # we ensure that no small prime factor remains.
            
            # Let's use a while loop on the array slice.
            # This is O(k log p) where k is the number of multiples.
            
            # Optimization: Only do the full removal for small p?
            # Actually, for the purpose of finding distinct phi values, 
            # exact factorization is required by the prompt? 
            # "The values phi(s), ..., phi(s + k - 1) must be pairwise distinct (exact factorization)."
            # This implies we must compute phi correctly.
            
            # Let's use a hybrid: Vectorized update for phi, and a loop for remaining.
            # To speed up the loop, we can use NumPy's `np.where` or similar?
            
            # Let's try a different strategy: 
            # Since we are targeting large n where collisions are rare, 
            # we don't necessarily need to scan the *entire* 10^6 block.
            # We can just compute phi for a smaller block, say 10,000 numbers.
            # 10,000 numbers is very fast even with Python loops.
            # And 10,000 >> 200, so we will find a run of 200 if it exists.
            
            # Let's reduce N to 10,000 or 50,000 to make the Python loops acceptable.
            # If N=50,000, the sieve is very fast.
            
            pass # This block is replaced by the actual implementation below

        return phi

    # Precompute primes up to sqrt(10^12) = 10^6
    primes = sieve_primes(10**6)
    
    # We will try a large base to avoid collisions
    # Base 10^11
    base = 10**11
    # Block size 50,000
    block_size = 50000
    
    lo = base
    hi = base + block_size - 1
    
    # Ensure hi <= 10^12
    if hi > 10**12:
        hi = 10**12
        lo = hi - block_size + 1
        if lo < 1:
            lo = 1
            hi = min(hi, lo + block_size - 1)

    # Compute phi using a Python-based segmented sieve for small blocks
    # This is simpler and fast enough for 50k numbers
    
    def compute_phi_py(lo, hi, primes):
        n = hi - lo + 1
        phi = list(range(lo, hi + 1))
        remaining = list(range(lo, hi + 1))
        
        if lo == 1:
            phi[0] = 1
            remaining[0] = 1
            
        for p in primes:
            if p > hi:
                break
            start = ((lo + p - 1) // p) * p
            for j in range(start, hi + 1, p):
                idx = j - lo
                if remaining[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
                        
        for i in range(n):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]
                
        return phi

    if time_left() < 10.0:
         # Fallback to small range if time is tight
         lo, hi = 1, 10000
         block_size = 10000

    phi_arr = compute_phi_py(lo, hi, primes)
    
    # Find the longest run of distinct values
    # Sliding window
    best_k = 1
    best_s = lo
    seen = {}
    left = 0
    
    for right in range(len(phi_arr)):
        val = phi_arr[right]
        seen[val] = seen.get(val, 0) + 1
        
        while seen[val] > 1:
            left_val = phi_arr[left]
            seen[left_val] -= 1
            left += 1
            
        current_k = right - left + 1
        if current_k > best_k:
            best_k = current_k
            best_s = lo + left
            
        # Cap at 200
        if best_k >= 200:
            break
            
    # Ensure validity
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
        
    return {'start': best_s, 'length': best_k}
