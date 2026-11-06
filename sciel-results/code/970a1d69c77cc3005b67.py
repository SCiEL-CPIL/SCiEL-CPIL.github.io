import time
import math
import random
import numpy as np

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i, v in enumerate(is_prime) if v]

def phi_segment(start, length, primes):
    """Compute phi(n) for n in [start, start+length-1] using segmented sieve with NumPy."""
    # Use int64 to avoid overflow issues during intermediate steps, though phi(n) < n <= 10^12 fits in int64
    # Actually, phi(n) can be up to 10^12, which fits in int64 (max ~9*10^18).
    # But intermediate calculations in vectorized ops might be safer with object or careful casting.
    # Let's stick to standard numpy int64 as it's fast and sufficient for n <= 10^12.
    
    phi_vals = np.arange(start, start + length, dtype=np.int64)
    remaining = np.arange(start, start + length, dtype=np.int64)
    
    max_val = start + length - 1
    sqrt_max = int(math.isqrt(max_val))
    
    for p in primes:
        if p > sqrt_max:
            break
        
        # Find indices in the segment divisible by p
        # first multiple of p >= start
        first = ((start + p - 1) // p) * p
        
        if first >= start + length:
            continue
            
        # Indices in the local array [0, length)
        # m = start + idx
        # m is divisible by p => start + idx = k*p => idx = k*p - start
        # k*p ranges from first to start+length-1
        # So idx ranges from first-start to start+length-1-start
        # Step is p
        
        start_idx = first - start
        # end_idx is exclusive in slice notation, but we want all indices
        # The indices are start_idx, start_idx + p, start_idx + 2p, ...
        
        # Create array of indices
        idxs = np.arange(start_idx, length, p)
        
        # Update phi_vals: phi(n) = phi(n) * (1 - 1/p) = phi(n) - phi(n)//p
        # Vectorized operation
        phi_vals[idxs] -= phi_vals[idxs] // p
        
        # Update remaining: remove all factors of p
        # remaining[idxs] /= p while divisible
        # We can do this in a loop for each p, but vectorized modulo check is possible
        # However, "while" is hard to vectorize easily without loops over steps.
        # Since p is small for most iterations, the number of divisions is small (log_p(n)).
        # For large p, the number of divisions is 1.
        
        # Efficient way:
        # While any remaining[idxs] is divisible by p:
        #   remaining[idxs] /= p
        #   # Filter indices that are still divisible? No, we just divide all that are divisible.
        #   # But we need to know WHICH ones are still divisible.
        #   # Actually, if we just divide all, we might divide non-divisible ones?
        #   # No, we only touch idxs where remaining was divisible at the start of the loop iteration?
        #   # Wait, remaining[idxs] changes.
        
        # Let's do a simple loop for the division part, it's usually 1-2 iterations for large p.
        # For small p (2, 3, 5), it's more, but p is small so the number of such primes is small.
        
        mask = (remaining[idxs] % p == 0)
        while np.any(mask):
            remaining[idxs[mask]] //= p
            mask = (remaining[idxs[mask]] % p == 0)
            # Note: idxs[mask] creates a new array of indices into the 'idxs' array, 
            # but we need indices into 'remaining'.
            # Actually, idxs are indices into the main array.
            # So:
            # current_indices = idxs[mask]
            # remaining[current_indices] //= p
            # mask = (remaining[current_indices] % p == 0)
            
            # Let's re-implement this correctly:
            break # The above logic was flawed in the draft. Let's do it properly below.

    # Corrected division logic:
    # For each p:
    #   indices = [i for i in range(start_idx, length, p)]
    #   for i in indices:
    #       while remaining[i] % p == 0: remaining[i] //= p
    # This is O(length * log n) in worst case but constant factors are high in Python.
    # Vectorized approach for division:
    #   idxs = np.arange(start_idx, length, p)
    #   while True:
    #       mask = (remaining[idxs] % p == 0)
    #       if not np.any(mask): break
    #       remaining[idxs[mask]] //= p
    #       # The mask needs to be updated for the next iteration?
    #       # Yes, because remaining[idxs[mask]] changed.
    #       # But idxs is fixed. We just check the same indices again.
    #       # This works.
    
    # Let's rewrite phi_segment completely to be safe and fast.
    
    return phi_vals

def phi_segment_v2(start, length, primes):
    """Compute phi(n) for n in [start, start+length-1] using segmented sieve with NumPy."""
    phi_vals = np.arange(start, start + length, dtype=np.int64)
    remaining = np.arange(start, start + length, dtype=np.int64)
    
    max_val = start + length - 1
    sqrt_max = int(math.isqrt(max_val))
    
    for p in primes:
        if p > sqrt_max:
            break
        
        first = ((start + p - 1) // p) * p
        if first >= start + length:
            continue
        
        start_idx = first - start
        idxs = np.arange(start_idx, length, p)
        
        # Update phi
        phi_vals[idxs] -= phi_vals[idxs] // p
        
        # Update remaining
        # We need to divide by p as long as divisible
        # Since p is a prime, the max number of factors is log_p(n).
        # For p=2, max ~40. For p=3, ~25. For large p, 1.
        # We can just loop while any are divisible.
        mask = (remaining[idxs] % p == 0)
        while np.any(mask):
            remaining[idxs[mask]] //= p
            mask = (remaining[idxs[mask]] % p == 0)
            # Wait, idxs[mask] gives the positions in the original array that were divisible in the previous step.
            # We need to check if they are STILL divisible.
            # Yes, that's what the line does.
            # But we also need to consider if NEW indices become divisible? No, dividing by p only reduces the value.
            # So only the same indices can be divisible again.
            # So this logic is correct.
            
    # Handle remaining prime factors
    # remaining[i] > 1 implies it is a prime factor
    # phi(n) = phi(n) * (1 - 1/p) = phi(n) - phi(n)//p
    # Here p = remaining[i]
    mask = remaining > 1
    if np.any(mask):
        phi_vals[mask] -= phi_vals[mask] // remaining[mask]
        
    return phi_vals

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Leave 2s safety margin
    
    # Precompute primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    def is_valid_run(s, k):
        phi_vals = phi_segment_v2(s, k, primes)
        # Check if all distinct
        # Converting to set is O(k). k=200 is small.
        return len(set(phi_vals.tolist())) == k

    # Try multiple candidate positions for k=200
    # Heuristic: Start with odd numbers? 
    # phi(2n) = n * phi(2) ... no.
    # Let's just use random and structured search.
    
    candidates = set()
    
    # Near 10^12
    for offset in range(0, 5000):
        candidates.add(10**12 - 199 - offset)
        
    # Near small values
    for s in range(1, 5000):
        candidates.add(s)
        
    # Random positions
    random.seed(42)
    for _ in range(20000):
        s = random.randint(1, 10**12 - 199)
        candidates.add(s)
        
    # Sort candidates to make it deterministic? Or just iterate.
    # We want to find a valid one quickly.
    # Random is good for discovery.
    
    # Convert to list for iteration
    candidate_list = list(candidates)
    random.shuffle(candidate_list)
    
    # Try each candidate for k=200
    for s in candidate_list:
        if time.time() > deadline - 10:
            break
        if s < 1 or s + 199 > 10**12:
            continue
        if is_valid_run(s, 200):
            return {'start': s, 'length': 200}
            
    # If k=200 not found, try smaller k values
    # We can reuse the phi values if we computed them?
    # phi_segment_v2 computes for a specific length.
    # If we check k=200, we have phi for 200 numbers.
    # If we check k=199, we could use the first 199.
    # But our loop above breaks on success. If it fails, we didn't save the result.
    # Let's modify to store best result.
    
    # Actually, the current logic above just returns if found.
    # If not found, we try smaller k.
    # To be efficient, we should probably just stick to k=200 if we have time,
    # or quickly drop to 199, 198...
    
    # Let's try k=199 with new random candidates or the same ones?
    # The same ones might work for 199 if they failed for 200 due to the last element.
    # But checking 199 is cheaper.
    
    for k in range(199, 190, -1):
        if time.time() > deadline - 2:
            break
        # Try a subset of candidates again, or new ones?
        # Let's try the same candidates first.
        for s in candidate_list[:1000]:
            if s < 1 or s + k - 1 > 10**12:
                continue
            if is_valid_run(s, k):
                return {'start': s, 'length': k}
                
    # Fallback
    return {'start': 4, 'length': 2}
