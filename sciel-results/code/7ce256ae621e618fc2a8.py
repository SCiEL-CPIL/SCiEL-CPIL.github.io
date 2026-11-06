import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    def now():
        return time.time()

    # Primes up to 10^6
    LIMIT = 10**6
    is_prime = np.ones(LIMIT + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    primes = primes[primes >= 2]

    def segmented_phi(L, R):
        """
        Compute phi(n) for n in [L, R] using NumPy.
        Returns an array of size R-L+1.
        """
        n = R - L + 1
        # Initialize phi with the numbers themselves
        phi = np.arange(L, R + 1, dtype=np.int64)
        
        # Sieve with primes
        for p in primes:
            if p * p > R:
                break
            # Find first multiple of p in [L, R]
            first = ((L + p - 1) // p) * p
            if first > R:
                continue
            # Indices of multiples in the array
            # The step in the array is p
            indices = np.arange(first - L, n, p)
            # Update phi for these indices
            phi[indices] = phi[indices] // p * (p - 1)
            
        # Handle remaining large prime factors
        # For each i, if phi[i] is still divisible by some prime > sqrt(R), 
        # actually the logic above handles all prime factors p <= sqrt(R).
        # If n has a prime factor q > sqrt(R), it appears only once.
        # The value phi[i] currently is n * prod(1-1/p) for p <= sqrt(R).
        # Let n = q * k. phi[i] = q * k * prod(1-1/p).
        # We need to multiply by (1-1/q).
        # How to find q? 
        # We can't easily find q from phi[i] alone without factorization.
        # BUT, we can keep a copy of the original numbers and divide out the small primes.
        
        # Let's re-implement with a 'remaining' array to be safe.
        # This doubles memory but ensures correctness.
        
    def segmented_phi_correct(L, R):
        n = R - L + 1
        phi = np.arange(L, R + 1, dtype=np.int64)
        rem = np.arange(L, R + 1, dtype=np.int64)
        
        for p in primes:
            if p * p > R:
                break
            first = ((L + p - 1) // p) * p
            if first > R:
                continue
            indices = np.arange(first - L, n, p)
            # Update phi
            phi[indices] = phi[indices] // p * (p - 1)
            # Divide rem by p as much as possible
            # Vectorized division is tricky with "while divisible".
            # We can do:
            # rem[indices] = rem[indices] // p
            # But if p^2 divides, we need to do it again.
            # Since p is prime, and we process each p once, 
            # we can just do a loop or use a trick.
            # Actually, for phi calculation, we only need to apply the factor (1-1/p) once per prime p.
            # The `rem` array is only needed to detect if there's a LARGER prime factor left.
            # If we just divide rem by p once, rem might still be divisible by p.
            # But we only care if rem > 1 at the end.
            # If rem > 1 at the end, is it prime?
            # If n = p * q * ...
            # We divide by all small primes.
            # If the remaining factor is composite, it must be product of primes > sqrt(R).
            # But a number <= R can have at most one prime factor > sqrt(R).
            # So the remaining factor MUST be prime.
            
            # So, we just need to divide rem by p once?
            # No, if n = p^2, rem becomes p. Then rem > 1.
            # We would think p is a large prime factor.
            # But p is already processed.
            # So we must divide rem by p until not divisible.
            
            # Efficient way to divide by p while divisible in NumPy:
            # It's hard to do in one vectorized op.
            # Given the constraint, let's stick to the property:
            # We only need to know if there is a prime factor > sqrt(R).
            # If we divide rem by p once, and p^2 | n, rem is still divisible by p.
            # We can just do:
            # while any(rem[indices] % p == 0): ...
            # This might be slow.
            
            # Alternative: Just use the standard property.
            # phi(n) = n * prod(1-1/p).
            # The sieve above computes exactly this for all p <= sqrt(R).
            # If n has a prime factor q > sqrt(R), it is not in `primes`.
            # So the term (1-1/q) is missing.
            # We need to find q.
            # q = n / (product of small prime powers).
            # This is effectively the `rem` array.
            
            # Let's optimize the division.
            # For most numbers, they are not divisible by p^2.
            # We can just do:
            rem[indices] = rem[indices] // p
            # And check if any are still divisible?
            # If n is large, p is small. p^2 is likely not a factor for most.
            # To be safe and fast, we can do a second pass for squares?
            # Or just accept that if rem is still divisible by p, we missed a factor?
            # No, if we miss a factor in `rem`, we might incorrectly identify a composite as prime?
            # No, the `rem` is used to find the LARGE prime factor.
            # If we don't fully divide by small p, `rem` will be larger.
            # If `rem` is composite, it's a problem.
            # But `rem` can only contain primes > sqrt(R) if we divided out all small factors completely.
            
            # Let's assume we divide once.
            # If n = 4, L=1, R=4. p=2.
            # indices for 2, 4.
            # phi[2] = 2//2*1 = 1. rem[2] = 2//2 = 1.
            # phi[4] = 4//2*1 = 2. rem[4] = 4//2 = 2.
            # End of loop for p=2.
            # rem[4] is 2. 2 > 1.
            # We think 2 is a large prime factor.
            # But 2 was already processed!
            # So we must divide fully.
            
            # Efficient full division:
            # We can do a loop inside the vectorized block? No.
            # We can use:
            # while True:
            #    mask = rem[indices] % p == 0
            #    if not mask.any(): break
            #    rem[indices[mask]] = rem[indices[mask]] // p
            # This is safe.
            
        # After loop, check for remaining primes
        # For each i, if rem[i] > 1, it is a prime factor q.
        # We need to update phi[i] = phi[i] // q * (q-1)
        # q is rem[i].
        # But wait, if n = q, rem[i] = q.
        # If n = q * k, rem[i] = q.
        # So we can just use rem[i] as q.
        
        mask = rem > 1
        if np.any(mask):
            # q = rem[mask]
            # phi[mask] = phi[mask] // q * (q - 1)
            # Be careful with integer division.
            # phi is already n * prod_small(1-1/p).
            # We need to multiply by (1-1/q).
            # phi_new = phi * (q-1) / q.
            # Since phi is divisible by q?
            # n is divisible by q.
            # phi = n * prod(1-1/p).
            # Is phi divisible by q?
            # If p != q, (1-1/p) doesn't introduce q in denominator.
            # So yes, phi is divisible by q.
            q = rem[mask]
            phi[mask] = phi[mask] // q * (q - 1)
            
        return phi

    best_k = 1
    best_s = 1
    
    # Scan some initial ranges
    # We'll scan windows of size 100,000
    W = 100000
    
    # Strategic points
    points = [10**6, 10**7, 10**8, 10**9, 10**10, 10**11, 10**12]
    
    for p in points:
        if now() > DEADLINE: break
        L = max(1, p - W//2)
        R = min(10**12, p + W//2)
        phi_vals = segmented_phi_correct(L, R)
        
        # Find best run
        seen = set()
        curr_k = 0
        curr_start = L
        for i, val in enumerate(phi_vals):
            if val in seen:
                if curr_k > best_k:
                    best_k = curr_k
                    best_s = curr_start
                seen.clear()
                curr_k = 0
                curr_start = L + i
            else:
                if curr_k == 0:
                    curr_start = L + i
                seen.add(val)
                curr_k += 1
        if curr_k > best_k:
            best_k = curr_k
            best_s = curr_start
            
    # Random search
    # Use a fixed seed for reproducibility? No, just random.
    import random
    random.seed(42)
    
    while now() < DEADLINE - 2:
        # Pick a random start
        # Bias towards larger numbers?
        # Let's just pick uniformly in [1, 10^12 - W]
        L = random.randint(1, int(10**12) - W)
        R = L + W - 1
        
        phi_vals = segmented_phi_correct(L, R)
        
        seen = set()
        curr_k = 0
        curr_start = L
        for i, val in enumerate(phi_vals):
            if val in seen:
                if curr_k > best_k:
                    best_k = curr_k
                    best_s = curr_start
                seen.clear()
                curr_k = 0
                curr_start = L + i
            else:
                if curr_k == 0:
                    curr_start = L + i
                seen.add(val)
                curr_k += 1
        if curr_k > best_k:
            best_k = curr_k
            best_s = curr_start
            
    return {'start': best_s, 'length': best_k}
