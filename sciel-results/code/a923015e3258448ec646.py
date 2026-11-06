import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i, v in enumerate(is_prime) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin
    
    # Precompute primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    best_k = 1
    best_s = 1
    
    # Block size for segmented phi sieve
    # Larger block -> more phi computations per sieve setup, but more sliding window checks
    # Smaller block -> more sieve setups, but less memory
    # 10^6 is a good balance. Memory for 10^6 ints is ~8MB (if using list of ints) or 4MB (if using array).
    # We will use a list of ints for phi values.
    block_size = 10**6
    
    # We will search for k=200 primarily.
    # If we find a run of length 200, we return it.
    # We also track the best run found so far.
    
    # To maximize the chance of finding a long run, we should sample random blocks.
    # However, structured scanning might be better if there are patterns?
    # Unlikely for phi. Random is safer.
    
    # We can also try to "polish" a region if we find a moderately long run (e.g., 50).
    # If we find a run of 50 at s, we can check s-1, s+1, etc. to extend it.
    
    def compute_phi_block(start, size, primes):
        """
        Compute phi(n) for n in [start, start+size-1].
        Returns a list of phi values.
        """
        phi = [0] * size
        for i in range(size):
            phi[i] = start + i
        
        for p in primes:
            if p * p > start + size - 1:
                # We still need to handle remaining prime factors?
                # No, the standard segmented sieve for phi works by:
                # 1. Initialize phi[i] = start+i
                # 2. For each prime p, for each multiple m of p in the block:
                #    phi[m] -= phi[m] // p
                # This correctly accounts for all prime factors <= sqrt(max_n).
                # What about prime factors > sqrt(max_n)?
                # If n has a prime factor q > sqrt(n), then q appears only once in the factorization.
                # The loop above handles all primes p <= sqrt(start+size-1).
                # After the loop, if phi[i] > 1, it means there is a remaining prime factor?
                # No, that's for factorization.
                # For phi, the formula is phi(n) = n * product(1 - 1/p) for all prime factors p.
                # The segmented sieve updates phi[i] by multiplying by (1-1/p) for each p dividing i.
                # If i has a prime factor q > sqrt(start+size-1), it will not be touched by the loop.
                # However, if q divides i, and q > sqrt(i), then i = q * k where k < sqrt(i).
                # If k=1, i=q (prime). Then phi(q) = q-1.
                # In our initialization, phi[i] = q.
                # The loop doesn't touch it because no p <= sqrt(q) divides q.
                # So phi[i] remains q. This is WRONG. phi(q) should be q-1.
                # We need to check for remaining prime factors.
                # If after the sieve, phi[i] == start+i, it might be prime?
                # No, phi[i] is reduced.
                # Let's trace:
                # n = 10. p=2: phi[10] = 10 - 10//2 = 5. p=5: 5*5 > 10, loop stops for p=5?
                # No, loop iterates over ALL primes in `primes`.
                # If p=5, and 5 divides 10, phi[10] = 5 - 5//5 = 4.
                # So it works.
                # What if n is prime? n=11.
                # p=2, 3. No divisors. phi[11] remains 11.
                # But phi(11) = 10.
                # So we MUST handle the case where the number itself is a prime factor remaining.
                # Standard segmented phi sieve:
                # After processing all p <= sqrt(max), if phi[i] > 1, it does NOT mean there is a remaining prime factor.
                # Wait.
                # If n is prime, phi[n] = n-1.
                # If n is composite, all its prime factors are <= sqrt(n) OR one is > sqrt(n) and the other is < sqrt(n).
                # If one is > sqrt(n), say q, then n = p*q with p < q.
                # p is processed. phi[n] becomes n*(1-1/p) = q.
                # q is not processed (if q > sqrt(max_block)).
                # So phi[n] remains q.
                # But correct phi[n] = n*(1-1/p)*(1-1/q) = q*(1-1/q) = q-1.
                # So we are missing the factor (1-1/q).
                # How to detect q?
                # If after the sieve, phi[i] != 1, and we haven't "divided out" the remaining prime?
                # Actually, the value phi[i] at the end of the sieve is n * product(1-1/p) for p <= sqrt(n).
                # If n has a prime factor q > sqrt(n), then n = p1...pk * q.
                # The product is over p1...pk.
                # So phi[i] = q * (product of (1-1/p) for p|n, p<=sqrt(n)).
                # This is not simply q.
                # Example: n = 15. sqrt(15) ~ 3.8. Primes 2, 3.
                # p=3 divides 15. phi[15] = 15 - 15//3 = 10.
                # Remaining factor 5.
                # Correct phi(15) = 8.
                # My phi[15] is 10.
                # So I need to apply (1-1/5) to 10? 10 * 4/5 = 8.
                # How do I know 5 is the factor?
                # It is not obvious from phi[i] alone.
                
                # Alternative:
                # We can just compute phi(n) individually for the block using the precomputed primes.
                # Since block size is 10^6, and we have primes up to 10^6, we can factor each n.
                # This is O(M * number_of_prime_factors).
                # Average number of prime factors is small (~log log n).
                # This might be faster and simpler than fixing the segmented sieve edge cases.
                # Let's switch to individual phi calculation for the block, but optimized.
                pass

        # Revised plan for compute_phi_block:
        # Use the precomputed primes to factor each n in the block.
        # This is safe and correct.
        
        phi = [0] * size
        for i in range(size):
            n = start + i
            temp = n
            result = n
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    result -= result // p
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                result -= result // temp
            phi[i] = result
        return phi

    # Sliding window check
    def find_max_run_in_block(phi_block, max_k=200):
        """
        Find the maximum run of distinct phi values in the block.
        Returns (start_index, length).
        """
        n = len(phi_block)
        best_len = 0
        best_start = 0
        
        # We only care about runs up to max_k
        # Use a dictionary or array to count frequencies
        # phi values can be up to 10^12, so dict is best.
        
        # Sliding window of size max_k
        # If we find a run of max_k, we can return immediately.
        
        # Optimization: 
        # We can slide a window of size max_k.
        # If all distinct, we have a run of max_k.
        # If not, we shrink or move.
        
        # Actually, we want the LONGEST run.
        # Standard sliding window for longest substring with distinct chars.
        
        counts = {}
        left = 0
        current_max_len = 0
        
        for right in range(n):
            val = phi_block[right]
            counts[val] = counts.get(val, 0) + 1
            
            while counts[val] > 1:
                left_val = phi_block[left]
                counts[left_val] -= 1
                if counts[left_val] == 0:
                    del counts[left_val]
                left += 1
                
            current_len = right - left + 1
            if current_len > current_max_len:
                current_max_len = current_len
                best_start = left
                if current_max_len >= max_k:
                    return best_start, max_k # We found a run of at least max_k
            
        return best_start, current_max_len

    # Search loop
    # We will test random blocks.
    # Number of blocks we can test?
    # Each block of 10^6 takes ~0.5-1s for phi computation?
    # Let's estimate. 10^6 iterations * ~10 ops = 10^7 ops. Python is ~10^6-10^7 ops/s.
    # So ~1-10s per block.
    # In 115s, we can do ~10-100 blocks.
    # This covers 10^7 to 10^8 numbers.
    # Probability of a run of 200 in 10^8 numbers?
    # If p is prob of run of 200 at a specific spot, we need p * 10^8 > 1.
    # If p is 10^-8, we have a chance.
    
    # To improve speed, we can use NumPy for the phi computation?
    # NumPy doesn't have built-in phi.
    # But we can vectorize the "divide by prime" step?
    # It's complex.
    
    # Let's try to speed up the individual phi calculation.
    # Precomputing smallest prime factor (SPF) up to 10^6?
    # No, n is up to 10^12.
    
    # Let's stick to the individual calculation but optimize the loop.
    # We can break early if temp becomes 1.
    
    # Also, we can skip blocks that are "bad" quickly?
    # No, we must compute phi to know.
    
    # Let's reduce block size to 10^5 to allow more random samples?
    # Overhead of setup is low.
    # 10^5 phi computations is ~0.1s.
    # 115s / 0.1s = 1150 blocks.
    # Total numbers scanned: 1150 * 10^5 = 1.15 * 10^8.
    # Similar coverage, but more random jumps.
    
    # Let's use block_size = 200_000.
    
    block_size = 200_000
    random.seed(42)
    
    # We will iterate for a fixed number of blocks or until time runs out.
    # We prioritize finding k=200.
    
    for _ in range(1000): # Max blocks
        if time.time() > deadline:
            break
            
        # Choose a random start
        # Ensure start + block_size - 1 <= 10^12
        max_start = 10**12 - block_size + 1
        if max_start < 1:
            break
        start = random.randint(1, max_start)
        
        # Compute phi block
        phi_block = compute_phi_block(start, block_size, primes)
        
        # Find max run
        s_rel, k_rel = find_max_run_in_block(phi_block, 200)
        
        if k_rel > best_k:
            best_k = k_rel
            best_s = start + s_rel
            if best_k == 200:
                return {'start': best_s, 'length': best_k}
                
    # Fallback
    if best_k >= 1:
        return {'start': best_s, 'length': best_k}
        
    return {'start': 2, 'length': 2}
