import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin for 140s limit (or 150s per prompt, let's use 130 to be safe against 120s evaluator limit mentioned in prompt text "120 s" vs "150s" in directive. Prompt says "120 s" in contract, "150s" in directive. Let's assume 120s hard limit, so 110s budget.)
    deadline = start_time + 110
    
    # We want to find a run of length k <= 200 where phi(n) are distinct.
    # The seed found a run of length 9 (score 1.0 implies seed k=9? Or score is k/seed_k? 
    # "search score is value/seed". If seed program returns k=9, score=1.0 means value=9.
    # Wait, the seed program searches range(1, 20001). It finds best k.
    # Let's check what the seed finds.
    # phi(1)=1, phi(2)=1 (collision). 
    # phi(3)=2, phi(4)=2 (collision).
    # phi(5)=4, phi(6)=2 (collision with phi(3,4)).
    # Let's just run the logic mentally or assume the seed found a small k, likely 5-10.
    # Actually, the prompt says "Maximize k".
    # Known longest runs of distinct totients are quite short.
    # For n < 10^12, the maximum run length is likely around 10-15.
    # The seed searches up to 20000. It likely finds the global max for that range.
    # To beat the seed, we need to find a longer run in a larger range (up to 10^12).
    
    # Strategy:
    # 1. Generate primes up to sqrt(10^12) = 10^6.
    # 2. Use a segmented sieve approach to compute phi(n) for segments of size ~10^5.
    # 3. For each segment, check for runs of distinct phi values.
    # 4. Since we need pairwise distinctness in a window of size k, and k is small (<20),
    #    we can compute phi for a window and check.
    
    # However, computing phi for all n up to 10^12 is impossible.
    # We must search smartly.
    # Collisions are common. phi(n) = phi(m) happens often.
    # We want to avoid numbers that share factors with neighbors in a way that causes phi collision.
    
    # Let's first generate primes up to 10^6.
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Function to compute phi(n) using primes
    def phi(n):
        result = n
        y = n
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                while y % p == 0:
                    y //= p
                result -= result // p
        if y > 1:
            result -= result // y
        return result
    
    # We will search in segments.
    # Let's start from a large number? Or scan from 1?
    # The seed scanned 1..20000.
    # Let's try scanning larger ranges.
    # But scanning 10^12 is too slow.
    # We need to prune.
    
    # Observation: If n is even, phi(n) is even (for n>2).
    # If n is odd, phi(n) is even (for n>1).
    # phi(1)=1, phi(2)=1.
    # So phi(n) is always even for n>=3.
    # This means phi(n) and phi(n+1) are both even.
    # This doesn't force collision, but restricts values.
    
    # Let's use a segmented sieve to compute phi for a block.
    # Block size: 10^5.
    # For each block [L, R]:
    #   Create array phi_vals of size R-L+1, init to n.
    #   Create array remaining of size R-L+1, init to n.
    #   For each prime p:
    #     Find first multiple of p in [L, R].
    #     For each multiple m:
    #       if remaining[m-L] % p == 0:
    #         remaining[m-L] //= p
    #         while remaining[m-L] % p == 0: remaining[m-L] //= p
    #         phi_vals[m-L] -= phi_vals[m-L] // p
    #   After processing all primes, for any remaining[i] > 1, 
    #     phi_vals[i] -= phi_vals[i] // remaining[i]
    
    # This is O(N log log N) per block? No, sum over primes of N/p is N log log N.
    # With N=10^5, this is fast.
    # We can process many blocks.
    
    # How far can we go in 110s?
    # Let's test speed with a small block first.
    
    block_size = 10**5
    best_k = 1
    best_s = 1
    
    # We know seed found something in 1..20000.
    # Let's start searching from 20000 upwards.
    # Actually, let's start from 1 to be safe, but skip 1..20000 if we trust seed?
    # No, seed might have found a local max. Let's restart search but faster.
    
    # To speed up, we can use the fact that we only care about runs of length > best_k.
    # Initially best_k is whatever the seed found. Let's assume seed found ~9.
    # We can quickly verify the seed's result to set baseline.
    
    # Let's just run the segmented sieve from 1.
    
    current_start = 1
    # We'll process blocks.
    # For each block, we compute phi for all n in block.
    # Then we scan for max run of distinct phi values.
    
    # Optimization: 
    # If we find a run of length k, we update best.
    # We can stop if we find k=200 (max allowed).
    
    # Let's implement the segmented phi sieve.
    
    def compute_phi_block(L, R, primes):
        size = R - L + 1
        phi_vals = [0] * size
        remaining = [0] * size
        for i in range(size):
            phi_vals[i] = L + i
            remaining[i] = L + i
        
        for p in primes:
            if p > R:
                break
            # Find first multiple of p >= L
            start = ((L + p - 1) // p) * p
            for m in range(start, R + 1, p):
                idx = m - L
                if remaining[idx] % p == 0:
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
                    phi_vals[idx] -= phi_vals[idx] // p
        
        # Handle remaining prime factors
        for i in range(size):
            if remaining[i] > 1:
                phi_vals[i] -= phi_vals[i] // remaining[i]
        
        return phi_vals
    
    # Main loop
    L = 1
    while L <= 10**12:
        if time.time() > deadline:
            break
        
        R = min(L + block_size - 1, 10**12)
        phi_vals = compute_phi_block(L, R, primes)
        
        # Find max run of distinct values in phi_vals
        # We need pairwise distinct in the window.
        # Since k is small, we can slide a window.
        # But checking pairwise distinct for a window of size k is O(k^2) or O(k) with set.
        # We want the LONGEST run.
        # Let's use a sliding window with a frequency map?
        # Or simply, for each start i, extend j while phi_vals[i..j] are distinct.
        # Since we expect k to be small (<20), this is fast.
        
        for i in range(size):
            # Early exit if max possible length from i is <= best_k
            if size - i <= best_k:
                break
            
            seen = set()
            k = 0
            # We can limit k to 200
            max_k_to_check = min(200, size - i, best_k + 10) # Check a bit further
            
            for j in range(i, i + max_k_to_check):
                v = phi_vals[j]
                if v in seen:
                    break
                seen.add(v)
                k += 1
            
            if k > best_k:
                best_k = k
                best_s = L + i
                if best_k >= 200:
                    return {'start': best_s, 'length': best_k}
        
        L = R + 1
    
    return {'start': best_s, 'length': best_k}
