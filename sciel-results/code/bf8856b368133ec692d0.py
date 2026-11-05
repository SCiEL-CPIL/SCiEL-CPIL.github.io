import math
import time
import random

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 140s usage
    DEADLINE = start_time + 140.0
    
    # 1. Precompute primes up to 10^6
    limit_p = 10**6
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i, v in enumerate(is_prime) if v]
    
    W = 200
    max_s = 10**12 - W + 1
    
    # Segmented phi calculation for a window [start, start+W-1]
    def segmented_phi(start):
        # nums[i] will hold the remaining factor part of start+i
        # phis[i] will hold the phi value of start+i
        nums = [start + i for i in range(W)]
        phis = [start + i for i in range(W)]
        
        # Process each prime
        for p in primes:
            # We only need to process p if p <= sqrt(start+W-1)
            # Actually, we can break early if p*p > start+W-1, 
            # but we still need to handle the remaining prime factors > 1.
            # The loop below handles multiples of p in the range.
            # If p is larger than the range width, it can appear at most once.
            
            # Find first multiple of p in [start, start+W-1]
            # m = ceil(start / p) * p
            m = ((start + p - 1) // p) * p
            
            if m > start + W - 1:
                # No multiple of p in this window
                # But if p*p > start+W-1, we can break out of the prime loop?
                # Not exactly, because p might be a factor of a number in the window
                # even if p > sqrt(max_num) is false? 
                # If p > sqrt(max_num), then p can divide at most one number in the window?
                # No, if p > sqrt(max_num), then p * p > max_num.
                # So p can divide n only if n = p * k where k < p.
                # Since n < p*p, k < p.
                # The standard sieve logic handles this:
                # We iterate p. If p*p > max_num, we stop the outer loop?
                # No, we must process all p that are <= max_num?
                # No, for phi calculation, we only need to divide by prime factors.
                # If we process all p <= sqrt(max_num), the remaining factor is 1 or prime.
                # So we can break if p*p > start+W-1.
                if p * p > start + W - 1:
                    break
                continue

            for x in range(m, start + W, p):
                idx = x - start
                if nums[idx] % p == 0:
                    phis[idx] -= phis[idx] // p
                    while nums[idx] % p == 0:
                        nums[idx] //= p

        # Handle remaining prime factors
        for i in range(W):
            if nums[i] > 1:
                phis[i] -= phis[i] // nums[i]
                
        return phis

    best_k = 1
    best_s = 1
    
    # Phase 1: Try a few specific starts to see if 200 is easy
    # Large numbers tend to have more distinct phi values?
    # Let's try random sampling.
    
    # To ensure we find a valid window, we sample random starts.
    # The probability of collision in 200 numbers is very low.
    # We expect to find one in the first few attempts.
    
    attempts = 0
    while time.time() < DEADLINE:
        # Random start
        s = random.randint(1, max_s)
        
        phis = segmented_phi(s)
        
        # Check distinctness
        if len(set(phis)) == W:
            return {'start': s, 'length': W}
            
        attempts += 1
        
        # If we've tried many times, maybe try structured search?
        # Or just keep going.
        if attempts % 1000 == 0:
            # Log progress if needed, but no stdout.
            pass

    # If we didn't find 200, we might have found a shorter run in a failed window?
    # The task asks to maximize k.
    # If we failed to find 200, we should return the best k we found.
    # But we didn't track sub-windows.
    # Let's add a fallback: check sub-windows of the last computed phis?
    # Or just return 1?
    # Given the low collision probability, it's highly unlikely we fail to find 200.
    # But to be safe, let's implement a quick check for the best sub-run in the last window.
    
    # Actually, let's just return the best known.
    # We can improve the search to track the longest sub-run in each window.
    
    # Let's restart the search logic to track best sub-run.
    # This is safer.
    
    # We'll reuse the last computed phis if available, or recompute one.
    # For simplicity and speed, we'll just return 1 if the loop finishes without success.
    # But 200 should be found.
    
    # Let's add a quick check for the best sub-run in a random window as a fallback.
    # This is better than returning 1.
    
    # Generate one more window
    s = random.randint(1, max_s)
    phis = segmented_phi(s)
    
    # Find longest sub-run of distinct phis
    best_k_local = 1
    best_s_local = s
    current_k = 1
    seen = {phis[0]}
    for i in range(1, W):
        if phis[i] in seen:
            # Reset
            if current_k > best_k_local:
                best_k_local = current_k
                best_s_local = s + i - current_k
            seen = {phis[i]}
            current_k = 1
        else:
            seen.add(phis[i])
            current_k += 1
            if current_k > best_k_local:
                best_k_local = current_k
                best_s_local = s + i - current_k
                
    # Check end
    if current_k > best_k_local:
        best_k_local = current_k
        best_s_local = s + W - current_k

    if best_k_local > best_k:
        best_k = best_k_local
        best_s = best_s_local

    return {'start': best_s, 'length': best_k}

# EVOLVE-BLOCK-END
