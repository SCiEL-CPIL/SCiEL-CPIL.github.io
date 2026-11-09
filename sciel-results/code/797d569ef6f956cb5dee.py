import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # We will search up to N. 
    # A sieve of 20 million takes ~1-2s. 
    # A linear scan of 20 million takes ~1-2s.
    # Total time ~5s, well within budget.
    # We can try to go higher if time permits, but 20M is a solid baseline for finding k > 30.
    N = 20000000
    
    # 1. Compute Euler's Totient function phi(n) for n in [1, N]
    # Using a standard sieve method
    phi = list(range(N + 1))
    
    # Optimization: Only iterate up to sqrt(N) for primes? 
    # No, the standard sieve iterates i from 2 to N. 
    # If phi[i] == i, i is prime.
    # This is O(N log log N).
    
    # To speed up Python loops, we can use a slightly different approach or just rely on the fact that N=20M is manageable.
    # Let's use the standard loop.
    
    # Alternative: Use numpy for faster vectorized operations if available and suitable.
    # However, the sieve logic is inherently sequential in terms of prime detection.
    # A pure Python sieve for 20M might take 2-4 seconds.
    
    # Let's stick to pure Python for reliability but optimize the inner loop.
    
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    # Check time after sieve
    if time.time() - start_time > deadline - 5:
        # If sieve took too long, return a small valid result
        # But 20M sieve is fast. This is just a safety net.
        return {'start': 1, 'length': 1}

    # 2. Find the longest run of consecutive integers with distinct totients
    # We use a sliding window approach.
    # Window [l, r] contains indices with distinct phi values.
    # We maintain a count array for phi values in the current window.
    # Since phi(n) <= n, we can use an array of size N+1.
    
    counts = [0] * (N + 1)
    
    best_k = 1
    best_s = 1
    
    l = 1
    r = 1
    
    # We iterate r from 1 to N
    # For each r, we add phi[r] to the window.
    # If phi[r] is already in the window (counts[phi[r]] > 0), we must move l forward
    # until the previous occurrence of phi[r] is removed from the window.
    
    for r in range(1, N + 1):
        val = phi[r]
        
        # If val is already in the window, shrink from the left
        if counts[val] > 0:
            # Move l until we remove the previous occurrence of val
            while l < r:
                left_val = phi[l]
                counts[left_val] -= 1
                l += 1
                if left_val == val:
                    break
        
        # Now val is not in the window (or it was the one we just removed, but counts is 0 now)
        counts[val] += 1
        
        current_k = r - l + 1
        if current_k > best_k:
            best_k = current_k
            best_s = l
            
        # Early termination if we hit the max allowed k
        if best_k >= 200:
            break
            
        # Time check
        if time.time() - start_time > deadline - 5:
            break

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
