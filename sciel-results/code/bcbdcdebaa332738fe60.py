"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

MAX_N = 10_000_000
TARGET_NUM, TARGET_DEN = 3, 2
DEADLINE = 85.0  # safety margin before 90s kill

def construct():
    t_start = time.time()
    
    # 1. Sieve for smallest prime factor (SPF) to enable fast factorization
    #    and to compute sigma(n) for all n up to MAX_N efficiently.
    #    Actually, for this specific problem, we can be smarter.
    #    We want to minimize |a/b - 3/2| subject to sigma(a) = sigma(b).
    #    Let a/b = 3/2 + eps. We want small eps.
    #    This implies 2a - 3b = k, where k is a small integer.
    #    a = (3b + k) / 2. For a to be integer, 3b+k must be even.
    #    Since 3 is odd, b and k must have the same parity.
    
    # Strategy:
    # Iterate over small k (the numerator of the error term).
    # For each k, find b such that a = (3b+k)/2 is integer, <= MAX_N,
    # and sigma(a) == sigma(b).
    # To do this efficiently, we need sigma(n) for many n.
    # Precomputing sigma(n) for all n up to MAX_N is O(N log N) or O(N) with linear sieve.
    # N = 10^7. Linear sieve is fast enough in C, but in Python it might be tight.
    # Let's try a modified sieve to compute sigma(n) directly.
    
    # sigma(n) is multiplicative.
    # We can use a sieve to compute sigma(n) for all n.
    # Initialize sigma = [0]*(MAX_N+1)
    # For i from 1 to MAX_N:
    #   for j from i to MAX_N step i:
    #     sigma[j] += i
    # This is O(N log N). Sum_{i=1}^N N/i = N * H_N ~ N log N.
    # For N=10^7, log N ~ 16. N log N ~ 1.6 * 10^8 operations.
    # In Python, 1.6 * 10^8 simple operations might take ~10-20 seconds.
    # This is risky but possibly feasible within 85s.
    # Let's optimize the inner loop or use a different approach.
    
    # Alternative: Factorize a and b on the fly?
    # If we iterate k from 1 upwards, and for each k iterate b?
    # The number of b's for a fixed k is roughly MAX_N / 1.5 ~ 6.6 * 10^6.
    # If we do this for k=1,2,3... we might not finish k=1 in time if factorization is slow.
    # But we don't need to check all b. We just need *one* pair for each k?
    # No, we want the *minimum* error. The minimum error corresponds to the smallest k.
    # However, it's possible that for k=1, no solution exists, or the solution has large a,b.
    # Wait, if k=1, a/b = 1.5 + 1/(2b). The error is 1/(2b).
    # To minimize error, we want large b.
    # But we are constrained by a,b <= MAX_N.
    # So for a fixed k, the best candidate is the largest valid b.
    # Is it guaranteed that the global minimum error comes from the smallest k?
    # Error = |k| / (2b).
    # If k=1, error ~ 1/(2b_max).
    # If k=2, error ~ 2/(2b_max) = 1/b_max.
    # So smaller k generally gives smaller error, provided a solution exists with comparable b.
    # So we should iterate k = 1, 2, 3... and for each k, find the *largest* b
    # such that a = (3b+k)/2 is integer, a,b <= MAX_N, and sigma(a)=sigma(b).
    # If we find one, its error is k/(2b).
    # We can stop when k/(2b) is smaller than the best found so far?
    # Actually, since b is bounded by MAX_N, the minimum possible error for a given k
    # is k / (2 * MAX_N).
    # If k / (2 * MAX_N) >= best_error_so_far, we can stop.
    # Because for any larger k', error >= k'/(2*MAX_N) >= k/(2*MAX_N) >= best_error.
    
    # So the plan:
    # 1. Precompute sigma(n) for n up to MAX_N.
    # 2. Iterate k = 1, 2, 3...
    # 3. For each k, determine the range of b.
    #    a = (3b+k)/2 <= MAX_N  =>  3b+k <= 2*MAX_N  =>  b <= (2*MAX_N - k)/3
    #    Also a >= 1, b >= 1.
    #    Parity: 3b+k even => b and k same parity.
    #    So b_max = floor((2*MAX_N - k)/3). Adjust for parity.
    # 4. Iterate b from b_max downwards.
    #    Check if sigma(a) == sigma(b).
    #    If yes, record (a,b) and error = k/(2b).
    #    Break inner loop (we found the best b for this k).
    # 5. If k/(2*b_max) >= best_error, break outer loop.
    
    # Let's implement the sigma sieve first.
    sigma = [0] * (MAX_N + 1)
    # This loop is the bottleneck.
    # Optimization: Use a list of lists? No, too much memory.
    # Use a simple nested loop.
    # We can try to speed it up by local variable binding.
    
    # Actually, there is a faster way to compute sigma using the divisor sum property.
    # sigma(n) = sum_{d|n} d.
    # The sieve above does exactly that.
    # Let's try to optimize the Python loop.
    # for i in range(1, MAX_N + 1):
    #     for j in range(i, MAX_N + 1, i):
    #         sigma[j] += i
        
    # This might be too slow in pure Python.
    # Let's see if we can use NumPy?
    # NumPy slicing might be faster.
    # sigma[i::i] += i
    # But sigma is a Python list. Converting to NumPy array might help.
    
    import numpy as np
    sigma_np = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Check time
    t_sieve_start = time.time()
    for i in range(1, MAX_N + 1):
        sigma_np[i::i] += i
        if time.time() - t_sieve_start > 50:
             # Fallback to smaller range or hardcoded if sieve is too slow?
             # But we need accuracy.
             pass
    t_sieve_end = time.time()
    # print(f"Sieve took {t_sieve_end - t_sieve_start:.2f}s") # Disabled for speed
    
    # Convert back to list? Or use np array directly?
    # Accessing np array is slightly slower than list in Python?
    # Actually, list access is faster for single elements in CPython.
    # But sigma_np[i::i] += i is vectorized.
    # Let's stick to list for the search phase, or keep np array.
    # Let's convert to list for faster random access in the search loop.
    # sigma_list = sigma_np.tolist()
    
    # Wait, the sieve loop `for i in range(1, MAX_N+1): sigma_np[i::i] += i`
    # creates a slice view. This is efficient.
    # However, doing this 10^7 times might still be slow due to loop overhead in Python.
    # 10^7 iterations of a Python loop is ~0.5-1s.
    # The vectorized operation is fast.
    # Total time likely 5-10s.
    
    # Let's proceed with search.
    best_a, best_b = None, None
    best_error = float('inf')
    
    # We need sigma values. Using np array is fine.
    # sigma_val = sigma_np[n]
    
    for k in range(1, MAX_N + 1):  # k can be large?
        # Theoretical max k: if b=1, a=3/2+k/2. a integer => k even.
        # a <= 10^7.
        # But we stop early.
        
        # Check if current k can beat best_error
        # Min error for this k is k / (2 * b_max_possible)
        # b_max_possible is roughly MAX_N / 1.5 = 2/3 MAX_N
        # So min_error_k ~ k / (4/3 MAX_N) = 3k / (4 MAX_N)
        
        # Calculate b_max for this k
        # b <= (2*MAX_N - k) / 3
        b_max = (2 * MAX_N - k) // 3
        if b_max < 1:
            break
        
        # Parity adjustment
        # 3b + k must be even.
        # If k is odd, b must be odd.
        # If k is even, b must be even.
        if (k % 2) != (b_max % 2):
            b_max -= 1
            
        if b_max < 1:
            continue
            
        # Early termination check
        # If k / (2 * b_max) >= best_error, then even the best b for this k
        # is not better than our current best.
        # And for larger k, the ratio k/b_max will increase (roughly).
        # Let's be careful. k increases, b_max decreases slightly.
        # So k/b_max increases.
        # So if this k fails, all larger k will fail.
        if k / (2.0 * b_max) >= best_error:
            break
            
        # Search b from b_max downwards
        # We want the largest b to minimize error for fixed k.
        # So we iterate b = b_max, b_max-1, ...
        # But we might have to check many b's if no solution exists near b_max.
        # How many?
        # The density of pairs with sigma(a)=sigma(b) is unknown.
        # If it's low, we might scan a lot.
        
        found = False
        # To speed up, we can step by 2 to maintain parity?
        # b must have same parity as k.
        # b_max already has correct parity.
        # So step by 2.
        
        for b in range(b_max, 0, -2):
            a = (3 * b + k) // 2
            # a is integer by construction
            if a > MAX_N:
                # This shouldn't happen if b_max was calculated correctly
                # b <= (2N-k)/3 => 3b+k <= 2N => a <= N
                continue
                
            if sigma_np[a] == sigma_np[b]:
                best_a = a
                best_b = b
                best_error = k / (2.0 * b)
                found = True
                break
                
            # Time check inside inner loop?
            # If we are deep in k and scanning many b, we might time out.
            if time.time() > DEADLINE - 5:
                if best_a is not None:
                    return [best_a, best_b]
                # If no best found yet, we are in trouble.
                # Return a default?
                # Let's just break and return whatever we have or a fallback.
                break
                
        if found:
            # We found a solution for this k.
            # Can we stop?
            # The error is k/(2b).
            # For the next k+1, the min error is (k+1)/(2*b_max_next).
            # b_max_next is slightly smaller than b_max.
            # So (k+1)/b_max_next > k/b_max_next > k/b_max (roughly).
            # So the error will increase.
            # Thus, the first k for which we find a solution gives the global minimum?
            # Not necessarily.
            # It is possible that for k=1, the solution has b=10. Error = 1/20 = 0.05.
            # For k=2, the solution has b=1000. Error = 2/2000 = 0.001.
            # 0.001 < 0.05.
            # So we CANNOT stop at the first k that yields a solution.
            # We must continue until the *theoretical minimum* error for the current k
            # exceeds the *actual best* error found.
            # The theoretical minimum for k is k / (2 * b_max_k).
            # We already check `if k / (2.0 * b_max) >= best_error: break`.
            # This check happens *before* the inner loop.
            # So if we found a solution for k=1 with error 0.05,
            # and for k=2, b_max=3333333, min_error = 2/(2*3333333) ~ 3e-7.
            # 3e-7 < 0.05, so we proceed to search k=2.
            # If we find a solution with error 0.001, best_error becomes 0.001.
            # For k=3, min_error ~ 3/(2*3333333) ~ 4.5e-7.
            # 4.5e-7 < 0.001, so we proceed.
            # This logic holds. We continue until the lower bound exceeds the best found.
            
            if time.time() > DEADLINE - 2:
                break

    if best_a is None:
        # Fallback if no solution found (should not happen)
        return [1, 1]
        
    return [best_a, best_b]

# EVOLVE-BLOCK-END
