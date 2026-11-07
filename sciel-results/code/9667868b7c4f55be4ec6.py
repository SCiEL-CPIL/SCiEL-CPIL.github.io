import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Primes to consider for smooth numbers
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    max_m = 10**18
    # k must be <= sqrt(M) <= 10^9
    # But if we generate k up to 10^9, the count is huge.
    # Let's start with a reasonable limit and increase if time permits.
    # Actually, the constraint is k * sigma(k) <= 10^18.
    # For small k, sigma(k) is small, so k can be close to 10^9.
    # For large k, sigma(k) is larger, so k must be smaller.
    
    solutions = defaultdict(list)
    
    # To avoid recursion depth issues and overhead, we can use a stack or just recursion with sys.setrecursionlimit
    # Recursion depth is at most the number of primes, which is small.
    
    def gen_smooth(idx, current_k, current_sigma):
        if time.time() > deadline:
            return
            
        if idx == len(primes):
            if current_k > 1:
                m_val = current_k * current_sigma
                if m_val <= max_m:
                    # Check if current_k <= sqrt(m_val)
                    # Since current_sigma >= current_k + 1, m_val >= current_k^2 + current_k
                    # So current_k < sqrt(m_val) is always true for k > 1.
                    solutions[m_val].append(current_k)
            return
            
        p = primes[idx]
        # Multiply by p^e
        # current_k * p^e
        # current_sigma * (1 + p + ... + p^e)
        
        k_val = current_k
        sigma_val = current_sigma
        sum_powers = 1
        p_power = 1
        
        while True:
            # Check if we can multiply by p again
            # Next k: k_val * p
            # Next sigma: sigma_val * (sum_powers * p + 1) ... wait.
            # If k = A * p^e, sigma(k) = sigma(A) * (1 + p + ... + p^e).
            # Let S_e = 1 + p + ... + p^e.
            # If we go to e+1, S_{e+1} = S_e + p^{e+1}.
            # So we can update incrementally.
            
            # Current state: k_val, sigma_val.
            # This state corresponds to some exponent e (implicitly).
            # We add this state to the dictionary.
            if k_val > 1:
                m_val = k_val * sigma_val
                if m_val <= max_m:
                    solutions[m_val].append(k_val)
                    
            # Try next exponent
            # new_k = k_val * p
            # new_sigma = sigma_val / S_e * S_{e+1}
            # It's easier to track S_e separately or just recompute?
            # Recomputing sigma for smooth numbers is fast if we have factors, but we don't store factors here.
            # Let's stick to the incremental method properly.
            
            # Let's restart the loop logic to be clearer.
            # We are at prime index idx.
            # We have already processed primes < idx.
            # current_k and current_sigma are for the number formed by primes < idx.
            
            # Now we add prime p.
            # We can have exponent 0, 1, 2, ...
            
            # Exponent 0 is handled by the recursive call to idx+1 (which is implicitly done by calling gen_smooth(idx+1, current_k, current_sigma) outside the loop? No, usually we handle e=0 by recursing immediately, then loop for e>=1).
            
            # Let's restructure:
            # 1. Recurse with e=0 (do not include p)
            # 2. Loop for e=1, 2, ...
            
            # But the current structure above appends to solutions at the leaf (idx == len(primes)).
            # This is inefficient because it checks all leaves.
            # Better: Append to solutions at every node? No, M is unique to the full k.
            # So we must reach the leaf to know the full k.
            # However, we can prune branches early if k * sigma(k) > 10^18.
            
            # Let's optimize the inner loop.
            # We need to compute new_sigma when multiplying by p.
            # If current_k = A, current_sigma = sigma(A).
            # New k = A * p^e.
            # New sigma = sigma(A) * (1 + p + ... + p^e).
            
            # Let's use a helper to update.
            # We'll iterate e from 1 upwards.
            
            break # Exit the while True structure above, let's rewrite the function body cleanly.

    # Rewriting gen_smooth for clarity and speed
    def gen_smooth_v2(idx, current_k, current_sigma):
        if time.time() > deadline:
            return

        if idx == len(primes):
            if current_k > 1:
                m_val = current_k * current_sigma
                if m_val <= max_m:
                    solutions[m_val].append(current_k)
            return

        p = primes[idx]
        
        # Option 1: Do not use prime p (exponent 0)
        gen_smooth_v2(idx + 1, current_k, current_sigma)
        
        # Option 2: Use prime p with exponent e >= 1
        # k_new = current_k * p^e
        # sigma_new = current_sigma * (1 + p + ... + p^e)
        
        k_curr = current_k
        sigma_curr = current_sigma
        sum_powers = 1 # 1 + p + ... + p^e
        p_power = 1    # p^e
        
        for e in range(1, 50): # Max exponent is small
            # Multiply by p
            if k_curr > max_m // p: # k_curr * p > 10^18, break
                break
            
            k_curr *= p
            p_power *= p
            sum_powers += p_power
            
            # Check overflow for sigma
            # sigma_curr * sum_powers
            # If sigma_curr > max_m // sum_powers, then M > 10^18
            if sum_powers > 0 and sigma_curr > max_m // sum_powers:
                break
                
            sigma_curr_new = sigma_curr * sum_powers
            k_curr_new = k_curr
            
            # Prune: if k_curr_new * sigma_curr_new > max_m, we can break?
            # Not necessarily, because we might add more primes? No, adding more primes increases both k and sigma.
            # So if M > max_m now, it will be > max_m later.
            if k_curr_new * sigma_curr_new > max_m:
                break
                
            # Recurse
            gen_smooth_v2(idx + 1, k_curr_new, sigma_curr_new)

    # Start generation
    # Initial k=1, sigma=1
    # We skip k=1 because 1*sigma(1)=1, and we want k>1 usually? 
    # The task says k_i >= 1. But M=1 has only k=1. 
    # We want M with many solutions. M=1 is bad.
    # The recursion handles k=1 at the start.
    
    # To save time, we can start with k=1, sigma=1.
    gen_smooth_v2(0, 1, 1)
    
    # Find best M
    best_m = 0
    best_k_list = []
    
    for m, k_list in solutions.items():
        if len(k_list) > len(best_k_list):
            best_m = m
            best_k_list = k_list
            
    # If we didn't find anything (shouldn't happen), fallback
    if not best_k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Verify and format
    # Ensure distinct
    distinct_k = list(set(best_k_list))
    # Ensure k <= sqrt(M)
    sqrt_m = math.isqrt(best_m)
    valid_k = [k for k in distinct_k if k <= sqrt_m]
    
    # Final check
    # The evaluator checks k * sigma(k) == M.
    # We trust our generation, but let's be safe.
    # We don't have a fast sigma function here, but we can assume correctness from generation.
    # However, the prompt says "checked with exact factorization".
    # If my generation logic is correct, it is exact.
    
    # Sort for determinism? Not required, but good.
    valid_k.sort()
    
    return {'M': best_m, 'k': valid_k}
