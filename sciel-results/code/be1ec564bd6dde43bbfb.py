import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Primes to consider for smooth numbers
    # Using more primes allows for more diverse k, but increases branching factor.
    # 15 primes was used before. Let's try up to 25 primes, but the iterative approach handles it better.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    max_m = 10**18
    # k must be <= sqrt(M) <= 10^9
    
    solutions = defaultdict(list)
    
    # Stack for iterative DFS
    # Each element: (idx, current_k, current_sigma)
    # We need to manage the state such that we can backtrack or just push all next states.
    # Since we want to explore all combinations, a standard DFS stack works.
    # State: (prime_index, k, sigma)
    # If prime_index == len(primes), we have a complete k.
    
    # Optimization: We can prune if k * sigma > max_m.
    # Also, we can prune if k > sqrt(max_m) because k must be <= sqrt(M) <= sqrt(max_m).
    # Wait, the constraint is k <= sqrt(M). Since M = k * sigma(k), this means k <= sqrt(k * sigma(k)) => k^2 <= k * sigma(k) => k <= sigma(k).
    # Since sigma(k) >= k + 1 for k > 1, this is always true for k > 1.
    # However, we also have the hard limit M <= 10^18.
    # So we just need k * sigma(k) <= 10^18.
    
    stack = [(0, 1, 1)]
    
    while stack:
        if time.time() > deadline:
            break
            
        idx, current_k, current_sigma = stack.pop()
        
        if idx == len(primes):
            if current_k > 1:
                m_val = current_k * current_sigma
                if m_val <= max_m:
                    solutions[m_val].append(current_k)
            continue
            
        p = primes[idx]
        
        # Option 1: Do not use prime p (exponent 0)
        # We push this state to be processed next (or later, depending on stack order).
        # To mimic recursion depth-first, we push the "next prime" state first?
        # No, standard stack DFS: push children.
        # Child 1: Exponent 0 -> (idx+1, current_k, current_sigma)
        # Child 2..: Exponent e >= 1
        
        # To ensure we don't miss any, we push all valid children.
        # However, pushing all at once uses more memory.
        # Better: Push the "skip" case, then loop for exponents and push those too.
        # Order doesn't matter for finding the max count, as long as we visit all.
        
        stack.append((idx + 1, current_k, current_sigma))
        
        # Option 2: Use prime p with exponent e >= 1
        k_curr = current_k
        sigma_curr = current_sigma
        sum_powers = 1 # 1 + p + ... + p^e
        p_power = 1    # p^e
        
        for e in range(1, 50):
            # Check if we can multiply by p
            if k_curr > max_m // p:
                break
                
            k_curr *= p
            p_power *= p
            sum_powers += p_power
            
            # Check sigma overflow
            if sum_powers > 0 and sigma_curr > max_m // sum_powers:
                break
                
            sigma_curr_new = sigma_curr * sum_powers
            k_curr_new = k_curr
            
            # Prune: if M > max_m, break (adding more primes will only increase M)
            if k_curr_new * sigma_curr_new > max_m:
                break
                
            # Push this state to stack
            # We are at prime idx, we have used p^e. Next prime is idx+1.
            stack.append((idx + 1, k_curr_new, sigma_curr_new))
            
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
    
    # Sort for determinism
    valid_k.sort()
    
    return {'M': best_m, 'k': valid_k}
