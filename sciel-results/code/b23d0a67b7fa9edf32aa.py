import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, leaving margin before 90s kill
    
    MAX_N = 256
    
    # 1. Identify primes
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    p_index = {p: i for i, p in enumerate(primes)}
    n_primes = len(primes)
    
    # 2. Compute weights w(k) for square-free kernels k
    # rad(n) is the product of distinct prime factors of n
    # We want max n for each rad(n)
    weights = {}
    for n in range(1, MAX_N + 1):
        # Compute rad(n)
        temp = n
        rad = 1
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                rad *= p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            rad *= temp
        
        if rad in weights:
            if n > weights[rad]:
                weights[rad] = n
        else:
            weights[rad] = n
            
    # 3. Create items for B&B
    # Item: (weight, prime_mask)
    # We only care about items that are "potentially useful".
    # An item k is useful if w(k) > 0.
    # Also, we can prune items: if there exists k' such that mask(k') is a superset of mask(k) and w(k') >= w(k), 
    # then k is dominated? No, because k' uses more primes.
    # Actually, if mask(k) == mask(k'), we keep max w. Already done.
    
    items = []
    for k, w in weights.items():
        if w == 0:
            continue
        # Compute prime mask for k
        mask = 0
        temp = k
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << p_index[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << p_index[temp])
            
        items.append((w, mask))
        
    # Sort by weight descending to find good solutions early
    items.sort(key=lambda x: x[0], reverse=True)
    
    n_items = len(items)
    
    # Precompute suffix sums for upper bound
    suffix_sums = [0] * (n_items + 1)
    for i in range(n_items - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items[i][0]
        
    best_sum = 0
    best_solution = []
    
    # Branch and Bound
    # State: idx, used_mask, current_sum, current_items
    def dfs(idx, used_mask, current_sum, current_items):
        nonlocal best_sum, best_solution
        
        if time.time() > deadline - 2:
            return
            
        # Update best if we have a valid solution (we always can stop here)
        # Note: We don't have to use all primes.
        if current_sum > best_sum:
            best_sum = current_sum
            best_solution = current_items[:]
            
        if idx == n_items:
            return
            
        # Upper bound
        # Simple bound: current_sum + suffix_sums[idx]
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        # Tighter bound: sum of weights of remaining items that don't conflict
        # This is expensive to compute at every node. 
        # Optimization: Only compute if simple bound is close to best_sum?
        # Or just use simple bound first.
        
        # Option 1: Skip item idx
        dfs(idx + 1, used_mask, current_sum, current_items)
        
        # Option 2: Take item idx
        w, mask = items[idx]
        if (mask & used_mask) == 0:
            current_items.append((w, mask))
            dfs(idx + 1, used_mask | mask, current_sum + w, current_items)
            current_items.pop()
            
    dfs(0, 0, 0, [])
    
    # Construct result from best_solution
    result_numbers = [1] # Always include 1? 
    # Wait, 1 is an item with mask 0 and weight 1.
    # The B&B might or might not include it.
    # Since it has weight 1 and no conflict, it should always be included in an optimal solution 
    # unless best_sum is 0 (impossible).
    # However, if the B&B didn't include it, we can add it.
    # But if the B&B included it, we don't add it again.
    
    # Let's check if 1 is in the solution
    has_one = False
    result_numbers = []
    for w, mask in best_solution:
        if mask == 0:
            # This is the number 1
            if w == 1:
                result_numbers.append(1)
                has_one = True
        else:
            # Reconstruct the number? 
            # We stored the weight w, which IS the number n.
            # Because w(k) = max n with rad(n)=k.
            # So w is the number itself.
            result_numbers.append(int(w))
            
    if not has_one:
        result_numbers.append(1)
        
    # Verify distinctness and coprimality
    # The B&B ensures disjoint prime masks, so they are coprime.
    # But we must ensure distinctness.
    # Since each number n has a unique rad(n), and we picked distinct items (k), 
    # could two different k's produce the same n? No, rad is a function of n.
    # Could the same n be produced by different k's? No.
    # So the numbers are distinct.
    
    # Final check
    def is_valid(s):
        if len(s) != len(set(s)):
            return False
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if gcd(s[i], s[j]) != 1:
                    return False
        return True
        
    if not is_valid(result_numbers):
        # Fallback
        result_numbers = [1] + primes
        
    return [int(x) for x in result_numbers]

# EVOLVE-BLOCK-END
