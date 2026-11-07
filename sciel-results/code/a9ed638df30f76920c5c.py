import math
import time

def construct():
    n = 256
    
    # Sieve of Eratosthenes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Split primes into small and large
    # Small primes: p^2 <= n => p <= 16
    small_primes = [p for p in primes if p * p <= n]
    large_primes = [p for p in primes if p * p > n]
    
    # Map large primes to indices for bitmask
    large_prime_to_idx = {p: i for i, p in enumerate(large_primes)}
    num_large_primes = len(large_primes)
    
    # Generate groups G_p for small primes
    # G_p: list of tuples (value, large_factors_mask, small_factors_larger_mask)
    # small_factors_larger_mask: bitmask of small primes (excluding p) that divide the number
    # We need to map small primes to indices for this mask too
    small_prime_to_idx = {p: i for i, p in enumerate(small_primes)}
    num_small_primes = len(small_primes)
    
    groups = []
    for p in small_primes:
        group_items = []
        for x in range(p, n + 1):
            if x % p != 0:
                continue
            # Check if spf(x) == p
            # x is not divisible by any prime < p
            is_spf = True
            for q in small_primes:
                if q >= p:
                    break
                if x % q == 0:
                    is_spf = False
                    break
            if not is_spf:
                continue
            
            # Factorize x to find large and small factors
            large_mask = 0
            small_mask_larger = 0
            temp = x
            for q in primes:
                if q * q > temp:
                    break
                if temp % q == 0:
                    while temp % q == 0:
                        temp //= q
                    if q in large_prime_to_idx:
                        large_mask |= (1 << large_prime_to_idx[q])
                    elif q in small_prime_to_idx:
                        idx_q = small_prime_to_idx[q]
                        idx_p = small_prime_to_idx[p]
                        if q != p:
                            # Only track small factors larger than p?
                            # Actually, if q < p, it shouldn't happen because spf(x)=p.
                            # If q > p, it blocks group q.
                            if q > p:
                                small_mask_larger |= (1 << idx_q)
                    else:
                        # q is p?
                        pass
            if temp > 1:
                if temp in large_prime_to_idx:
                    large_mask |= (1 << large_prime_to_idx[temp])
                elif temp in small_prime_to_idx:
                    idx_temp = small_prime_to_idx[temp]
                    idx_p = small_prime_to_idx[p]
                    if temp != p and temp > p:
                        small_mask_larger |= (1 << idx_temp)
            
            group_items.append((x, large_mask, small_mask_larger))
        
        groups.append(group_items)
    
    # Precompute sum of all large primes
    total_large_sum = sum(large_primes)
    
    best_sum = 1 # Start with 1
    
    def search(idx, current_sum, blocked_large_mask, blocked_small_mask):
        nonlocal best_sum
        
        if idx == num_small_primes:
            # Calculate sum of available large primes
            # Available if not in blocked_large_mask
            # Note: blocked_large_mask only contains bits for large primes
            sum_large = 0
            for i, p in enumerate(large_primes):
                if not (blocked_large_mask & (1 << i)):
                    sum_large += p
            total = current_sum + sum_large
            if total > best_sum:
                best_sum = total
            return
        
        p = small_primes[idx]
        group = groups[idx]
        
        # Option 1: Pick nothing from this group
        search(idx + 1, current_sum, blocked_large_mask, blocked_small_mask)
        
        # Option 2: Pick one item from this group
        for x, l_mask, s_mask in group:
            # Check conflict with existing blocked large primes
            if l_mask & blocked_large_mask:
                continue
            
            # Check if this group is already blocked by previous choices
            # blocked_small_mask has bits for small primes > previous p
            # If bit for current p is set in blocked_small_mask, we can't pick anything
            # But we check this before calling search for this idx?
            # Yes, we should check if group idx is blocked.
            # How do we know if it's blocked?
            # If a previous choice had small_factor == p, then p is blocked.
            # We pass blocked_small_mask.
            # If (blocked_small_mask & (1 << idx)) is set, skip all picks.
            if blocked_small_mask & (1 << idx):
                continue
            
            new_blocked_large = blocked_large_mask | l_mask
            new_blocked_small = blocked_small_mask | s_mask
            
            search(idx + 1, current_sum + x, new_blocked_large, new_blocked_small)
            
    # Initial call
    # blocked_small_mask initially 0
    # We need to ensure we don't exceed time
    start_time = time.time()
    
    # Optimization: Sort groups by size or something? No, order is fixed by prime value for correctness of "spf" logic?
    # Actually, the logic "if q > p, block q" relies on processing in increasing p order.
    # So we must process small_primes in increasing order.
    
    search(0, 0, 0, 0)
    
    # Reconstruct result?
    # The task asks for the list of ints.
    # We only tracked the sum. We need the actual numbers.
    # We should modify search to track the best choice.
    
    best_choice = None
    
    def search_with_path(idx, current_sum, blocked_large_mask, blocked_small_mask, path):
        nonlocal best_sum, best_choice
        
        if idx == num_small_primes:
            sum_large = 0
            large_list = []
            for i, p in enumerate(large_primes):
                if not (blocked_large_mask & (1 << i)):
                    sum_large += p
                    large_list.append(p)
            total = current_sum + sum_large
            if total > best_sum:
                best_sum = total
                best_choice = path + large_list
            return
        
        p = small_primes[idx]
        group = groups[idx]
        
        # Option 1: Pick nothing
        search_with_path(idx + 1, current_sum, blocked_large_mask, blocked_small_mask, path)
        
        # Option 2: Pick one
        if not (blocked_small_mask & (1 << idx)):
            for x, l_mask, s_mask in group:
                if l_mask & blocked_large_mask:
                    continue
                new_blocked_large = blocked_large_mask | l_mask
                new_blocked_small = blocked_small_mask | s_mask
                search_with_path(idx + 1, current_sum + x, new_blocked_large, new_blocked_small, path + [x])
                
    search_with_path(0, 0, 0, 0, [])
    
    # Add 1
    result = [1] + best_choice
    
    # Ensure types are int
    return [int(x) for x in result]

print(construct())
