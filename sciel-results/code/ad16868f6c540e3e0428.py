import time
import random
import sys

def _is_valid_fast(set_vals):
    """
    Check if no element in set_vals is the mean of any subset of >=2 other elements.
    Uses bitset DP for subset sums.
    """
    m = len(set_vals)
    if m < 3:
        return True
    
    # To speed up, we can just check for each a
    # If we sort, it doesn't matter for the check logic, but maybe for cache?
    # We will iterate through each element a in the set.
    
    for a in set_vals:
        # Get others
        # Since m is small (<=48), list comprehension is fine
        others = [v for v in set_vals if v != a]
        n_others = len(others)
        
        # Max possible sum of k elements is k * max(others)
        # We need to check if k*a is in sums of size k.
        # k ranges from 2 to n_others.
        # Max target sum is n_others * a.
        # Actually, max sum of k elements from others is k * max(others).
        # If k * a > k * max(others), it's impossible.
        
        # DP: dp[k] is a bitset of sums achievable with exactly k elements from 'others'
        dp = [0] * (n_others + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        # Optimization: We only need to track sums up to max(k*a) for k in 2..n_others
        # Max k is n_others. Max target is n_others * a.
        # But sums can be larger. However, if sum > max_target, we don't care.
        # Let's mask to keep integers small.
        # Max target for any k is k * a.
        # The largest k is n_others. So max_target = n_others * a.
        # If n_others * a is small, we can mask.
        # n_others <= 47, a <= 512. Max target ~ 24k. Bitset size ~ 3KB. Very safe.
        
        max_target_sum = n_others * a
        if max_target_sum < 50000: # Always true here
            mask = (1 << (max_target_sum + 1)) - 1
        else:
            mask = None
            
        for v in others:
            for k in range(n_others, 0, -1):
                dp[k] |= (dp[k-1] << v)
                if mask:
                    dp[k] &= mask
        
        # Check if k*a is achievable with k elements
        for k in range(2, n_others + 1):
            target = k * a
            if target <= max_target_sum:
                if (dp[k] >> target) & 1:
                    return False
            # If target > max_target_sum, it's 0, so no violation from this k
            # But wait, if target > max_target_sum, it means we can't form it.
            # So we just check if the bit is set.
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    N = 512
    MAX_K = 48
    
    best_set = []
    best_size = 0
    
    # Pre-generate some orderings to use in restarts
    # 1. Descending
    # 2. Ascending
    # 3. Random
    # 4. Primes first
    # 5. Powers of 2 mixed?
    
    primes = [x for x in range(1, N + 1) if all(x % i != 0 for i in range(2, int(x**0.5)+1)) and x > 1]
    composites = [x for x in range(1, N + 1) if x not in primes]
    
    orderings = []
    for _ in range(10):
        orderings.append(list(range(N, 0, -1)))
    for _ in range(10):
        r = list(range(1, N + 1))
        random.shuffle(r)
        orderings.append(r)
    for _ in range(10):
        p = primes[:]
        c = composites[:]
        random.shuffle(p)
        random.shuffle(c)
        orderings.append(p + c)
    for _ in range(10):
        # Alternating high/low
        r1 = list(range(1, N//2 + 1))
        r2 = list(range(N//2 + 1, N + 1))
        random.shuffle(r1)
        random.shuffle(r2)
        mixed = []
        i, j = 0, 0
        while i < len(r1) or j < len(r2):
            if i < len(r1): mixed.append(r1[i]); i+=1
            if j < len(r2): mixed.append(r2[j]); j+=1
        orderings.append(mixed)
        
    random.shuffle(orderings)
    
    for restart in range(200): # Cap restarts, but time is the real limit
        if time.time() > deadline:
            break
            
        ordering = orderings[restart % len(orderings)]
        # Add some randomness to the ordering if we've seen it before?
        # Or just use the pre-shuffled ones.
        
        selected = []
        
        # Phase 1: Greedy Construction
        for x in ordering:
            if time.time() > deadline:
                break
            if len(selected) >= MAX_K:
                break
                
            # Check if x can be added
            if _is_valid_fast(selected + [x]):
                selected.append(x)
                
        current_size = len(selected)
        
        # Phase 2: Local Search (Swaps) to increase size or improve structure
        # If we reached 48, we are done (it's valid by construction).
        if current_size >= MAX_K:
            return selected
            
        # If not, try to improve.
        # We try to swap elements to "unblock" the addition of new elements.
        # Or simply swap to find a configuration where more elements can be added.
        
        # Let's do a simple hill climbing on size.
        # But size is fixed during swaps.
        # So we alternate:
        # 1. Try to add any element.
        # 2. If fail, swap a random element in with a random element out.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try to add
            if len(selected) < MAX_K:
                # Try all remaining elements? Too slow if we do full check for each.
                # Random sample?
                remaining = [x for x in range(1, N+1) if x not in selected]
                random.shuffle(remaining)
                for x in remaining[:20]: # Check a few candidates
                    if _is_valid_fast(selected + [x]):
                        selected.append(x)
                        improved = True
                        if len(selected) >= MAX_K:
                            return selected
                        break # Restart the loop to try adding more
                        
            # If no addition, try a swap
            if not improved and time.time() < deadline - 5:
                if selected:
                    # Pick random u in selected, v not in selected
                    u_idx = random.randint(0, len(selected)-1)
                    u = selected[u_idx]
                    remaining = [x for x in range(1, N+1) if x not in selected]
                    if remaining:
                        v = random.choice(remaining)
                        # Try swap
                        new_selected = selected[:]
                        new_selected[u_idx] = v
                        # Check validity of new_selected
                        if _is_valid_fast(new_selected):
                            selected = new_selected
                            improved = True
                            # Don't break, loop again to try adding

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
            if best_size == MAX_K:
                return best_set
                
    return best_set[:MAX_K]

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found set of size {len(res)}")
    print(res[:10])
