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
    
    # Sort to potentially help with early termination or cache, though logic is symmetric
    # Actually, we just iterate through each element a
    for a in set_vals:
        # Get others
        others = [v for v in set_vals if v != a]
        n_others = len(others)
        
        if n_others < 2:
            continue
            
        # Max target sum for k elements is k * a.
        # We need to check k from 2 to n_others.
        # Max target overall is n_others * a.
        max_target_sum = n_others * a
        
        # We only care about sums up to max_target_sum.
        # However, for a specific k, target is k*a.
        # If we mask to max_target_sum, we cover all k.
        
        # Optimization: If max_target_sum is very large, the bitset gets huge.
        # But n_others <= 47, a <= 512. Max target ~ 24064.
        # 24064 bits is ~ 3KB. This is very small.
        
        mask = (1 << (max_target_sum + 1)) - 1
        
        # dp[k] is a bitset of sums achievable with exactly k elements from 'others'
        dp = [0] * (n_others + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for v in others:
            for k in range(n_others, 0, -1):
                dp[k] |= (dp[k-1] << v)
                dp[k] &= mask
        
        # Check if k*a is achievable with k elements
        for k in range(2, n_others + 1):
            target = k * a
            if target <= max_target_sum:
                if (dp[k] >> target) & 1:
                    return False
            
    return True

def _generate_cantor_set(limit):
    """
    Generate positive integers <= limit with base-3 digits only 0 or 1.
    """
    cantor_set = []
    # Max power of 3 <= limit
    p = 1
    powers = []
    while p <= limit:
        powers.append(p)
        p *= 3
    
    # Generate all combinations of these powers
    # Number of powers is roughly log_3(limit)
    # For limit 512, powers are 1, 3, 9, 27, 81, 243. (6 powers)
    # 2^6 = 64 combinations.
    for mask in range(1, 1 << len(powers)):
        val = 0
        for i in range(len(powers)):
            if mask & (1 << i):
                val += powers[i]
        if val <= limit:
            cantor_set.append(val)
            
    return sorted(cantor_set)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin for 90s kill
    
    N = 512
    MAX_K = 48
    
    # Strategy 1: Cantor Set Construction
    # This set is structurally "sparse" and likely satisfies the condition.
    cantor_candidates = _generate_cantor_set(N)
    
    # We need 48 distinct positive integers.
    # The Cantor set has ~63 elements.
    # Let's try to take a subset of size 48 that is valid.
    # If the whole set is valid, any subset is valid.
    # So we just check if a random subset of size 48 is valid?
    # Or better, check if the first 48 are valid.
    
    best_set = []
    
    if len(cantor_candidates) >= MAX_K:
        # Try the first 48
        candidate = cantor_candidates[:MAX_K]
        if _is_valid_fast(candidate):
            return candidate
        
        # If the first 48 fail (unlikely if the whole set is valid, but maybe interaction?),
        # try random subsets of the Cantor set
        for _ in range(100):
            if time.time() > deadline:
                break
            random.shuffle(cantor_candidates)
            candidate = cantor_candidates[:MAX_K]
            if _is_valid_fast(candidate):
                return candidate
        
        # If Cantor set fails, use it as a seed for local search
        # But first, let's see if we can build a valid set from it greedily
        selected = []
        for x in cantor_candidates:
            if time.time() > deadline:
                break
            if len(selected) >= MAX_K:
                break
            if _is_valid_fast(selected + [x]):
                selected.append(x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
            
    # Strategy 2: Standard Greedy (Ascending)
    if len(best_set) < MAX_K:
        selected = []
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if len(selected) >= MAX_K:
                break
            if _is_valid_fast(selected + [x]):
                selected.append(x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
            
    # Strategy 3: Standard Greedy (Descending)
    if len(best_set) < MAX_K:
        selected = []
        for x in range(N, 0, -1):
            if time.time() > deadline:
                break
            if len(selected) >= MAX_K:
                break
            if _is_valid_fast(selected + [x]):
                selected.append(x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
            
    # Strategy 4: Local Search on best_set
    # Try to increase size or maintain size while swapping for better "spread"
    if time.time() < deadline - 10:
        current = best_set[:]
        if not current:
            current = [1]
            
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try to add
            if len(current) < MAX_K:
                remaining = [x for x in range(1, N + 1) if x not in current]
                random.shuffle(remaining)
                for x in remaining[:50]:
                    if _is_valid_fast(current + [x]):
                        current.append(x)
                        improved = True
                        if len(current) >= MAX_K:
                            return current
                        break
                        
            # If no addition, try swaps
            if not improved and time.time() < deadline - 5:
                if current:
                    u_idx = random.randint(0, len(current) - 1)
                    u = current[u_idx]
                    remaining = [x for x in range(1, N + 1) if x not in current]
                    if remaining:
                        v = random.choice(remaining)
                        new_current = current[:]
                        new_current[u_idx] = v
                        if _is_valid_fast(new_current):
                            current = new_current
                            improved = True

        if len(current) > len(best_set):
            best_set = current[:]

    return best_set[:MAX_K]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
    print(res)
