"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 85.0  # seconds, safety margin before 90s limit

def _check_violation(candidate, a_idx):
    """
    Check if element at a_idx is the mean of any subset of at least 2 OTHER elements.
    Returns True if violation exists.
    """
    a = candidate[a_idx]
    others = candidate[:a_idx] + candidate[a_idx+1:]
    n = len(others)
    if n < 2:
        return False
    
    # Transform: z_i = y_i - a
    zs = [y - a for y in others]
    
    # For k=2: check all pairs directly
    for i in range(n):
        for j in range(i+1, n):
            if zs[i] + zs[j] == 0:
                return True
    
    # For k>=3: use DP with sets
    # dp[k] = set of achievable sums using exactly k elements from zs
    dp = [set() for _ in range(n+1)]
    dp[0].add(0)
    
    for z in zs:
        for k in range(n, 0, -1):
            if dp[k-1]:
                for s in list(dp[k-1]):
                    dp[k].add(s + z)
    
    # Check if 0 is in dp[k] for any k >= 3
    for k in range(3, n+1):
        if 0 in dp[k]:
            return True
    
    return False

def _full_check(candidate):
    """Check all elements for violations."""
    for i in range(len(candidate)):
        if _check_violation(candidate, i):
            return True
    return False

def _generate_multiplicative_pool():
    """Generate pool of 2^i * 3^j <= 512."""
    pool = set()
    for i in range(10):  # 2^9 = 512
        p2 = 2 ** i
        if p2 > 512:
            break
        for j in range(10):  # 3^5 = 243, 3^6 = 729 > 512
            val = p2 * (3 ** j)
            if val > 512:
                break
            pool.add(val)
    return sorted(pool)

def construct():
    start_time = time.time()
    
    # Generate candidate pools
    mult_pool = _generate_multiplicative_pool()
    full_pool = list(range(1, 513))
    
    # Try multiple strategies and orderings
    best_set = []
    
    strategies = []
    
    # Strategy 1: Multiplicative pool, increasing
    strategies.append(('mult_inc', mult_pool))
    
    # Strategy 2: Multiplicative pool, decreasing
    strategies.append(('mult_dec', list(reversed(mult_pool))))
    
    # Strategy 3: Full pool, increasing
    strategies.append(('full_inc', full_pool))
    
    # Strategy 4: Full pool, decreasing
    strategies.append(('full_dec', list(reversed(full_pool))))
    
    # Strategy 5: Multiplicative pool, random orders (a few)
    for _ in range(3):
        pool_copy = mult_pool[:]
        random.shuffle(pool_copy)
        strategies.append(('mult_rand', pool_copy))
    
    for name, order in strategies:
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        selected = []
        for x in order:
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            if x in selected:
                continue
            
            candidate = selected + [x]
            
            # Check if adding x creates any violation
            # Only need to check:
            # 1. If x is the mean of a subset of selected (size >= 2)
            # 2. If any element in selected is now the mean of a subset including x
            
            # For efficiency, check all elements in candidate
            # But we can optimize: since selected was valid before,
            # only violations involving x are new.
            
            valid = True
            n_sel = len(selected)
            
            # Check if x is mean of subset of selected (size >= 2)
            if n_sel >= 2:
                if _check_violation(candidate, n_sel):  # x is at index n_sel
                    valid = False
            
            # Check if any selected element is mean of subset including x
            if valid:
                for i in range(n_sel):
                    if _check_violation(candidate, i):
                        valid = False
                        break
            
            if valid:
                selected.append(x)
                
                if len(selected) >= 48:
                    break
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Final verification
    if best_set and not _full_check(best_set):
        return best_set
    
    # Fallback: return a known valid small set
    return [1, 2, 4, 8, 16, 32, 64, 128, 256]

# EVOLVE-BLOCK-END
