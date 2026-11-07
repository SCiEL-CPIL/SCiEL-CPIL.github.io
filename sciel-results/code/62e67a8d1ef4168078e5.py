"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _is_valid_bitmask(s):
    """Check if set s satisfies cardinality-separated subset sum property using bitmasks."""
    n = len(s)
    if n == 0:
        return True
    # reachable[k] is a bitmask of achievable sums with exactly k elements
    reachable = [0] * (n + 1)
    reachable[0] = 1  # sum 0 is achievable with 0 elements
    
    for x in s:
        # Iterate backwards to avoid using x multiple times in one step
        # We need to update reachable[k] based on reachable[k-1] from the PREVIOUS state
        # Since we are modifying in place, we need to be careful.
        # Actually, it's safer to compute new_reachable from old_reachable.
        # But for validity check of a fixed set, we can just do:
        # new_reach[k] = old_reach[k] | (old_reach[k-1] << x)
        # We can update in reverse order k=n..1
        for k in range(n, 0, -1):
            reachable[k] |= (reachable[k-1] << x)
            
    # Check pairwise disjointness
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if reachable[i] & reachable[j]:
                return False
    return True

def _try_add_bitmask(reach, x):
    """
    Try to add x to the set represented by reach (list of bitmasks).
    reach[k] is bitmask of sums with exactly k elements.
    Returns new_reach (list of bitmasks) if valid, else None.
    """
    n = len(reach) - 1
    new_reach = [0] * (n + 2)
    
    # Copy old reach
    for k in range(n + 1):
        new_reach[k] = reach[k]
        
    # Add new sums
    for k in range(1, n + 2):
        new_reach[k] |= (reach[k-1] << x)
        
    # Check pairwise disjointness for new_reach
    # Optimization: Only check pairs involving the "new" parts?
    # No, just check all pairs. N is small.
    for i in range(n + 2):
        for j in range(i + 1, n + 2):
            if new_reach[i] & new_reach[j]:
                return None
                
    return new_reach

def _greedy_build_bitmask(pool, order=None):
    """Greedy build using bitmask DP."""
    if order is None:
        order = list(pool)
    
    current_set = []
    # reach[0] = 1 (bit 0 set)
    reach = [1]
    
    for x in order:
        if x in current_set:
            continue
        new_reach = _try_add_bitmask(reach, x)
        if new_reach is not None:
            current_set.append(x)
            reach = new_reach
            
    return current_set

def construct():
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    
    best = list(range(128, 136))  # 8 elements, known valid fallback
    best_len = len(best)
    
    pool = list(range(1, 257))
    
    # Try deterministic strategies first with bitmask version
    # Strategy 1: smallest first
    result = _greedy_build_bitmask(pool, sorted(pool))
    if len(result) > best_len:
        best = result
        best_len = len(result)
        
    # Strategy 2: largest first
    result = _greedy_build_bitmask(pool, sorted(pool, reverse=True))
    if len(result) > best_len:
        best = result
        best_len = len(result)
        
    # Strategy 3: Superincreasing
    superinc = []
    s = 0
    for i in range(1, 257):
        if i > s:
            superinc.append(i)
            s += i
        if len(superinc) >= 60:
            break
    if superinc:
        result = _greedy_build_bitmask(pool, superinc)
        if len(result) > best_len:
            best = result
            best_len = len(result)
            
    # Strategy 4: Primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True
    primes = [x for x in pool if is_prime(x)]
    result = _greedy_build_bitmask(pool, primes)
    if len(result) > best_len:
        best = result
        best_len = len(result)

    # Randomized greedy with multiple restarts
    rng = random.Random(42)
    # Increase restart count since bitmask is faster
    max_restarts = 5000
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random permutation of pool
        shuffled = pool[:]
        rng.shuffle(shuffled)
        
        # Biased strategies
        if restart % 5 == 0:
            # Prefer smaller values
            shuffled = sorted(shuffled, key=lambda x: x + rng.random() * 50)
        elif restart % 5 == 1:
            # Prefer larger values  
            shuffled = sorted(shuffled, key=lambda x: -x + rng.random() * 50)
        elif restart % 5 == 2:
            # Prefer "spread out" values (modulo 100)
            shuffled = sorted(shuffled, key=lambda x: (x * 101) % 257 + rng.random() * 50)
        elif restart % 5 == 3:
            # Fibonacci-like preference
            fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
            def fib_dist(x):
                return min(abs(x - f) for f in fibs)
            shuffled = sorted(shuffled, key=lambda x: fib_dist(x) + rng.random() * 50)
        # else: fully random
        
        result = _greedy_build_bitmask(pool, shuffled)
        if len(result) > best_len:
            best = result
            best_len = len(result)
            # Try to extend with remaining elements
            remaining = [x for x in pool if x not in best]
            rng.shuffle(remaining)
            # Rebuild with best first then remaining
            order = list(best) + remaining
            result2 = _greedy_build_bitmask(pool, order)
            if len(result2) > best_len:
                best = result2
                best_len = len(result2)
                
    # Final: try to extend best set
    if best_len < 60:
        remaining = [x for x in range(1, 257) if x not in best]
        rng.shuffle(remaining)
        # Build fresh with best elements first
        order = list(best) + remaining
        result = _greedy_build_bitmask(pool, order)
        if len(result) > best_len:
            best = result
            best_len = len(result)
            
    # Ensure return type is correct
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
