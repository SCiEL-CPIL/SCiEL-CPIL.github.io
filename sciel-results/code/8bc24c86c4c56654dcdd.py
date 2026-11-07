import time
import random
from itertools import combinations

def _verify(arr):
    """
    Verify if the set satisfies the cardinality-separated subset sum property.
    Uses bitset DP for efficiency.
    """
    n = len(arr)
    if n < 1 or n > 60:
        return False
    
    # Check types and range
    seen = set()
    for v in arr:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v < 1 or v > 256:
            return False
        if v in seen:
            return False
        seen.add(v)
        
    # dp[k] is a bitset where bit s is set iff there exists a subset of cardinality k summing to s
    # Max possible sum is n * 256.
    # We use Python integers as bitsets.
    max_sum = sum(arr)
    dp = [0] * (n + 1)
    dp[0] = 1  # Sum 0 achievable with 0 elements
    
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
            
    # Check disjointness of sums for different cardinalities
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
        
    return True

def _try_add(arr, v, dp_state):
    """
    Try adding v to arr.
    Returns (success, new_dp_state)
    dp_state: list of bitsets dp[k]
    """
    n = len(arr)
    max_n = n + 1
    new_dp = [0] * (max_n + 1)
    
    # Copy existing dp
    for k in range(n + 1):
        new_dp[k] = dp_state[k]
        
    # Update with v
    # We must iterate backwards to avoid using v multiple times
    for k in range(max_n, 0, -1):
        new_dp[k] |= (new_dp[k - 1] << v)
        
    # Check disjointness
    acc = 0
    for k in range(max_n + 1):
        if new_dp[k] & acc:
            return False, None
        acc |= new_dp[k]
        
    return True, new_dp

def _grow_greedy(start, deadline, max_time_per_step=0.5):
    """
    Grow the set greedily.
    """
    cur = list(start)
    cur_set = set(cur)
    n = len(cur)
    
    # Initialize DP
    max_sum = sum(cur)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in cur:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
            
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        # Try adding smallest valid element first? Or random?
        # Smallest might be safer for keeping sums low?
        # Actually, we want to fit in 1..256.
        candidates = [v for v in range(1, 257) if v not in cur_set]
        
        # Heuristic: try candidates that minimize the "gap" violation or something?
        # Just try all, pick the first one that works.
        # To allow for larger sets, maybe try larger numbers first?
        # No, let's just iterate.
        
        for v in candidates:
            success, new_dp = _try_add(cur, v, dp)
            if success:
                cur.append(v)
                cur_set.add(v)
                dp = new_dp
                n = len(cur)
                improved = True
                break
                
    return cur

def _reverse_greedy(n_target, deadline):
    """
    Reverse greedy: start with n_target elements, try to find a valid set.
    Strategy: use consecutive integers in the highest range, then try variations.
    """
    # First, try consecutive ranges from the top
    for start in range(256 - n_target + 1, 0, -1):
        if time.monotonic() > deadline:
            break
        arr = list(range(start, start + n_target))
        if _verify(arr):
            return arr
    return None

def _local_search(arr, deadline):
    """
    Local search: try to swap elements to improve the set.
    """
    cur = list(arr)
    cur_set = set(cur)
    n = len(cur)
    
    # Compute current DP
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in cur:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    
    # Try to add elements by removing others
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        
        # Try each element in cur
        for i, old_v in enumerate(cur):
            if time.monotonic() > deadline:
                break
            
            # Try replacing old_v with a new value
            for new_v in range(1, 257):
                if new_v in cur_set:
                    continue
                
                # Build new array
                new_arr = cur[:i] + [new_v] + cur[i+1:]
                new_arr_set = set(new_arr)
                
                # Check if valid
                if _verify(new_arr):
                    # Try to grow
                    grown = _grow_greedy(new_arr, deadline)
                    if len(grown) > len(cur):
                        cur = grown
                        cur_set = set(cur)
                        n = len(cur)
                        improved = True
                        break
            
            if improved:
                break
    
    return cur

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0  # Safety margin
    
    best = None
    best_len = 0
    
    # Strategy 1: Reverse greedy - try to find large consecutive ranges
    for n in range(40, 8, -1):
        if time.monotonic() > deadline - 2:
            break
        
        arr = _reverse_greedy(n, deadline)
        if arr is not None:
            best = arr
            best_len = n
            # Try to grow it
            if time.monotonic() < deadline - 5:
                grown = _grow_greedy(arr, deadline)
                if len(grown) > best_len:
                    best = grown
                    best_len = len(grown)
            break  # Found a valid set, stop searching for larger n
    
    # Strategy 2: If no large set found, try medium sizes
    if best is None or best_len < 20:
        for n in range(20, 9, -1):
            if time.monotonic() > deadline - 2:
                break
            
            arr = _reverse_greedy(n, deadline)
            if arr is not None:
                if best is None or n > best_len:
                    best = arr
                    best_len = n
                    # Try to grow it
                    if time.monotonic() < deadline - 5:
                        grown = _grow_greedy(arr, deadline)
                        if len(grown) > best_len:
                            best = grown
                            best_len = len(grown)
                break
    
    # Strategy 3: Random search for larger n
    rng = random.Random(42)
    
    target_len = best_len + 1 if best is not None else 15
    
    attempts = 0
    while time.monotonic() < deadline - 5 and attempts < 2000:
        attempts += 1
        
        # Generate a random set of size target_len
        # Bias towards tight clusters
        # Pick a base, pick tight offsets
        if rng.random() < 0.5:
            # Tight cluster
            width = rng.randint(target_len, target_len + 5)
            start = rng.randint(1, 257 - width)
            # Pick target_len distinct numbers in [start, start+width]
            pool = rng.sample(range(start, start + width + 1), target_len)
        else:
            # Uniform random
            pool = rng.sample(range(1, 256), target_len)
            
        if _verify(pool):
            # Grow it
            grown = _grow_greedy(pool, deadline)
            if len(grown) > best_len:
                best = grown
                best_len = len(grown)
            target_len = best_len + 1
    
    # Strategy 4: Local search on best found
    if best is not None and time.monotonic() < deadline - 2:
        improved_best = _local_search(best, deadline)
        if len(improved_best) > best_len:
            best = improved_best
            best_len = len(best)
    
    if best is None:
        # Fallback: a known small valid set
        best = [1, 2, 4, 8, 16, 32, 64, 128]
        # Verify fallback
        if not _verify(best):
             # Very conservative
             best = list(range(249, 257)) # 8 consecutive
             if not _verify(best):
                 best = [256]
    
    # Ensure Python ints
    best = [int(v) for v in best]
    return best
