import time
import random
import sys

_START_TIME = time.time()
_DEADLINE = 88.0  # Safety margin before 90s kill

def _is_sidon_with_diagonals(subset):
    """Check if subset has all unordered pair sums unique including diagonals (2a[i])."""
    n = len(subset)
    if n <= 1:
        return True
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _exact_max_sidon_subset_size(arr, time_limit=0.05):
    """
    Exact B&B solver to find the maximum Sidon subset size.
    Optimized for N=12. Returns the size.
    """
    n = len(arr)
    if n <= 1:
        return n
    
    # Sort to help with pruning? Not strictly necessary for B&B on indices, 
    # but consistent ordering helps.
    arr_sorted = sorted(arr)
    best = 0
    start_time = time.time()
    
    # We use a recursive DFS with include/exclude.
    # State: idx, current_size, current_sums (set of sums), current_elements (list)
    # To speed up, we can precompute some things or use bitmasks if sums were small, 
    # but sums are up to 256, so a set of ints is fine.
    
    def dfs(idx, current_size, current_sums, current_elements):
        nonlocal best, start_time
        
        # Time check
        if time.time() - start_time > time_limit:
            return False
            
        # Pruning: if we can't beat best
        if current_size + (n - idx) <= best:
            return False
            
        # Update best
        if current_size > best:
            best = current_size
            if best == n:
                return True # Can't do better than n
            
        if idx == n:
            return False
        
        x = arr_sorted[idx]
        
        # Option 1: Include x
        # Check if x can be added to current_elements
        can_add = True
        new_sums = set()
        for y in current_elements:
            s = x + y
            if s in current_sums:
                can_add = False
                break
            new_sums.add(s)
        
        if can_add:
            s_diag = 2 * x
            if s_diag in current_sums:
                can_add = False
            else:
                new_sums.add(s_diag)
        
        if can_add:
            # Try including x first (often finds larger solutions faster, improving pruning)
            if dfs(idx + 1, current_size + 1, current_sums | new_sums, current_elements + [x]):
                return True
        
        # Option 2: Exclude x
        if dfs(idx + 1, current_size, current_sums, current_elements):
            return True
            
        return False

    dfs(0, 0, set(), [])
    return best

def construct():
    global _START_TIME, _DEADLINE
    
    # Ensure valid output: 12 distinct integers in [0, 128]
    
    # Initial candidates: Try some structured sets that might have low Sidon numbers
    # Clustering creates many sum collisions, which reduces the size of the largest Sidon subset.
    # For example, if many numbers are close, a_i + a_j = a_k + a_l is more likely.
    
    candidates = []
    
    # 1. Dense cluster at 0
    candidates.append(list(range(12)))
    
    # 2. Two dense clusters
    candidates.append([0, 1, 2, 3, 4, 5, 60, 61, 62, 63, 64, 65])
    
    # 3. Arithmetic progression with small step (more collisions than large step)
    candidates.append([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    
    # 4. Random dense sets
    rng = random.Random(42)
    for _ in range(20):
        c = sorted(rng.sample(range(0, 129), 12))
        candidates.append(c)
        
    # 5. Try to minimize spread
    candidates.append(sorted(rng.sample(range(0, 30), 12)))
    candidates.append(sorted(rng.sample(range(0, 20), 12)))
    
    best_arr = None
    best_score = 12 # Max possible
    
    for c in candidates:
        if len(set(c)) != 12:
            continue
        if any(x < 0 or x > 128 for x in c):
            continue
        score = _exact_max_sidon_subset_size(c, time_limit=0.1)
        if score < best_score:
            best_score = score
            best_arr = c[:]
            
    # If no candidate worked (shouldn't happen), fallback
    if best_arr is None:
        best_arr = list(range(12))
        best_score = _exact_max_sidon_subset_size(best_arr, time_limit=0.1)
        
    # Simulated Annealing
    # Objective: Minimize best_score (max Sidon subset size)
    # Move: Swap one element with a random other in [0, 128]
    
    current_arr = best_arr[:]
    current_score = best_score
    best_arr_found = best_arr[:]
    best_score_found = best_score
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    rng = random.Random(123)
    
    while time.time() < _DEADLINE:
        # Generate neighbor
        new_arr = current_arr[:]
        idx = rng.randint(0, 11)
        old_val = new_arr[idx]
        
        # Pick a new value
        new_val = rng.randint(0, 128)
        
        # Ensure distinctness
        if new_val == old_val:
            continue
        if new_val in new_arr[:idx] or new_val in new_arr[idx+1:]:
            continue
            
        new_arr[idx] = new_val
        
        # Evaluate
        new_score = _exact_max_sidon_subset_size(new_arr, time_limit=0.05)
        
        delta = new_score - current_score
        
        if delta < 0:
            # Improvement
            current_arr = new_arr
            current_score = new_score
            if current_score < best_score_found:
                best_score_found = current_score
                best_arr_found = new_arr[:]
        else:
            # Accept with probability exp(-delta / T)
            # Since delta is integer >= 0, and T is small, this is rare for large T
            # But for delta=0, always accept to explore plateau
            if delta == 0:
                if rng.random() < 0.5:
                    current_arr = new_arr
            else:
                prob = exp_val = 2.718281828 ** (-delta / T)
                if rng.random() < prob:
                    current_arr = new_arr
                    current_score = new_score
                    
        T *= cooling_rate
        
        # Early stop if we find a very low score (e.g., 4 or 5)
        # The theoretical minimum for N=12 is likely around 4-5.
        # If we hit 4, we can stop or keep searching for 3 (unlikely).
        if best_score_found <= 4:
            # Keep searching a bit more to see if we can find 3?
            # But 3 is extremely hard. 4 is a good target.
            # Let's run for a bit longer to be sure.
            pass

    # Final validation
    assert len(best_arr_found) == 12
    assert len(set(best_arr_found)) == 12
    assert all(0 <= x <= 128 for x in best_arr_found)
    
    return best_arr_found

# EVOLVE-BLOCK-START
# (construct is defined above; this block is for SCiEL evolution markers)
# EVOLVE-BLOCK-END
