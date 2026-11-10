"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _has_4ap_slow(a, n):
    """Check if set a (as a boolean array of length n+1) contains a 4-term AP."""
    # a[i] is 1 if i is in the set
    # 4-term AP: x, x+d, x+2d, x+3d all in set
    # For each d, check all x such that x+3d <= n
    for d in range(1, n // 3 + 1):
        # x ranges from 1 to n-3d
        max_x = n - 3 * d
        for x in range(1, max_x + 1):
            if a[x] and a[x+d] and a[x+2*d] and a[x+3*d]:
                return True
    return False

def _count_3aps(a, n):
    """Count distinct 3-term APs (positive difference) in set represented by boolean array a."""
    count = 0
    # 3-term AP: x, x+d, x+2d with d>0, all in set
    for d in range(1, n // 2 + 1):
        max_x = n - 2 * d
        for x in range(1, max_x + 1):
            if a[x] and a[x+d] and a[x+2*d]:
                count += 1
    return count

def _has_4ap_fast(a_set, n):
    """Check if a_set (a Python set of ints) contains a 4-term AP."""
    for d in range(1, n // 3 + 1):
        for x in range(1, n - 3 * d + 1):
            if x in a_set and (x+d) in a_set and (x+2*d) in a_set and (x+3*d) in a_set:
                return True
    return False

def _count_3aps_fast(a_set, n):
    """Count distinct 3-term APs (positive difference) in set a_set."""
    count = 0
    for d in range(1, n // 2 + 1):
        for x in range(1, n - 2 * d + 1):
            if x in a_set and (x+d) in a_set and (x+2*d) in a_set:
                count += 1
    return count

def _construct_greedy():
    """Greedy construction: add x if it doesn't create a 4-AP with existing elements."""
    selected = set()
    for x in range(1, 257):
        # Check if adding x would create a 4-AP
        ok = True
        # x could be the 1st, 2nd, 3rd, or 4th term of a 4-AP
        # Case 1: x is 1st term: x, x+d, x+2d, x+3d
        for d in range(1, (256 - x) // 3 + 1):
            if (x+d) in selected and (x+2*d) in selected and (x+3*d) in selected:
                ok = False
                break
        if not ok:
            continue
        # Case 2: x is 2nd term: x-d, x, x+d, x+2d
        for d in range(1, x // 1 + 1):
            if x - d >= 1 and (x-d) in selected and (x+d) in selected and (x+2*d) in selected:
                ok = False
                break
        if not ok:
            continue
        # Case 3: x is 3rd term: x-2d, x-d, x, x+d
        for d in range(1, x // 2 + 1):
            if x - 2*d >= 1 and (x-2*d) in selected and (x-d) in selected and (x+d) in selected:
                ok = False
                break
        if not ok:
            continue
        # Case 4: x is 4th term: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if x - 3*d >= 1 and (x-3*d) in selected and (x-2*d) in selected and (x-d) in selected:
                ok = False
                break
        if not ok:
            continue
        selected.add(x)
    return sorted(selected)

def _sa_improve(initial_set, n, time_budget, seed):
    """Simulated annealing to improve a valid 4-AP-free set by maximizing 3-AP count."""
    rng = random.Random(seed)
    a_set = set(initial_set)
    current_score = _count_3aps_fast(a_set, n)
    best_set = set(a_set)
    best_score = current_score
    
    # Precompute all possible 3-APs and 4-APs for faster evaluation
    # Actually, let's use a different approach: track which elements are in the set
    # and use efficient delta updates
    
    # For SA, we try moves: add an element, remove an element, or swap
    # We need to check 4-AP constraint efficiently
    
    # Let's use a boolean array for fast membership
    a_arr = np.zeros(n + 1, dtype=np.int8)
    for x in a_set:
        a_arr[x] = 1
    
    def _check_4ap_with_change(arr, x, adding):
        """Check if adding/removing x creates/destroys a 4-AP. 
        If adding: check if x completes any 4-AP.
        If removing: just return False (removal can't create a 4-AP)."""
        if not adding:
            return False
        # Check all 4-APs that include x
        # x as 1st: x, x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            if arr[x+d] and arr[x+2*d] and arr[x+3*d]:
                return True
        # x as 2nd: x-d, x, x+d, x+2d
        for d in range(1, min(x, (n - x) // 2 + 1) + 1):
            if x - d >= 1 and arr[x-d] and arr[x+d] and arr[x+2*d]:
                return True
        # x as 3rd: x-2d, x-d, x, x+d
        for d in range(1, min(x // 2, n - x) + 1):
            if x - 2*d >= 1 and arr[x-2*d] and arr[x-d] and arr[x+d]:
                return True
        # x as 4th: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if x - 3*d >= 1 and arr[x-3*d] and arr[x-2*d] and arr[x-d]:
                return True
        return False
    
    def _delta_3ap_add(arr, x):
        """Count how many 3-APs are created by adding x."""
        delta = 0
        # x as 1st: x, x+d, x+2d
        for d in range(1, (n - x) // 2 + 1):
            if arr[x+d] and arr[x+2*d]:
                delta += 1
        # x as 2nd: x-d, x, x+d
        for d in range(1, min(x, n - x) + 1):
            if x - d >= 1 and arr[x-d] and arr[x+d]:
                delta += 1
        # x as 3rd: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            if x - 2*d >= 1 and arr[x-2*d] and arr[x-d]:
                delta += 1
        return delta
    
    def _delta_3ap_remove(arr, x):
        """Count how many 3-APs are destroyed by removing x (negative delta)."""
        delta = 0
        # x as 1st: x, x+d, x+2d
        for d in range(1, (n - x) // 2 + 1):
            if arr[x+d] and arr[x+2*d]:
                delta -= 1
        # x as 2nd: x-d, x, x+d
        for d in range(1, min(x, n - x) + 1):
            if x - d >= 1 and arr[x-d] and arr[x+d]:
                delta -= 1
        # x as 3rd: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            if x - 2*d >= 1 and arr[x-2*d] and arr[x-d]:
                delta -= 1
        return delta
    
    start_time = time.time()
    t0 = 100.0
    t_final = 0.1
    cooling = 0.9995
    
    steps = 0
    while time.time() - start_time < time_budget:
        t = max(t_final, t0 * (cooling ** steps))
        # Choose a random move
        move_type = rng.random()
        if move_type < 0.4:
            # Add a random element
            candidates = [x for x in range(1, n+1) if arr[x] == 0]
            if not candidates:
                steps += 1
                continue
            x = rng.choice(candidates)
            if _check_4ap_with_change(arr, x, True):
                steps += 1
                continue
            delta = _delta_3ap_add(arr, x)
            if delta >= 0 or rng.random() < np.exp(delta / t):
                arr[x] = 1
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(i for i in range(1, n+1) if arr[i])
            else:
                pass
        elif move_type < 0.7:
            # Remove a random element
            candidates = [x for x in range(1, n+1) if arr[x] == 1]
            if not candidates:
                steps += 1
                continue
            x = rng.choice(candidates)
            delta = _delta_3ap_remove(arr, x)
            # Removal is always valid (can't create 4-AP)
            if delta >= 0 or rng.random() < np.exp(delta / t):
                arr[x] = 0
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(i for i in range(1, n+1) if arr[i])
        else:
            # Swap: remove one, add another
            candidates_in = [x for x in range(1, n+1) if arr[x] == 1]
            candidates_out = [x for x in range(1, n+1) if arr[x] == 0]
            if not candidates_in or not candidates_out:
                steps += 1
                continue
            x_remove = rng.choice(candidates_in)
            x_add = rng.choice(candidates_out)
            # First remove x_remove
            arr[x_remove] = 0
            # Check if adding x_add creates a 4-AP
            if _check_4ap_with_change(arr, x_add, True):
                arr[x_remove] = 1
                steps += 1
                continue
            delta = _delta_3ap_remove(arr, x_remove) + _delta_3ap_add(arr, x_add)
            if delta >= 0 or rng.random() < np.exp(delta / t):
                arr[x_add] = 1
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(i for i in range(1, n+1) if arr[i])
            else:
                arr[x_remove] = 1
                # Revert
        steps += 1
    
    return sorted(best_set)

def construct():
    n = 256
    deadline = time.time() + 80  # 80 second budget, leave 10s margin
    
    # Start with greedy construction
    greedy = _construct_greedy()
    best_result = greedy
    best_score = _count_3aps_fast(set(greedy), n)
    
    # Try multiple SA restarts with different seeds
    seed = 42
    while time.time() < deadline:
        # Use a portion of remaining time for this SA run
        remaining = deadline - time.time()
        if remaining < 2:
            break
        sa_budget = min(remaining - 1, 15)  # Each SA run gets up to 15s
        if sa_budget < 1:
            break
        
        improved = _sa_improve(greedy, n, sa_budget, seed)
        improved_set = set(improved)
        improved_score = _count_3aps_fast(improved_set, n)
        
        # Verify no 4-AP
        if _has_4ap_fast(improved_set, n):
            # Shouldn't happen, but be safe
            pass
        else:
            if improved_score > best_score:
                best_score = improved_score
                best_result = improved
        
        seed += 1
    
    return best_result

# EVOLVE-BLOCK-END
