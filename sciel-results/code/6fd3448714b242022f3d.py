"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 256

# EVOLVE-BLOCK-START
import numpy as np

def _fast_sidon_check(s):
    """Check if set s is a Sidon set. Returns True if Sidon."""
    sums = set()
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i, n):
            val = si + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def _get_sums_set(s):
    """Compute the set of all pairwise sums a+b for a<=b in s."""
    sums = set()
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i, n):
            sums.add(si + s[j])
    return sums

def _is_maximal_fast(s, N, sums_set):
    """
    Check if Sidon set s is inclusion-maximal in [1,N].
    """
    s_set = set(s)
    for x in range(1, N + 1):
        if x in s_set:
            continue
        if 2 * x in sums_set:
            continue
        blocked = False
        for a in s:
            if (x + a) in sums_set:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _build_greedy_blocking(N):
    """
    Build a Sidon set using a heuristic that prioritizes elements
    which block the most other elements.
    """
    selected = []
    sums = set()
    # blocked[y] = True if y is blocked by current selected set
    blocked = [False] * (N + 1)
    
    # Initialize blocked state?
    # Initially no sums, so nothing is blocked.
    # But we need to update blocked as we add elements.
    
    # To speed up, we can precompute potential conflicts?
    # For now, calculate newly blocked on the fly.
    
    while True:
        best_x = -1
        best_gain = -1
        best_new_sums = None
        
        # Find all valid candidates
        candidates = []
        for x in range(1, N + 1):
            if x in selected:
                continue
            
            # Check if x is valid (Sidon compatible)
            new_sums = [2 * x]
            for a in selected:
                new_sums.append(x + a)
                
            # Check internal collisions in new_sums
            if len(set(new_sums)) != len(new_sums):
                continue
            
            # Check collision with existing sums
            if sums.intersection(new_sums):
                continue
            
            candidates.append((x, set(new_sums)))
            
        if not candidates:
            break
            
        # Evaluate each candidate
        for x, new_sums_set in candidates:
            # Calculate how many elements are newly blocked
            # An element y is blocked if 2y in sums_total or y+a in sums_total
            # sums_total = sums U new_sums_set
            
            # We can just check the condition for each y not in selected
            gain = 0
            for y in range(1, N + 1):
                if y in selected:
                    continue
                if blocked[y]:
                    continue
                    
                # Check if y is blocked by sums U new_sums_set
                is_blocked = False
                if 2 * y in sums or 2 * y in new_sums_set:
                    is_blocked = True
                else:
                    for a in selected:
                        if (y + a) in sums or (y + a) in new_sums_set:
                            is_blocked = True
                            break
                            
                if is_blocked:
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_x = x
                best_new_sums = new_sums_set
                
        # Add best_x
        if best_x == -1:
            break
            
        selected.append(best_x)
        sums.update(best_new_sums)
        
        # Update blocked array
        for y in range(1, N + 1):
            if y in selected:
                continue
            if blocked[y]:
                continue
            if 2 * y in sums:
                blocked[y] = True
            else:
                for a in selected:
                    if (y + a) in sums:
                        blocked[y] = True
                        break
                        
    return selected

def construct():
    best_size = N + 1
    best_set = None
    start_time = time.time()
    time_limit = 28.0
    
    random.seed(42)
    
    # Run multiple attempts with randomization in tie-breaking or initial noise?
    # The blocking heuristic is deterministic given the order of candidates?
    # We iterate x from 1 to N. If we shuffle, we might get different results?
    # Actually, the gain calculation depends only on the set, not the order of checking candidates.
    # But if two candidates have the same gain, we pick the first one (smallest x).
    # To explore different structures, we can add random noise to the gain or shuffle candidates.
    
    for attempt in range(50): # Reduced attempts as each is slower
        if time.time() - start_time > time_limit:
            break
        
        # Introduce randomness: shuffle the range of x for candidate evaluation?
        # Or just run the deterministic one?
        # Let's try to randomize the "tie-breaking" by adding small random noise to gain.
        
        # Since _build_greedy_blocking is deterministic, we need to modify it to accept a random seed or noise.
        # For now, let's just run it once? No, let's try to randomize the selection among top-k candidates.
        
        # Simpler: Just run the deterministic version. If it's good, we're done.
        # If not, we might need a more complex search.
        # Given the 30s limit, 50 deterministic runs is fast.
        
        s = _build_greedy_blocking(N)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            
        # If we find a very small set, we can break?
        # Theoretical min is ~ sqrt(N) = 16.
        # If we get 18, it's good.
        
    if best_set is None:
        return list(range(1, N + 1)) # Fallback, though invalid
        
    return best_set
# EVOLVE-BLOCK-END
