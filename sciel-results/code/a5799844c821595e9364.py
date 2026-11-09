import itertools
import random
import time
import math

N = 16

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_isosceles(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(all_points)
    
    # Precompute isosceles relationships?
    # For speed, we can precompute for each point p, the list of points q such that p and q are "compatible" with a third point?
    # Actually, let's just use the function. N=16 is small enough.
    
    best_set = None
    best_size = float('inf')
    
    # Run multiple randomized greedy attempts
    # Strategy: Greedy with Maximum Coverage
    # At each step, pick the valid point that blocks the most other valid points.
    
    num_trials = 1000  # We will break early if time runs out
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        random.seed(trial) # Or just use global random state
        
        chosen = []
        chosen_set = set()
        
        # Valid candidates: all points initially
        # We maintain a list of points that can still be added
        # Instead of checking all points every time, we can just iterate and filter.
        
        while True:
            if time.time() > deadline:
                break
                
            # Find all valid points
            valid_points = []
            for p in all_points:
                if p in chosen_set:
                    continue
                # Check if p can be added to chosen
                valid = True
                for a in chosen:
                    for b in chosen:
                        if a == b:
                            continue
                        if _is_isosceles(p, a, b):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    valid_points.append(p)
            
            if not valid_points:
                break
                
            # If we have valid points, pick one
            # Heuristic: Maximize coverage (number of other valid points blocked)
            # To speed up, if len(valid_points) is small, just pick random.
            if len(valid_points) <= 3:
                p = random.choice(valid_points)
            else:
                # Calculate coverage for each valid point
                # Coverage(p) = number of q in valid_points \ {p} such that 
                # adding p makes q invalid.
                # q becomes invalid if exists a in chosen such that {p, a, q} is isosceles.
                # Note: q is currently valid, so {a, b, q} is not isosceles for any a,b in chosen.
                # So the only new isosceles triangles involving q must involve p.
                
                max_cov = -1
                best_p = None
                # Optimization: If chosen is empty, coverage is 0 for all. Just pick random.
                if not chosen:
                    p = random.choice(valid_points)
                else:
                    # Sample if too many? 256 points max. 256*256*16 checks is ~1M ops. Fine.
                    for p in valid_points:
                        cov = 0
                        for q in valid_points:
                            if p == q:
                                continue
                            # Check if q is blocked by p given chosen
                            blocked = False
                            for a in chosen:
                                if _is_isosceles(p, a, q):
                                    blocked = True
                                    break
                            if blocked:
                                cov += 1
                        if cov > max_cov:
                            max_cov = cov
                            best_p = p
                    # Add some randomness to tie-breaking
                    # Find all with max_cov and pick random
                    candidates = [p for p in valid_points if (not chosen or True)] # Re-calc? No, just pick from those with max cov.
                    # To avoid re-calculation, we can just pick best_p with some probability, or random from top K.
                    # For simplicity, pick best_p. Or randomize slightly.
                    if random.random() < 0.1:
                        p = random.choice(valid_points)
                    else:
                        p = best_p
            
            chosen.append(p)
            chosen_set.add(p)
            
        # End of construction
        if chosen:
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_set = chosen.copy()
        
        # Optional: Local Search / Pruning
        # Try to remove points from best_set?
        # If we remove p, is the set still maximal?
        # If yes, update best_set.
        # This is expensive to check for maximality.
        # Check: For all q not in S', S' U {q} is not no-isosceles.
        # This is O(N^2 * |S'|^2).
        # We can do this on the current best_set occasionally.
        
        if best_set and time.time() < deadline - 10:
            # Try pruning on the best set found so far
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                s_copy = best_set.copy()
                s_set = set(s_copy)
                for i, p in enumerate(s_copy):
                    # Try removing p
                    s_temp = [x for x in s_copy if x != p]
                    s_temp_set = set(s_temp)
                    
                    # Check if s_temp is maximal
                    # Maximal means: for all q in all_points \ s_temp, 
                    # s_temp U {q} contains an isosceles triangle.
                    is_maximal = True
                    for q in all_points:
                        if q in s_temp_set:
                            continue
                        # Check if q can be added to s_temp
                        valid_q = True
                        for a in s_temp:
                            for b in s_temp:
                                if a == b:
                                    continue
                                if _is_isosceles(q, a, b):
                                    valid_q = False
                                    break
                            if not valid_q:
                                break
                        if valid_q:
                            # q can be added, so s_temp is NOT maximal
                            is_maximal = False
                            break
                    
                    if is_maximal:
                        # s_temp is a smaller maximal set
                        best_set = s_temp
                        best_size = len(best_set)
                        improved = True
                        break # Restart the while loop with the new best_set
    
    if best_set is None:
        # Fallback: Return a simple greedy set
        chosen = []
        for p in all_points:
            if all(not _is_isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
                chosen.append(p)
        return chosen
        
    return best_set

# EVOLVE-BLOCK-END
