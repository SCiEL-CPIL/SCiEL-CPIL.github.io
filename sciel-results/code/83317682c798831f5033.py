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

def _find_addable(S, all_points):
    """Find points that can be added to S without creating an isosceles triangle."""
    S_set = set(S)
    addable = []
    for p in all_points:
        if p in S_set:
            continue
        valid = True
        for a in S:
            for b in S:
                if a == b:
                    continue
                if _is_isosceles(p, a, b):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            addable.append(p)
    return addable

def _local_search(S, all_points, deadline):
    """
    Local search on the space of maximal no-isosceles sets.
    Tries to remove points. If removal breaks maximality, performs a swap (remove one, add one)
    to explore the space of same-size maximal sets, hoping to find a removable point later.
    """
    S = list(S)
    
    # Track best size found in this local search
    best_local_size = len(S)
    best_local_set = S.copy()
    
    max_swaps = 1000 # Limit swaps to avoid infinite loops in one trial
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
            
        if not S:
            break
            
        random.shuffle(S)
        improved = False
        
        for v in S:
            # Try removing v
            S_temp = [p for p in S if p != v]
            
            # Check if S_temp is maximal
            # If it is, we accept the removal (size decreases)
            # If not, we find addable points and perform a swap
            
            addable = _find_addable(S_temp, all_points)
            
            if not addable:
                # S_temp is maximal!
                S = S_temp
                if len(S) < best_local_size:
                    best_local_size = len(S)
                    best_local_set = S.copy()
                improved = True
                break # Restart with smaller S
            
            else:
                # S_temp is not maximal. We must add a point to restore independence/maximality context.
                # We pick a random addable point u.
                # S_new = S_temp U {u}
                # This maintains size.
                # We accept this move to explore the neighborhood.
                u = random.choice(addable)
                S = S_temp + [u]
                # We do not update best_local_set here as size is same, 
                # but this new S might be more "prunable".
                # We break to restart the shuffle/try-remove loop with the new S.
                break
        
        if not improved:
            # We performed a swap. Continue loop to try removing from new S.
            # If we are stuck (no improvement in size for a while), we might want to restart?
            # But the random nature of shuffle should help.
            pass
            
    return best_local_set

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(all_points)
    
    best_set = None
    best_size = float('inf')
    
    # Run multiple randomized greedy attempts
    num_trials = 1000  # We will break early if time runs out
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        random.seed(trial)
        
        chosen = []
        chosen_set = set()
        
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
                
            # Heuristic: Maximize coverage
            if len(valid_points) <= 3:
                p = random.choice(valid_points)
            else:
                max_cov = -1
                best_p = None
                if not chosen:
                    p = random.choice(valid_points)
                else:
                    for p in valid_points:
                        cov = 0
                        for q in valid_points:
                            if p == q:
                                continue
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
                    if random.random() < 0.1:
                        p = random.choice(valid_points)
                    else:
                        p = best_p
            
            chosen.append(p)
            chosen_set.add(p)
            
        # End of construction
        if chosen:
            # Run local search on this greedy result
            # Allocate some time for local search
            ls_deadline = min(deadline, time.time() + 2.0) # 2 seconds per trial for local search?
            # If we have many trials, we can't spend 2s each.
            # Let's spend less time if we have done many trials.
            # Or just use a fixed small budget.
            
            # Better: Use remaining time proportionally?
            # Let's just run local search with a tight deadline.
            ls_start = time.time()
            # Limit local search to 1 second per trial to allow many trials
            ls_deadline = ls_start + 1.0
            
            improved_set = _local_search(chosen, all_points, ls_deadline)
            
            if len(improved_set) < best_size:
                best_size = len(improved_set)
                best_set = improved_set.copy()
                
            # If we found a very small set, we can spend more time polishing it?
            # For now, just keep the best.

    if best_set is None:
        # Fallback
        chosen = []
        for p in all_points:
            if all(not _is_isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
                chosen.append(p)
        return chosen
        
    return best_set

# EVOLVE-BLOCK-END
