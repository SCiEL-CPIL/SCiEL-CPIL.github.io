import time
import random
from math import gcd
from collections import Counter
from itertools import combinations

# Precompute GCD matrix for 1..128
# Indices 0..127 correspond to 1..128
GCD_MAT = [[gcd(i+1, j+1) for j in range(128)] for i in range(128)]

def get_conflicts(x, S):
    """
    Returns list of pairs (a, b) from S such that
    gcd(a, b) == gcd(a, x) == gcd(b, x)
    """
    if not S:
        return []
    
    sx = x - 1
    s_list = list(S)
    n = len(s_list)
    conflicts = []
    
    # Optimization: Precompute gcds with x for all in S
    # gcd_x = {y: GCD_MAT[sx][y-1] for y in S}
    # But we access GCD_MAT directly in inner loop.
    
    for i in range(n):
        a = s_list[i]
        ga = GCD_MAT[sx][a-1]
        for j in range(i + 1, n):
            b = s_list[j]
            gb = GCD_MAT[sx][b-1]
            if ga == gb:
                # Potential conflict, verify gcd(a,b)
                if GCD_MAT[a-1][b-1] == ga:
                    conflicts.append((a, b))
    return conflicts

def greedy_init(order):
    S = set()
    for x in order:
        # Check if x conflicts with any pair in S
        # Efficient check:
        # For any pair (a,b) in S, if gcd(a,b)==gcd(a,x)==gcd(b,x), conflict.
        # We can just call get_conflicts and see if it's empty.
        if not get_conflicts(x, S):
            S.add(x)
    return S

def local_search(S, time_limit):
    """
    Improves S via local search.
    Returns the best set found.
    """
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    start_time = time.time()
    last_improvement = start_time
    
    while time.time() < time_limit:
        # 1. Try to add any non-conflicting element
        added = False
        candidates = [x for x in range(1, 129) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            if not get_conflicts(x, S):
                S.add(x)
                added = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                    last_improvement = time.time()
                break
        
        if added:
            continue
        
        # 2. Stuck. Perform a kick (swap)
        if not candidates:
            break # All elements in S
        
        x = random.choice(candidates)
        conflicts = get_conflicts(x, S)
        
        if not conflicts:
            # Should have been caught above, but just in case
            S.add(x)
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
                last_improvement = time.time()
            continue
        
        # Find victim to remove
        # Count how many conflicts each element in S is involved in
        elem_counts = Counter()
        for a, b in conflicts:
            elem_counts[a] += 1
            elem_counts[b] += 1
        
        # Remove the most frequent element
        victim = max(elem_counts, key=elem_counts.get)
        
        S.remove(victim)
        S.add(x)
        
        # Check if this kick allows immediate addition?
        # The loop will handle it, but we can check if size increased (it didn't, swap)
        # However, if we removed a 'bad' element, maybe we can add another 'good' one?
        # The next iteration of the while loop will try to add.
        
        # Stagnation check
        if time.time() - last_improvement > 5.0:
            # Revert to best and perturb
            S = set(best_S)
            # Remove 2 random elements
            to_remove = random.sample(list(S), min(2, len(S)))
            for v in to_remove:
                S.remove(v)
            # Add 2 random elements not in S
            candidates = [x for x in range(1, 129) if x not in S]
            if candidates:
                to_add = random.sample(candidates, min(2, len(candidates)))
                for v in to_add:
                    S.add(v)
            last_improvement = time.time()
            
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial Greedy with natural order
    S = greedy_init(range(1, 129))
    best_S = set(S)
    
    # Run local search
    # We can run multiple restarts if time permits
    # But local search itself restarts internally.
    # Let's just run local search on the initial result.
    # If it finishes fast, we can try another init.
    
    while time.time() < deadline:
        # Run local search with a short time slice or until it finishes?
        # local_search has its own time limit.
        # Let's give it the remaining time minus a small buffer.
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # If we just started, use full remaining time.
        # If we are restarting, use full remaining time.
        
        current_S = local_search(best_S, deadline)
        
        if len(current_S) > len(best_S):
            best_S = current_S
        else:
            # If no improvement, try a different initialization
            # Randomized greedy
            order = list(range(1, 129))
            random.shuffle(order)
            S = greedy_init(order)
            # Quick local search on this new start?
            # Or just check if it's better?
            # Let's run local search on it.
            # But we need to be careful not to loop forever if it's worse.
            # local_search returns best found.
            
            # To avoid wasting time on bad inits, we can do a very short local search
            # or just check greedy size.
            # Greedy size is usually lower than local search.
            # Let's just run local_search on this new S with remaining time.
            # If it doesn't beat best_S, we loop again with another shuffle.
            
            # Actually, the loop above `while time.time() < deadline`
            # will call local_search again.
            # If we pass `best_S` again, it will just search around best_S.
            # We want to search around `S` (the new greedy).
            
            # So, modify logic:
            # 1. Get a candidate S (either current best or new greedy)
            # 2. Run local search on it.
            # 3. Update best.
            
            # The code above runs local_search on best_S.
            # Let's restructure.
            
            pass # The pass is a placeholder, logic below handles it.

    # Restructured main loop for clarity in final code:
    # 1. Init best_S with greedy.
    # 2. Loop:
    #    a. If time low, break.
    #    b. Run local_search on best_S (or a perturbed version) for remaining time?
    #    No, local_search runs until its own time limit.
    #    If we pass `deadline`, it runs until deadline.
    #    So we should only call it once?
    #    If local_search handles restarts internally (via stagnation check),
    #    then one call is enough.
    
    # The `local_search` function I wrote has internal restarts.
    # So calling it once with `deadline` should be sufficient.
    
    # Let's just call it once.
    
    # Wait, the code above in `construct` has a while loop.
    # Let's simplify `construct`.
    
    pass

# Final construct implementation
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Initial Greedy
    S = greedy_init(range(1, 129))
    
    # 2. Local Search (with internal restarts)
    best_S = local_search(S, deadline)
    
    return list(best_S)
