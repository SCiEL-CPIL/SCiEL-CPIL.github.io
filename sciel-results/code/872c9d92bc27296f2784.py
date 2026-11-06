import time
import random

def is_valid_add(S, y, n=512):
    """
    Check if adding y to S creates a non-constant 4-term arithmetic progression.
    A 4-AP is {a, a+d, a+2d, a+3d} with d != 0.
    y can be any of the 4 terms.
    """
    # y as 1st term: a=y. Need y+d, y+2d, y+3d in S
    # Max d such that y+3d <= n => d <= (n-y)//3
    max_d = (n - y) // 3
    for d in range(1, max_d + 1):
        if (y + d) in S and (y + 2*d) in S and (y + 3*d) in S:
            return False
            
    # y as 2nd term: a+d=y => a=y-d. Need y-d, y+d, y+2d in S
    # y-d >= 1 => d <= y-1
    # y+2d <= n => d <= (n-y)//2
    max_d = min(y - 1, (n - y) // 2)
    for d in range(1, max_d + 1):
        if (y - d) in S and (y + d) in S and (y + 2*d) in S:
            return False
            
    # y as 3rd term: a+2d=y => a=y-2d. Need y-2d, y-d, y+d in S
    # y-2d >= 1 => d <= (y-1)//2
    # y+d <= n => d <= n-y
    max_d = min((y - 1) // 2, n - y)
    for d in range(1, max_d + 1):
        if (y - 2*d) in S and (y - d) in S and (y + d) in S:
            return False
            
    # y as 4th term: a+3d=y => a=y-3d. Need y-3d, y-2d, y-d in S
    # y-3d >= 1 => d <= (y-1)//3
    max_d = (y - 1) // 3
    for d in range(1, max_d + 1):
        if (y - 3*d) in S and (y - 2*d) in S and (y - d) in S:
            return False
            
    return True

def greedy_construct(randomize=False, n=512):
    S = set()
    candidates = list(range(1, n + 1))
    if randomize:
        random.shuffle(candidates)
    for y in candidates:
        if is_valid_add(S, y, n):
            S.add(y)
    return S

def local_search(S, deadline, n=512):
    """
    Perform local search on S to maximize size while maintaining no 4-AP.
    Uses add and swap moves.
    """
    S = set(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add
        candidates_to_add = [y for y in range(1, n + 1) if y not in S]
        random.shuffle(candidates_to_add)
        
        added = False
        for y in candidates_to_add:
            if is_valid_add(S, y, n):
                S.add(y)
                improved = True
                added = True
                break
                
        if added:
            continue
            
        # 2. Try to swap
        # To be efficient, we can limit the number of swap attempts per iteration
        candidates_to_swap_out = list(S)
        random.shuffle(candidates_to_swap_out)
        
        # We don't want to try all pairs (x,y). 
        # Heuristic: Try swapping an element x with a random y not in S.
        # If valid, accept.
        
        swaps_made = 0
        for x in candidates_to_swap_out:
            if time.time() >= deadline:
                break
                
            # Generate a few random y's to try swapping with x
            # Or iterate through a subset of candidates_to_add
            # Let's try a few random y's from the complement
            comp = [y for y in range(1, n + 1) if y not in S]
            random.shuffle(comp)
            
            for y in comp[:10]: # Try 10 random y's for this x
                # Check if S - {x} + {y} is valid
                # This is equivalent to checking if y is valid in S-{x}
                # But we must ensure no 4-AP is formed involving y and elements in S-{x}
                # AND that removing x doesn't break anything (it doesn't, it only reduces APs)
                
                # Efficient check:
                # Temporarily remove x
                S.remove(x)
                valid = is_valid_add(S, y, n)
                if valid:
                    S.add(y)
                    improved = True
                    swaps_made += 1
                    break
                else:
                    S.add(x) # Restore x
            
            if improved:
                break
                
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    best_set = greedy_construct(randomize=False)
    best_size = len(best_set)
    
    # Perform local search on the initial greedy set
    current_set = local_search(best_set, deadline)
    if len(current_set) > best_size:
        best_set = current_set
        best_size = len(best_set)
        
    # Try multiple randomized restarts if time permits
    while time.time() < deadline:
        # Randomized greedy initialization
        S_init = greedy_construct(randomize=True)
        
        # Local search on this new set
        S_opt = local_search(S_init, deadline)
        
        if len(S_opt) > best_size:
            best_set = S_opt
            best_size = len(best_set)
            
    return sorted(best_set)
