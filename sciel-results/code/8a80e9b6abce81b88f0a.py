import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Shift to 0-indexed for Golomb ruler standard
    # Required: {1, 2, 5, 11} -> {0, 1, 4, 10}
    required = [0, 1, 4, 10]
    
    # Initial state
    S = required[:]
    used_diffs = set()
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            used_diffs.add(S[j] - S[i])
            
    best_set = S[:]
    best_size = len(S)
    
    # Pre-compute valid next candidates? No, dynamic.
    
    def search(current_S, current_diffs, current_size):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if current_size > best_size:
            best_size = current_size
            best_set = current_S[:]
            
        # Pruning: 
        # If we can't beat best_size, stop.
        # Max possible size is limited by range and diff count.
        # Loose bound: current_size + (283 - current_S[-1])
        if current_size + (283 - current_S[-1]) <= best_size:
            return
            
        # Try adding next element x
        # x must be > current_S[-1]
        # To pack tightly, iterate x from current_S[-1]+1 upwards
        # But we can skip x if it creates duplicate diff
        
        last = current_S[-1]
        # Optimization: We don't need to try all x up to 283 immediately if we know we need to fill more.
        # But we don't know how many more we need.
        
        # Heuristic: Try x that allows more room for future elements?
        # For now, simple iteration.
        
        for x in range(last + 1, 284):
            # Check if x creates duplicate differences with existing elements
            valid = True
            new_diffs = []
            for y in current_S:
                d = x - y
                if d in current_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            # If valid, recurse
            new_S = current_S + [x]
            new_diffs = current_diffs | set(new_diffs)
            
            search(new_S, new_diffs, current_size + 1)
            
            # Time check inside loop?
            if time.time() > deadline:
                return

    search(S, used_diffs, best_size)
    
    # Map back to 1..284
    return [x + 1 for x in best_set]
