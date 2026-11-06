import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    all_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                all_triples.append((i, j, k))
                
    # Cache the 7 other triples for each potential (x', y', z') completion?
    # No, just compute on the fly.
    
    def is_safe(triple, current_set):
        x, y, z = triple
        # We need to check if there exists x', y', z' such that
        # the 7 other triples are in current_set.
        
        # Optimization: 
        # For the 7 triples to be present, specifically:
        # (x', y, z) must be in set.
        # (x, y', z) must be in set.
        # (x, y, z') must be in set.
        
        # Let's iterate x'
        # For a fixed x', we need (x', y, z) in set.
        # If not, skip this x'.
        
        candidates_x = []
        for xp in range(n):
            if xp != x and (xp, y, z) in current_set:
                candidates_x.append(xp)
                
        if not candidates_x:
            return True # No conflict possible via x'
            
        # Now for each xp in candidates_x, we need to find yp, zp
        # such that:
        # (x, yp, z) in set
        # (x, y, zp) in set
        # (xp, yp, z) in set
        # (xp, y, zp) in set
        # (x, yp, zp) in set
        # (xp, yp, zp) in set
        
        # And yp != y, x, xp; zp != z, x, y, xp, yp.
        
        for xp in candidates_x:
            # Find candidate y's
            # Must have (x, yp, z) in set AND (xp, yp, z) in set
            candidates_y = []
            for yp in range(n):
                if yp == y or yp == x or yp == xp:
                    continue
                if (x, yp, z) in current_set and (xp, yp, z) in current_set:
                    candidates_y.append(yp)
                    
            if not candidates_y:
                continue
                
            for yp in candidates_y:
                # Find candidate z's
                # Must have (x, y, zp) in set AND (xp, y, zp) in set AND (x, yp, zp) in set AND (xp, yp, zp) in set
                # And zp != z, x, y, xp, yp
                for zp in range(n):
                    if zp == z or zp == x or zp == y or zp == xp or zp == yp:
                        continue
                    if ((x, y, zp) in current_set and
                        (xp, y, zp) in current_set and
                        (x, yp, zp) in current_set and
                        (xp, yp, zp) in current_set):
                        return False # Conflict found
                    
        return True

    best_count = 0
    best_set = set()
    
    # Run multiple greedy restarts
    # 150s limit. 
    # Each restart: 560 triples. 
    # is_safe is fast.
    
    max_restarts = 5000
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle(all_triples)
        current_set = set()
        
        for t in all_triples:
            if is_safe(t, current_set):
                current_set.add(t)
                
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = current_set
            
    return [list(t) for t in best_set]
