import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    
    # Build conflict graph
    # conflict[a] is a set of b such that a != b and (a*b) % (a+b) == 0
    conflict = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a].add(b)
                conflict[b].add(a)
                
    # Initial set: All odd numbers
    current_set = set(range(1, N + 1, 2))
    current_size = len(current_set)
    best_set = set(current_set)
    best_size = current_size
    
    # Local search: try to swap or add
    # 1. Try to add any number not in set that has no conflicts in set
    # 2. Try to swap: remove x, add y if y has fewer conflicts with set - {x}
    
    # Precompute degrees for initial set?
    # Just iterate.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding
        for x in range(1, N + 1):
            if x not in current_set:
                if not (conflict[x] & current_set):
                    current_set.add(x)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(current_set)
        
        # Try swapping
        # For each x in set, try to remove it and add y not in set
        # This is expensive. Limit iterations or use random restarts.
        
        # Simpler: Random restarts with greedy
        # But we already have a good start.
        
        # Let's try a few random swaps
        for _ in range(100):
            if time.time() > deadline:
                break
            x = random.choice(list(current_set))
            # Find y not in set that conflicts with x but has fewer total conflicts with set
            # Actually, just try to replace x with y if y is "better"
            # Metric: number of conflicts with current_set
            conflicts_x = len(conflict[x] & current_set) # should be 0 if valid set
            # Wait, current_set is independent, so conflict[x] & current_set is empty for x in current_set?
            # No, conflict[x] contains neighbors. Since set is independent, no neighbor is in set.
            # So conflicts_x is 0.
            
            # We want to add y. y must not conflict with current_set - {x}.
            # And we remove x.
            # Gain: 1 (add y) - 1 (remove x) = 0 size change.
            # But might open up more additions.
            
            # Let's just try to add nodes that have 0 conflicts.
            # If none, try to remove one node to allow adding two others?
            
            pass
            
    # Final polish: try to add any remaining isolated nodes
    for x in range(1, N + 1):
        if x not in current_set:
            if not (conflict[x] & current_set):
                current_set.add(x)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                
    if current_size > best_size:
        best_set = current_set
        
    return list(best_set)
