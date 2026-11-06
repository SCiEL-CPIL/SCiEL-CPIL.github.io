import time

def construct():
    start_time = time.time()
    deadline = 85.0
    
    S = set(range(91, 181))
    
    def can_add(x, S):
        for y in S:
            if y <= x:
                continue
            if (x * y) % (y - x) != 0:
                continue
            z = (x * y) // (y - x)
            if z in S and z != y:
                return False
        return True

    # Phase 1: Greedy add
    for x in range(90, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)
            
    best_S = set(S)
    best_size = len(S)
    
    # Phase 2: Local Search
    # Try to swap elements to increase size
    # Or just try to add remaining elements by removing blockers
    
    remaining = [x for x in range(1, 181) if x not in S]
    
    # Sort remaining by some heuristic? Maybe smaller first?
    # Or just iterate.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in remaining:
            if time.time() > deadline:
                break
            if x in S:
                continue
            # Check if x can be added
            if can_add(x, S):
                S.add(x)
                remaining.remove(x)
                improved = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
            else:
                # Try to remove a blocker
                # Find all blockers
                blockers = []
                for y in S:
                    if y <= x:
                        continue
                    if (x * y) % (y - x) != 0:
                        continue
                    z = (x * y) // (y - x)
                    if z in S and z != y:
                        blockers.append((y, z))
                
                # Try removing y or z from a blocker
                # To maximize size, we remove 1 and add 1, so size stays same.
                # But maybe it allows adding more later?
                # For now, let's just try to find a state where we can add x.
                # If we remove y, we might be able to add x.
                # We only accept if we can add x AND we don't lose more than we gain?
                # Actually, swapping 1 for 1 keeps size same.
                # We want to INCREASE size.
                # So this local search might not increase size directly.
                # It might help if we can add x and then add another number later?
                # Given time, maybe skip complex swaps.
                
                pass

    return list(best_S)
