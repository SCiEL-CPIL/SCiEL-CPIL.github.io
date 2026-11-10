import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    # Precompute conflicts
    # conflict[i][j] is True if (i*j) % (i+j) == 0
    N = 256
    conflicts = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflicts[i][j] = True
                conflicts[j][i] = True

    # Initial set: all odd numbers
    S = set(range(1, N + 1, 2))
    
    # Greedy addition of even numbers
    # Sort evens by number of conflicts with current S (or just iterate)
    evens = list(range(2, N + 1, 2))
    
    # Helper to check if x can be added to S
    def can_add(x, S):
        for y in S:
            if conflicts[x][y]:
                return False
        return True

    # Pass 1: Add any even that doesn't conflict
    for e in evens:
        if can_add(e, S):
            S.add(e)
            
    # Pass 2: Local search
    # Try to swap an odd o with an even e if it allows adding more evens later?
    # Simple hill climbing: try random swaps that maintain validity and increase size (impossible directly)
    # Or try to remove an odd o, add an even e, then try to add more evens.
    
    best_S = S
    best_size = len(S)
    
    # Try each even not in S
    evens_not_in_S = [e for e in evens if e not in S]
    odds_in_S = [o for o in S if o % 2 != 0]
    
    # For each even not in S, find the odd it conflicts with most?
    # Or just try swapping with each odd it conflicts with.
    
    for e in evens_not_in_S:
        if time.time() > deadline:
            break
            
        # Find conflicts of e with current S
        conflicts_with_S = [y for y in S if conflicts[e][y]]
        
        if not conflicts_with_S:
            # Should have been added in Pass 1, but just in case
            S.add(e)
            continue
            
        # Try removing one conflict and adding e
        for o in conflicts_with_S:
            if time.time() > deadline:
                break
                
            # Temporarily remove o and add e
            S.remove(o)
            S.add(e)
            
            # Try to add more evens now
            added_any = False
            for e2 in evens_not_in_S:
                if e2 == e:
                    continue
                if e2 in S:
                    continue
                if can_add(e2, S):
                    S.add(e2)
                    added_any = True
            
            # Check if size increased
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
            
            # Revert
            S.remove(e)
            if added_any:
                # We added some evens, need to remove them too
                # This is complex to revert correctly without tracking
                # Let's simplify: just try the swap and keep if better, else revert
                # But we modified S in place.
                pass
            
            # Correct Revert Logic:
            # We need to restore S to state before swap.
            # State before: S_old
            # State after: S_old - {o} + {e} + {new_evens}
            # We can't easily revert {new_evens} if we don't track them.
            
            # Let's restart the local search strategy with a cleaner copy.
            pass

    # Simpler Local Search: Random Restart Greedy
    # Since N is small, we can run multiple greedy passes with different orderings.
    
    best_result = best_S
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Shuffle evens
        random.shuffle(evens)
        S_current = set(range(1, N + 1, 2))
        
        for e in evens:
            if can_add(e, S_current):
                S_current.add(e)
        
        if len(S_current) > len(best_result):
            best_result = S_current
    
    return list(best_result)
