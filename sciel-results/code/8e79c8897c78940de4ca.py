import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s/90s limits
    
    N = 256
    # Initial set: all odd numbers
    S = set(range(1, N + 1, 2))
    
    # Precompute conflicts for all pairs? No, that's 256^2 which is fine, but let's do it on the fly or optimize.
    # Actually, let's just use a simple check function.
    # (a*b) % (a+b) != 0
    
    def can_add(x, s_set):
        for y in s_set:
            if (x * y) % (x + y) == 0:
                return False
        return True

    # 1. Greedy Add Evens
    # Try adding even numbers in a specific order.
    # Randomized restarts with different orders might help.
    # But first, let's try a deterministic greedy pass with a few heuristics.
    
    # Heuristic: Add evens that have the fewest conflicts with current S?
    # Or just try all evens in random order multiple times?
    
    best_set = set(S)
    best_size = len(best_set)
    
    # Let's do a few random restarts with greedy addition of evens
    # And then a local search (swap) phase.
    
    for restart in range(50):  # Up to 50 restarts
        if time.time() > deadline - 5:
            break
            
        S = set(range(1, N + 1, 2))
        
        # Try adding evens in random order
        evens = list(range(2, N + 1, 2))
        random.shuffle(evens)
        
        for e in evens:
            if can_add(e, S):
                S.add(e)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        
        # Local Search: Try to swap elements to increase size?
        # Or just try to add more? No, if greedy failed to add, it's because of conflicts.
        # Can we remove an odd number to allow adding two evens?
        # This is a set cover / maximum independent set type problem on the conflict graph.
        # Nodes: 1..256. Edge between a,b if (a*b)%(a+b)==0.
        # We want Max Independent Set.
        
        # Since we start with all odds (which are independent? Let's check).
        # If a,b are odd, a+b is even. a*b is odd. Odd % Even != 0 unless Odd is 0, which it isn't.
        # So all odds are a valid independent set. Size 128.
        
        # Now we add evens.
        # Local search: Try to remove one element from S and see if we can add two more?
        # Or just try to swap one element for another?
        
        # Let's try a simple hill climbing:
        # For each element in S, try removing it.
        # Then try to add any element not in S that is now compatible.
        # If size increases, keep it.
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2:
                break
                
            # Try removing each element
            for x in list(S):
                if time.time() > deadline - 2:
                    break
                    
                S.remove(x)
                # Try to add any element not in S
                added_any = False
                for y in range(1, N + 1):
                    if y not in S and can_add(y, S):
                        S.add(y)
                        added_any = True
                        break # Only add one at a time? Or try to add all possible?
                        # Adding all possible might be better.
                
                # Let's try to add ALL possible elements after removal
                if added_any:
                    # Re-check all candidates
                    for y in range(1, N + 1):
                        if y not in S and can_add(y, S):
                            S.add(y)
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = set(S)
                    improved = True
                else:
                    # Restore x if it didn't help? 
                    # Actually, if len(S) didn't increase, we should probably restore the state 
                    # to continue searching from the previous state, or accept if equal?
                    # If we accept equal, we might wander. Let's restore if not improved.
                    # But wait, if we removed x and added y, and size is same, it's a neutral move.
                    # For now, let's only accept if strictly better to ensure progress?
                    # Or accept neutral to escape local minima?
                    # Let's accept if >= best_size found in this branch? 
                    # Simpler: If current S size > size before removal, keep. Else restore.
                    # We lost track of size before removal.
                    # Let's re-implement this part more carefully.
                    
                    # Actually, the code above modifies S in place.
                    # If len(S) <= best_size, we might want to undo.
                    # But undoing is tricky.
                    # Let's just stick to the best_set found so far and restart if stuck?
                    # The restart loop already handles this.
                    pass

    # Final check and return
    # Ensure best_set is valid
    # (It should be, as we built it with checks)
    
    return list(best_set)
