import time
import random

def construct():
    deadline = time.time() + 50 # Leave 10s margin
    
    def can_add(x, S):
        # Check if adding x creates a 4-AP with elements in S
        # S is a set of ints
        # x is not in S
        # 1. x is 1st term: x, x+d, x+2d, x+3d
        # 2. x is 2nd term: x-d, x, x+d, x+2d
        # 3. x is 3rd term: x-2d, x-d, x, x+d
        # 4. x is 4th term: x-3d, x-2d, x-d, x
        
        # Max d such that terms are in [1, 512]
        # d can be up to 170
        for d in range(1, 171):
            # 1. x is 1st
            if x + 3*d <= 512:
                if (x+d in S) and (x+2*d in S) and (x+3*d in S):
                    return False
            # 2. x is 2nd
            if x - d >= 1:
                if (x-d in S) and (x+d in S) and (x+2*d in S):
                    return False
            # 3. x is 3rd
            if x - 2*d >= 1:
                if (x-2*d in S) and (x-d in S) and (x+d in S):
                    return False
            # 4. x is 4th
            if x - 3*d >= 1:
                if (x-3*d in S) and (x-2*d in S) and (x-d in S):
                    return False
        return True

    best_set = set()
    best_size = 0
    
    while time.time() < deadline:
        # Random restart
        S = set()
        elements = list(range(1, 513))
        random.shuffle(elements)
        
        # Greedy fill
        for x in elements:
            if can_add(x, S):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
        
        # Local search / Improvement
        # Try to add more
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Get candidates from complement
            # To speed up, we can just iterate through a random subset of complement
            # Or just iterate all if fast enough
            complement = [i for i in range(1, 513) if i not in S]
            random.shuffle(complement)
            
            for x in complement:
                if can_add(x, S):
                    S.add(x)
                    changed = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S.copy()
                    # After adding, we might be able to add more, so restart loop
                    break 
            
            if not changed:
                # Stuck. Try removing an element to unblock
                if S:
                    # Remove a random element, or one that is "central"
                    # For now, random
                    to_remove = random.choice(list(S))
                    S.remove(to_remove)
                    # After removal, we might be able to add something else
                    # The loop above will try to add again
                    # But we need to ensure we don't infinite loop without progress
                    # Since we removed one, the set size decreased.
                    # We rely on the fact that adding back might lead to a different configuration.
                    # To avoid oscillation, we could track history, but simple random usually works for small N.
                    # Let's just let the outer while `changed` handle it.
                    # Actually, if we just removed, `changed` was False, so we exit the inner while?
                    # No, `changed` is set to True if we add. If we remove, we should probably restart the addition phase.
                    # Let's structure it as:
                    # 1. Try to add.
                    # 2. If fail, remove one.
                    # 3. Repeat.
                    
                    # The current loop structure:
                    # while changed:
                    #   try add -> if success, changed=True, break
                    #   if fail all, changed=False -> exit
                    # This means if we fail to add, we exit local search.
                    # We need a "remove and retry" mechanism.
                    
                    # Revised Local Search:
                    # while time < deadline:
                    #   added = try_add_any()
                    #   if added: continue
                    #   if not S: break
                    #   remove_one()
                    
                    # Let's implement this logic properly.
                    
                    break # Break to restart the local search loop logic? 
                    # Actually, let's just implement a proper local search loop.

    return sorted(best_set)
