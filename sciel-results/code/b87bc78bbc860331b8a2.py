import time
import random
from itertools import combinations

def has_sidon_4(a_set):
    # Returns a Sidon-4 subset if found, else None
    for comb in combinations(a_set, 4):
        sums = set()
        for i in range(4):
            for j in range(i, 4):
                s = comb[i] + comb[j]
                if s in sums:
                    return None # This comb is NOT Sidon
                sums.add(s)
        return comb # This comb IS Sidon
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    best_set = None
    # We don't really need "best" if we can find a 0.
    # But we track best in case we time out.
    # Since we don't have a good metric for "closeness" other than count,
    # and counting is slow, let's just try to find a valid one.
    
    # If we can't find one, we return the best we found (lowest count) or seed.
    # Let's try to find a valid one.
    
    R = 50 # Range
    
    for _ in range(100): # 100 restarts
        if time.time() > deadline:
            break
            
        s = random.sample(range(R + 1), 12)
        s.sort()
        
        # Try to repair s
        for step in range(1000): # Max 1000 repair steps
            if time.time() > deadline:
                break
                
            violator = has_sidon_4(s)
            if violator is None:
                return s
            
            # Pick an element from the violator to move
            # Indices in s
            idxs = [s.index(x) for x in violator]
            idx = random.choice(idxs)
            old_val = s[idx]
            
            # Try to find a new value that breaks this violator
            # Ideally, it shouldn't create new violators, but that's hard to check fast.
            # Let's just try random moves and check if the specific violator is broken.
            # Then we re-check the whole set.
            
            found_move = False
            for _ in range(50): # 50 attempts to find a good move
                new_val = random.randint(0, R)
                if new_val == old_val or new_val in s:
                    continue
                
                s[idx] = new_val
                s.sort()
                
                # Check if the specific violator is broken
                # Note: violator elements might have changed indices, but values are same except one.
                # Actually, violator is a tuple of values.
                # If we moved one of them, the tuple `violator` is no longer a subset of s?
                # Yes, `violator` contained `old_val`. Now it has `new_val`.
                # So `violator` is not in s.
                # We need to check if ANY Sidon-4 exists.
                
                # This is the expensive part.
                # Let's just check if the specific set of 4 values that formed the violator
                # is now broken.
                # Let new_violator_vals = set(violator) - {old_val} | {new_val}
                # But this new set might not be in s if other elements moved? No, only one moved.
                # So the set of 4 values is now `new_violator_vals`.
                # We can check if `new_violator_vals` is Sidon.
                # If it is NOT Sidon, we might have fixed it.
                # But we might have created a NEW Sidon-4 elsewhere.
                
                # To be safe, we must check `has_sidon_4(s)`.
                # This is slow.
                
                # Optimization:
                # Only check `has_sidon_4(s)` if the specific violator is broken.
                # If the specific violator is still Sidon, revert immediately.
                
                new_violator_set = set(violator)
                new_violator_set.remove(old_val)
                new_violator_set.add(new_val)
                
                # Check if new_violator_set is Sidon
                nv = tuple(sorted(new_violator_set))
                sums = set()
                is_sidon = True
                for i in range(4):
                    for j in range(i, 4):
                        sv = nv[i] + nv[j]
                        if sv in sums:
                            is_sidon = False
                            break
                        sums.add(sv)
                    if not is_sidon:
                        break
                
                if is_sidon:
                    # Still a violator, revert
                    s[idx] = old_val
                    s.sort()
                    continue
                
                # Specific violator is broken.
                # Now check if any OTHER violator exists.
                # This is the full check.
                if has_sidon_4(s) is None:
                    # Success!
                    return s
                else:
                    # We broke one but maybe created another or left others.
                    # Let's accept this move if it reduces the number of violators?
                    # We don't know the count.
                    # Let's just accept it and continue the loop.
                    # It's a random walk.
                    found_move = True
                    break
            
            if not found_move:
                # Revert if we didn't find a move that broke the specific violator
                # (The loop above reverts if is_sidon is True)
                # If we broke the specific violator but has_sidon_4(s) is not None,
                # we accepted the move (s is updated).
                # If we never broke it, s is reverted.
                # If we break out of inner loop with found_move=True, s is updated.
                # If we break out with found_move=False, s is reverted.
                # Wait, if we break out with found_move=True, we don't revert.
                # This allows random walks.
                
                # If we are stuck (no move broke the specific violator), 
                # let's try a different violator or restart.
                # For simplicity, let's just break and restart the local search.
                break
                
    return best_set if best_set else list(range(12))
