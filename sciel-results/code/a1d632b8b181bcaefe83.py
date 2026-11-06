import random
import time

N = 256

def construct():
    best_size = N + 1
    best_set = None
    
    # Use a fixed random seed for reproducibility in local tests, 
    # but the task implies we can use randomness. 
    # We will run multiple trials.
    
    # Precompute nothing specific, N is small.
    
    for trial in range(200):
        # Randomize start to explore different local optima
        random.seed(trial)
        
        selected = []
        # sums_bitset: bit i is 1 if sum i is in the set of sums
        # Max sum is 512.
        sums_bitset = 0
        
        # Candidates are implicitly 1..N.
        # We maintain a list of currently valid candidates.
        # Initially all are valid.
        # But checking validity against sums_bitset is O(|selected|).
        # |selected| grows slowly.
        
        # To speed up, we can maintain a list of candidates and filter it.
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        while candidates:
            # Filter valid candidates
            valid = []
            for x in candidates:
                # Check if x can be added
                # Condition: 2x not in sums, and x+s not in sums for all s in selected
                if (sums_bitset >> (2 * x)) & 1:
                    continue
                is_valid = True
                for s in selected:
                    if (sums_bitset >> (x + s)) & 1:
                        is_valid = False
                        break
                if is_valid:
                    valid.append(x)
            
            if not valid:
                break
            
            # Heuristic: Pick x in valid that maximizes the number of other valid candidates it blocks.
            # "Blocking" y means y becomes invalid after adding x.
            # New sums added by x: new_sums = {x+s for s in selected} U {2x}
            # y is blocked if:
            # 1. 2y is in new_sums
            # 2. y+s is in new_sums for some s in selected
            
            best_x = None
            max_blocked = -1
            
            # Optimization: If valid is small, check all. If large, we can still check all because N=256 is small.
            # The cost per x is O(|selected| + |valid| * |selected|).
            # |valid| <= 256, |selected| ~ 10. Cost ~ 256 * 10 * 10 = 25k ops per step.
            # 20 steps * 25k = 500k ops per trial.
            # 200 trials * 500k = 100M ops. Might be tight in Python, but bit ops are fast.
            # Let's try to optimize the blocking check.
            
            # For each x in valid:
            #   new_sums_set = set()
            #   for s in selected: new_sums_set.add(x+s)
            #   new_sums_set.add(2*x)
            #   blocked_count = 0
            #   for y in valid:
            #     if y == x: continue
            #     if 2*y in new_sums_set: blocked_count += 1; continue
            #     # Check if any y+s in new_sums_set
            #     # This is the expensive part.
            #     # Alternative: Precompute for each y, the sums that would block it?
            #     # Blockers(y) = {2y} U {y+s for s in selected}
            #     # y is blocked if new_sums_set intersects Blockers(y).
            
            # Let's use the intersection method with sets for clarity first, then optimize if needed.
            # But sets creation is expensive.
            
            # Better:
            # For a fixed x, new_sums is a set of size |selected|+1.
            # For each y, we check if 2y in new_sums (O(1) if set) or if any y+s in new_sums.
            # Instead of iterating s for each y, we can iterate new_sums and see which y's they block?
            # No, that's harder to map back to y.
            
            # Let's stick to the direct check but use a list for new_sums and check membership?
            # Or convert new_sums to a bitmask?
            # new_sums_bitset = 0
            # for s in selected: new_sums_bitset |= 1 << (x+s)
            # new_sums_bitset |= 1 << (2*x)
            
            # Then for y:
            # if (new_sums_bitset >> (2*y)) & 1: blocked
            # else:
            #   # Check if any y+s is in new_sums_bitset
            #   # This still requires iterating s.
            
            # Let's try a different heuristic which is faster:
            # Pick x that maximizes the number of "collisions" with existing structure?
            # Or simply: Pick x that is "central" or has high degree in the conflict graph?
            
            # Actually, a known good heuristic for minimal maximal Sidon sets is to pick the element that
            # creates the most "wasted" space or conflicts.
            # Let's try: Pick x that minimizes the number of *remaining* valid candidates?
            # That is exactly "max blocked".
            
            # Let's implement "max blocked" with bitsets for the new sums.
            
            for x in valid:
                # Build bitmask for new sums
                ns = 0
                for s in selected:
                    ns |= 1 << (x + s)
                ns |= 1 << (2 * x)
                
                blocked = 0
                for y in valid:
                    if y == x:
                        continue
                    # Check if y is blocked by ns
                    # 1. 2y in ns
                    if (ns >> (2 * y)) & 1:
                        blocked += 1
                        continue
                    # 2. y+s in ns for some s in selected
                    # We can check this by iterating s in selected
                    is_blocked = False
                    for s in selected:
                        if (ns >> (y + s)) & 1:
                            is_blocked = True
                            break
                    if is_blocked:
                        blocked += 1
                
                if blocked > max_blocked:
                    max_blocked = blocked
                    best_x = x
                    if max_blocked == len(valid) - 1:
                        break
            
            # Commit best_x
            selected.append(best_x)
            # Update sums_bitset
            for s in selected[:-1]:
                sums_bitset |= 1 << (best_x + s)
            sums_bitset |= 1 << (2 * best_x)
            
            # Update candidates
            # We can just keep the list and filter in the next iteration's "valid" step.
            # But removing best_x and obviously blocked ones speeds up the next "valid" filter.
            # Let's just remove best_x from candidates. The "valid" filter will handle the rest.
            candidates.remove(best_x)
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected[:]
            # Early stop if we find a very good solution
            if best_size <= 15:
                break

    return best_set
