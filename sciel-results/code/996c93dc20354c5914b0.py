import math
import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Helper function to check compatibility
    def is_compatible(m1, r1, m2, r2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    # Helper to check if a pair is compatible with a list
    def is_compatible_with_list(m, r, sel):
        for m2, r2 in sel:
            if not is_compatible(m, r, m2, r2):
                return False
        return True
        
    # Helper to calculate score
    def calc_score(sel):
        return sum(1.0 / m for m, r in sel)

    # 1. Greedy Initialization
    # Sort moduli by 1/m descending (m ascending)
    selected = []
    for m in range(20, 201):
        # Try residues in random order to break symmetry? 
        # Actually, for a fixed m, all residues have the same weight.
        # We just need to find *any* compatible residue.
        # Randomizing helps escape local optima in the greedy phase if we do restarts,
        # but for a single pass, deterministic is fine. 
        # However, the choice of residue affects future compatibility.
        # Let's try a few random orders for the residue selection for each m?
        # No, let's stick to a simple greedy: pick the first compatible residue.
        # To improve, we can try all residues and pick the one that leaves the most
        # options for future moduli? That's expensive.
        # Let's just pick the first one found.
        
        found = False
        # Randomize residue order for this specific m to add some variation?
        # Let's use a fixed order 0..m-1 for determinism in the base greedy.
        for r in range(m):
            if is_compatible_with_list(m, r, selected):
                selected.append((m, r))
                found = True
                break
        
    best_selected = list(selected)
    best_score = calc_score(best_selected)
    
    # 2. Local Search
    # We will use the current selection as the start.
    current = list(best_selected)
    current_set = set(current)
    current_score = best_score
    
    # Pre-compute list of all candidates for random selection
    # We can just generate random (m, r) on the fly.
    
    # To speed up compatibility checks, we can maintain a list of selected (m, r).
    # N is at most 181. O(N) check is fast.
    
    # Local Search Loop
    # Strategy: 
    # 1. Try to add a random unselected pair.
    # 2. If no addition found after K tries, perform a "kick" (remove or change residue).
    # 3. Repeat until time limit.
    
    attempts_since_kick = 0
    max_attempts_per_kick = 200
    
    while time.time() - start_time < time_limit:
        
        # Phase 1: Try to add
        added = False
        for _ in range(max_attempts_per_kick):
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random modulus from [20, 200]
            m = random.randint(20, 200)
            r = random.randint(0, m - 1)
            
            if (m, r) in current_set:
                continue
                
            if is_compatible_with_list(m, r, current):
                current.append((m, r))
                current_set.add((m, r))
                current_score += 1.0 / m
                added = True
                attempts_since_kick = 0 # Reset counter on success
                
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(current)
                    # best_set = set(current) # Not strictly needed if we use list for checks, but good for consistency if we were using it.
            
            if not added:
                attempts_since_kick += 1
        
        if not added:
            # Phase 2: Kick
            # Option A: Remove a random element
            # Option B: Change residue of a random element
            
            if len(current) > 0:
                if random.random() < 0.3:
                    # Remove
                    idx = random.randint(0, len(current) - 1)
                    removed = current.pop(idx)
                    current_set.remove(removed)
                    current_score -= 1.0 / removed[0]
                else:
                    # Change residue
                    idx = random.randint(0, len(current) - 1)
                    m, r_old = current[idx]
                    r_new = random.randint(0, m - 1)
                    if r_new != r_old:
                        # Check if new residue is compatible with the rest
                        # Temporarily remove old
                        current_set.remove((m, r_old))
                        # Check against current (which still has the old element in list, 
                        # so we must be careful. is_compatible_with_list checks against list.
                        # We should check against list excluding the element at idx.
                        
                        # Efficient check:
                        compatible = True
                        for i, (m2, r2) in enumerate(current):
                            if i == idx:
                                continue
                            if not is_compatible(m, r_new, m2, r2):
                                compatible = False
                                break
                        
                        if compatible:
                            current[idx] = (m, r_new)
                            current_set.add((m, r_new))
                            # Score doesn't change
                        else:
                            # Revert
                            current_set.add((m, r_old))
            else:
                # If empty, just break or restart
                break
                
    return [[m, r] for m, r in best_selected]
