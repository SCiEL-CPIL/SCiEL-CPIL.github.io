import time
import random
from math import lcm

def construct():
    deadline = time.time() + 85
    
    # LCM of 1..22
    D = 232792560
    vals = {x: D // x for x in range(1, 23)}
    all_elements = list(range(1, 23))
    
    def get_sums(S):
        """Return a list of all subset sums of S."""
        sums = [0]
        for x in S:
            v = vals[x]
            # Add v to all existing sums
            # We need to be careful not to modify the list while iterating
            # Create new list of sums
            new_sums = []
            for s in sums:
                new_sums.append(s + v)
            sums.extend(new_sums)
        return sums

    def is_valid(S):
        """Check if all subset sums of S are distinct."""
        sums = get_sums(S)
        return len(sums) == len(set(sums))

    def find_collision(S):
        """Find two distinct subsets A, B with same sum. Returns (A, B) or None."""
        sum_to_sub = {}
        # Generate subsets as bitmasks or lists?
        # For small S, list of elements is fine.
        # We iterate 2^k subsets.
        k = len(S)
        # Use itertools.combinations for each size?
        # Or just bit manipulation.
        # Let's use bit manipulation for speed.
        # Map index to element.
        elem_list = list(S)
        for mask in range(1, 1 << k):
            s = 0
            subset = []
            for i in range(k):
                if mask & (1 << i):
                    s += vals[elem_list[i]]
                    subset.append(elem_list[i])
            if s in sum_to_sub:
                return (sum_to_sub[s], subset)
            else:
                sum_to_sub[s] = subset
        return None

    best_set = []
    best_size = 0
    
    # Greedy Removal Strategy
    # Try multiple restarts with different heuristics
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        S = set(all_elements)
        
        # Heuristic: 0 = remove smallest value (largest x), 1 = remove largest value (smallest x), 2 = random
        heuristic = restart % 3
        
        while time.time() < deadline:
            if is_valid(S):
                # Try to grow
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    # Try adding elements not in S
                    # Order: try adding smaller x first (larger values) or larger x?
                    # Let's try all candidates
                    for y in all_elements:
                        if y not in S:
                            new_S = S | {y}
                            if is_valid(new_S):
                                S = new_S
                                changed = True
                                break # Restart growth loop
                    
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = list(S)
                
                # If we reached size 22, we are done
                if len(S) == 22:
                    return list(S)
                
                # If we can't grow, we are stuck in a local optimum.
                # To escape, we must remove an element.
                # But removing reduces size.
                # We only do this if we want to try to find a *different* local optimum
                # that might be larger.
                # However, if we just grew to a local max, removing is counter-productive
                # unless we expect to add back MORE elements.
                
                # Let's try a "kick": remove a random element, then regrow.
                # This is simulated annealing style.
                if len(S) < 22:
                    # Remove a random element to escape local optimum
                    # But only if we haven't found a very large set yet?
                    # Or always?
                    # If we remove, we might end up with a smaller set.
                    # We keep best_set, so it's safe.
                    if random.random() < 0.5: # 50% chance to kick
                        to_remove = random.choice(list(S))
                        S.discard(to_remove)
                        # Regrow
                        changed = True
                        while changed and time.time() < deadline:
                            changed = False
                            for y in all_elements:
                                if y not in S:
                                    new_S = S | {y}
                                    if is_valid(new_S):
                                        S = new_S
                                        changed = True
                                        break
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = list(S)
                        # Continue loop, might kick again
                    else:
                        # Stay in this local optimum? No, we should break to next restart
                        # or try a different kick.
                        # Let's break to next restart to try a different starting point
                        # or heuristic.
                        break
            else:
                # Invalid, find collision
                collision = find_collision(S)
                if collision is None:
                    # Should not happen if is_valid is False
                    break
                A, B = collision
                C = (set(A) ^ set(B))
                if not C:
                    # A == B? Should not happen for distinct subsets
                    break
                
                # Choose element to remove
                if heuristic == 0:
                    # Remove smallest value (largest x)
                    x_to_remove = max(C, key=lambda x: x) # Max x means min val
                elif heuristic == 1:
                    # Remove largest value (smallest x)
                    x_to_remove = min(C, key=lambda x: x)
                else:
                    # Random
                    x_to_remove = random.choice(list(C))
                
                S.discard(x_to_remove)
        
    # Final check: if best_set is empty, fallback
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
