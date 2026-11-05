# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 4096
    
    def is_valid_addition(x, selected, sums):
        """
        Check if x can be added to selected with existing sums.
        Returns (True, new_sums) if valid, (False, None) if not.
        """
        new_sums = set()
        
        # (x, x, x)
        s = 3 * x
        if s in sums:
            return False, None
        new_sums.add(s)
        
        # (x, x, y) and (x, y, y)
        for y in selected:
            s = 2 * x + y
            if s in sums or s in new_sums:
                return False, None
            new_sums.add(s)
            
            s = x + 2 * y
            if s in sums or s in new_sums:
                return False, None
            new_sums.add(s)
        
        # (x, y, z) for y < z in selected
        for i in range(len(selected)):
            y = selected[i]
            for j in range(i + 1, len(selected)):
                z = selected[j]
                s = x + y + z
                if s in sums or s in new_sums:
                    return False, None
                new_sums.add(s)
        
        return True, new_sums

    def greedy_build(order):
        selected = []
        sums = set()
        for x in order:
            valid, new_sums = is_valid_addition(x, selected, sums)
            if valid:
                selected.append(x)
                sums.update(new_sums)
        return selected, sums

    def rebuild_sums(selected):
        """Rebuild the sums set from scratch for a given set."""
        sums = set()
        k = len(selected)
        for i in range(k):
            a = selected[i]
            for j in range(i, k):
                b = selected[j]
                for l in range(j, k):
                    c = selected[l]
                    sums.add(a + b + c)
        return sums

    best_set = []
    best_sums = set()
    best_size = 0
    
    # 1. Initial Greedy: Standard (smallest first)
    standard_order = list(range(1, n + 1))
    s, sums = greedy_build(standard_order)
    best_set = s
    best_sums = sums
    best_size = len(s)
    
    # 2. Randomized Greedy Restarts
    np.random.seed(42)
    max_restarts = 300
    for i in range(max_restarts):
        if time.time() > deadline - 25:  # Leave time for local search
            break
        order = np.random.permutation(n) + 1
        s, sums = greedy_build(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            best_sums = sums
    
    # 3. Local Search: "remove 1, add 2" moves to increase size
    if best_size < 40:
        current_set = list(best_set)
        current_sums = set(best_sums)
        current_size = best_size
        
        improvement_found = True
        while improvement_found and current_size < 40:
            improvement_found = False
            
            # Try many random removals
            for attempt in range(800):
                if time.time() > deadline - 5:
                    break
                if current_size >= 40:
                    break
                
                # Pick a random element to remove
                idx = np.random.randint(len(current_set))
                y_to_remove = current_set[idx]
                
                # Remove y_to_remove
                reduced_set = [x for x in current_set if x != y_to_remove]
                reduced_sums = rebuild_sums(reduced_set)
                
                # Try to add 2 elements
                # First, generate a diverse set of candidates
                # Include: random candidates, and candidates near existing elements
                candidates = set()
                
                # Random candidates
                for _ in range(30):
                    c = np.random.randint(1, n + 1)
                    candidates.add(c)
                
                # Candidates near existing elements (gaps)
                sorted_reduced = sorted(reduced_set)
                for x in sorted_reduced:
                    for delta in [-3, -2, -1, 1, 2, 3]:
                        c = x + delta
                        if 1 <= c <= n and c not in reduced_set:
                            candidates.add(c)
                
                candidates = list(candidates)
                np.random.shuffle(candidates)
                
                found = False
                for x1 in candidates[:40]:  # Limit to 40 candidates for x1
                    if x1 in reduced_set:
                        continue
                    valid1, sums1 = is_valid_addition(x1, reduced_set, reduced_sums)
                    if not valid1:
                        continue
                    
                    temp_set = reduced_set + [x1]
                    temp_sums = reduced_sums.union(sums1)
                    
                    # Now try to find x2
                    # Generate candidates for x2
                    candidates2 = set()
                    for _ in range(30):
                        c = np.random.randint(1, n + 1)
                        candidates2.add(c)
                    
                    sorted_temp = sorted(temp_set)
                    for x in sorted_temp:
                        for delta in [-3, -2, -1, 1, 2, 3]:
                            c = x + delta
                            if 1 <= c <= n and c not in temp_set and c != x1:
                                candidates2.add(c)
                    
                    candidates2 = list(candidates2)
                    np.random.shuffle(candidates2)
                    
                    for x2 in candidates2[:40]:  # Limit to 40 candidates for x2
                        if x2 in temp_set:
                            continue
                        valid2, sums2 = is_valid_addition(x2, temp_set, temp_sums)
                        if valid2:
                            # Success!
                            new_set = temp_set + [x2]
                            new_sums = temp_sums.union(sums2)
                            
                            current_set = new_set
                            current_sums = new_sums
                            current_size += 1
                            improvement_found = True
                            found = True
                            break
                    if found:
                        break
                
                if found:
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set
                        best_sums = current_sums
                    break  # Restart the while loop with the new set
        
        # Final check
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
            best_sums = current_sums
    
    return best_set

# EVOLVE-BLOCK-END
