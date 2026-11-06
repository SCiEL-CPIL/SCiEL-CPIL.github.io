import random
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    best_set = []
    best_size = 0
    
    max_n = 1000
    
    def run_greedy(order):
        chosen = []
        counts = {}
        
        for c in order:
            valid = True
            
            # Check 2*c
            if counts.get(2 * c, 0) >= 2:
                valid = False
            
            if valid:
                # Check c + x for x in chosen
                for x in chosen:
                    s = c + x
                    if counts.get(s, 0) >= 2:
                        valid = False
                        break
            
            if valid:
                # Update counts
                counts[2 * c] = counts.get(2 * c, 0) + 1
                chosen.append(c)
                for x in chosen[:-1]:
                    s = c + x
                    counts[s] = counts.get(s, 0) + 1
                
        return chosen, counts

    def local_search(initial_set, initial_counts, max_iterations=2000):
        S = set(initial_set)
        counts = dict(initial_counts)
        C = set(range(1, max_n + 1)) - S
        
        # Initialize conflict_count
        # conflict_count[y] = number of sums s such that y is part of a pair (a,b) with a+b=s and counts[s]==2
        conflict_count = {y: 0 for y in S}
        
        # Pre-compute pairs for sums with count 2 to initialize conflict_count
        # We need to find pairs (a,b) in S such that a+b=s and counts[s]==2
        # Iterate s from 2 to 2000
        for s in range(2, 2 * max_n + 1):
            if counts.get(s, 0) == 2:
                # Find pairs
                found_pairs = []
                for a in S:
                    b = s - a
                    if b in S and a <= b:
                        found_pairs.append((a, b))
                # There should be exactly 2 pairs
                for a, b in found_pairs:
                    conflict_count[a] += 1
                    if a != b:
                        conflict_count[b] += 1
        
        def can_add(x):
            if counts.get(2 * x, 0) >= 2:
                return False
            for y in S:
                if counts.get(x + y, 0) >= 2:
                    return False
            return True

        def add(x):
            S.add(x)
            C.remove(x)
            # Update conflict_count for x
            conflict_count[x] = 0
            # Update counts
            s_2x = 2 * x
            if counts.get(s_2x, 0) == 1:
                # Becomes 2, so x is in a conflict
                conflict_count[x] += 1
            counts[s_2x] = counts.get(s_2x, 0) + 1
            
            for y in S:
                if y != x:
                    s = x + y
                    if counts.get(s, 0) == 1:
                        # Becomes 2, so x and y are in a conflict
                        conflict_count[x] += 1
                        conflict_count[y] += 1
                    counts[s] = counts.get(s, 0) + 1

        def remove(x):
            S.remove(x)
            C.add(x)
            
            # Update conflict_count for x and others
            # x is removed, so its conflict_count is no longer relevant, but we need to update others
            # For each y in S, consider sum s = x + y
            for y in S:
                s = x + y
                if counts.get(s, 0) == 2:
                    # It was a conflict. Now it becomes 1.
                    # So y loses a conflict
                    conflict_count[y] -= 1
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
            
            s_2x = 2 * x
            if counts.get(s_2x, 0) == 2:
                 # It was a conflict. Now it becomes 1.
                 # x was in a conflict with itself? 
                 # conflict_count[x] included this. But x is removed.
                 # Do we need to update anyone else? No.
                 pass
            counts[s_2x] -= 1
            if counts[s_2x] == 0:
                del counts[s_2x]
                
            # Remove x from conflict_count dict? Or just leave it.
            # We don't use conflict_count for elements in C.
            # But if we add x back, we recalculate.
            # So we can just leave it or delete it.
            # Let's delete to keep it clean.
            del conflict_count[x]

        best_local = len(S)
        best_local_set = set(S)
        
        for _ in range(max_iterations):
            if time.time() - start_time > deadline:
                break
                
            if not C:
                break
            
            # Try to add a random element from C
            added = False
            candidates = random.sample(list(C), min(10, len(C)))
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(x):
                    add(x)
                    added = True
                    if len(S) > best_local:
                        best_local = len(S)
                        best_local_set = set(S)
                    break
            
            if not added:
                # Stuck. Try to remove an element from S that is involved in the most conflicts
                if S:
                    # Find y with max conflict_count
                    # If all are 0, pick random
                    max_conf = -1
                    best_y = None
                    # Random sample to avoid O(N) scan? N is small (~400). Scan is fine.
                    for y in S:
                        c_val = conflict_count.get(y, 0)
                        if c_val > max_conf:
                            max_conf = c_val
                            best_y = y
                    if best_y is None:
                        best_y = random.choice(list(S))
                    remove(best_y)
                    
        return list(best_local_set)

    # 1. Deterministic Ascending
    asc_order = list(range(1, max_n + 1))
    current_set, current_counts = run_greedy(asc_order)
    best_set = current_set
    best_size = len(current_set)
    
    # Apply local search to the baseline
    if time.time() - start_time < deadline:
        improved = local_search(best_set, current_counts)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved

    # 2. Random Restarts with Local Search
    while time.time() - start_time < deadline:
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        current_set, current_counts = run_greedy(order)
        
        # If greedy result is already better, update
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # Try to improve with local search
        if time.time() - start_time < deadline - 2.0:
             improved = local_search(current_set, current_counts)
             if len(improved) > best_size:
                 best_size = len(improved)
                 best_set = improved

    return best_set
