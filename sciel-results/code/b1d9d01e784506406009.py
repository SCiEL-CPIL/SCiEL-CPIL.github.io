import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    N = 1000
    MAX_SUM = 2 * N  # 2000

    def build_counts_numpy(chosen):
        """Build sum counts as a numpy array for speed."""
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        arr = np.array(sorted(chosen), dtype=np.int32)
        n = len(arr)
        for i in range(n):
            a = arr[i]
            for j in range(i, n):
                b = arr[j]
                counts[a + b] += 1
        return counts

    def is_valid_add_numpy(counts, chosen_set, x):
        """Check if x can be added using numpy array counts."""
        # Check all sums x + y for y in chosen_set
        for y in chosen_set:
            s = x + y
            if counts[s] >= 2:
                return False
        if counts[2 * x] >= 2:
            return False
        return True

    def add_element_numpy(counts, chosen_set, x):
        """Add x to chosen_set and update counts."""
        for y in chosen_set:
            counts[x + y] += 1
        counts[2 * x] += 1
        chosen_set.add(x)

    def remove_element_numpy(counts, chosen_set, x):
        """Remove x from chosen_set and update counts."""
        for y in chosen_set:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen_set.discard(x)

    # Phase 1: Multiple greedy passes with numpy
    best_chosen = set()
    best_size = 0
    
    # Forward greedy
    chosen_set = set()
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    for x in range(1, N + 1):
        if is_valid_add_numpy(counts, chosen_set, x):
            add_element_numpy(counts, chosen_set, x)
    best_chosen = chosen_set.copy()
    best_size = len(chosen_set)
    
    # Backward greedy
    if time.time() < deadline - 10:
        chosen_set = set()
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for x in range(N, 0, -1):
            if is_valid_add_numpy(counts, chosen_set, x):
                add_element_numpy(counts, chosen_set, x)
        if len(chosen_set) > best_size:
            best_chosen = chosen_set.copy()
            best_size = len(chosen_set)
    
    # Random greedy passes - increase count since numpy is faster
    random.seed(99999)
    for _ in range(300):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        chosen_set = set()
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for x in order:
            if is_valid_add_numpy(counts, chosen_set, x):
                add_element_numpy(counts, chosen_set, x)
        if len(chosen_set) > best_size:
            best_chosen = chosen_set.copy()
            best_size = len(chosen_set)
    
    # Phase 2: Try to add more elements to the best solution
    chosen_set = best_chosen.copy()
    counts = build_counts_numpy(list(chosen_set))
    
    # Try to add any element that can be added
    added = True
    while added and time.time() < deadline - 5:
        added = False
        for x in range(1, N + 1):
            if x not in chosen_set and is_valid_add_numpy(counts, chosen_set, x):
                add_element_numpy(counts, chosen_set, x)
                added = True
                break
    
    best_size = len(chosen_set)
    best_chosen = chosen_set.copy()
    
    # Phase 3: Enhanced local search - try to improve by swapping
    if time.time() < deadline - 2:
        counts = build_counts_numpy(list(best_chosen))
        best_size = len(best_chosen)
        
        # More iterations with better candidate selection
        for iteration in range(500):
            if time.time() > deadline - 2:
                break
            
            if len(best_chosen) == 0:
                break
            
            # Pick a random element to remove
            old_x = random.choice(list(best_chosen))
            
            # Temporarily remove it
            remove_element_numpy(counts, best_chosen, old_x)
            
            # Try to add a different element - prioritize candidates that are "less constraining"
            found = False
            candidates = [x for x in range(1, N + 1) if x != old_x and x not in best_chosen]
            
            # Sort candidates by how many conflicts they would create (fewer = better)
            # Use a heuristic: prefer smaller sums with existing elements
            candidate_scores = []
            for x in candidates:
                conflict_count = 0
                for y in best_chosen:
                    s = x + y
                    if counts[s] >= 2:
                        conflict_count += 1
                if counts[2 * x] >= 2:
                    conflict_count += 1
                if conflict_count == 0:
                    candidate_scores.append((0, x))
                elif conflict_count == 1:
                    candidate_scores.append((1, x))
                else:
                    candidate_scores.append((2, x))
            
            # Try candidates with 0 conflicts first, then 1, then 2
            candidate_scores.sort()
            
            for score, x in candidate_scores[:200]:
                if score > 1:
                    break
                if is_valid_add_numpy(counts, best_chosen, x):
                    add_element_numpy(counts, best_chosen, x)
                    found = True
                    break
            
            if not found:
                # Put old_x back
                add_element_numpy(counts, best_chosen, old_x)
            
            # Update best if improved
            if len(best_chosen) > best_size:
                best_size = len(best_chosen)
    
    return list(best_chosen)
