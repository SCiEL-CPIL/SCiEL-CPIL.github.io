import time
import random

def _check_add(chosen, x, counts):
    # Check if adding x violates the B2[2] condition
    # New sums: x + y for y in chosen, and 2*x
    # We need to ensure no sum exceeds 2 representations
    
    pass

def _try_greedy(order):
    chosen = []
    counts = {}
    
    for x in order:
        # Generate new sums
        new_sums = []
        for y in chosen:
            new_sums.append(x + y)
        new_sums.append(2 * x)
        
        # Check validity
        valid = True
        for s in new_sums:
            if counts.get(s, 0) == 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1

    return chosen

def _local_search(chosen, counts, time_limit, rng):
    """
    Local search: try to improve the set by removing elements and adding others.
    Returns improved (chosen, counts) if found within time limit.
    """
    chosen_set = set(chosen)
    all_numbers = set(range(1, 1001))
    
    # Try swap operations: remove one element, try to add one or more others
    # We'll do a simple hill climbing: try single swaps first
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Shuffle to explore different orders
        candidates_to_remove = list(chosen_set)
        rng.shuffle(candidates_to_remove)
        
        for remove_x in candidates_to_remove:
            if time.time() >= time_limit:
                break
            
            # Remove x from chosen
            new_chosen = [y for y in chosen if y != remove_x]
            new_chosen_set = set(new_chosen)
            
            # Recalculate counts for new_chosen
            new_counts = {}
            for i, a in enumerate(new_chosen):
                for b in new_chosen[i:]:
                    s = a + b
                    new_counts[s] = new_counts.get(s, 0) + 1
            
            # Try to add elements not in new_chosen_set
            # Prioritize elements that were previously rejected or random ones
            available = [x for x in range(1, 1001) if x not in new_chosen_set]
            rng.shuffle(available)
            
            added_count = 0
            for add_x in available:
                # Check if we can add add_x
                new_sums = []
                for y in new_chosen:
                    new_sums.append(add_x + y)
                new_sums.append(2 * add_x)
                
                valid = True
                for s in new_sums:
                    if new_counts.get(s, 0) == 2:
                        valid = False
                        break
                
                if valid:
                    new_chosen.append(add_x)
                    new_chosen_set.add(add_x)
                    for s in new_sums:
                        new_counts[s] = new_counts.get(s, 0) + 1
                    added_count += 1
                
                if len(new_chosen) >= len(chosen):
                    # We've already matched or exceeded the original size
                    # Try to add more if possible
                    continue
                else:
                    break
            
            # If we added more elements than we removed, we improved
            if len(new_chosen) > len(chosen):
                chosen = new_chosen
                chosen_set = new_chosen_set
                counts = new_counts
                improved = True
                break
    
    return chosen, counts

def construct():
    start_time = time.time()
    limit_time = 115.0  # Safety margin
    
    best_set = []
    best_counts = {}
    
    # 1. Deterministic Greedy (Ascending)
    s1 = _try_greedy(range(1, 1001))
    if len(s1) > len(best_set):
        best_set = s1
        # Recalculate counts
        best_counts = {}
        for i, a in enumerate(s1):
            for b in s1[i:]:
                s = a + b
                best_counts[s] = best_counts.get(s, 0) + 1
    
    # 2. Deterministic Greedy (Descending)
    s2 = _try_greedy(range(1000, 0, -1))
    if len(s2) > len(best_set):
        best_set = s2
        best_counts = {}
        for i, a in enumerate(s2):
            for b in s2[i:]:
                s = a + b
                best_counts[s] = best_counts.get(s, 0) + 1
    
    # 3. Randomized Greedy with multiple seeds
    rng = random.Random(42)
    candidates = list(range(1, 1001))
    
    # Run randomized restarts until time limit
    while time.time() - start_time < limit_time - 30:  # Leave 30s for local search
        rng.shuffle(candidates)
        s = _try_greedy(candidates)
        if len(s) > len(best_set):
            best_set = s
            best_counts = {}
            for i, a in enumerate(s):
                for b in s[i:]:
                    s_sum = a + b
                    best_counts[s_sum] = best_counts.get(s_sum, 0) + 1
    
    # 4. Local search to improve the best solution
    if time.time() - start_time < limit_time - 5:
        best_set, best_counts = _local_search(best_set, best_counts, limit_time - 5, rng)
    
    return best_set
