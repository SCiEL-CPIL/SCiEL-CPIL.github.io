import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    N = 1000
    
    def build_counts(chosen):
        """Build sum counts for a given set."""
        counts = {}
        for i in range(len(chosen)):
            a = chosen[i]
            for j in range(i, len(chosen)):
                b = chosen[j]
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts
    
    def is_valid_add(chosen_set, counts, x):
        """Check if x can be added to chosen."""
        for y in chosen_set:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        if counts.get(2 * x, 0) >= 2:
            return False
        return True
    
    def greedy(order, chosen_set, counts):
        """Greedy construction given an order."""
        for x in order:
            if x not in chosen_set and is_valid_add(chosen_set, counts, x):
                for y in chosen_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen_set.add(x)
        return chosen_set, counts
    
    # Phase 1: Multiple greedy passes
    best_chosen = set()
    best_size = 0
    
    # Forward greedy
    chosen_set = set()
    counts = {}
    for x in range(1, N + 1):
        if is_valid_add(chosen_set, counts, x):
            for y in chosen_set:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            counts[2 * x] = counts.get(2 * x, 0) + 1
            chosen_set.add(x)
    best_chosen = chosen_set.copy()
    best_size = len(chosen_set)
    
    # Backward greedy
    if time.time() < deadline - 10:
        chosen_set = set()
        counts = {}
        for x in range(N, 0, -1):
            if is_valid_add(chosen_set, counts, x):
                for y in chosen_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen_set.add(x)
        if len(chosen_set) > best_size:
            best_chosen = chosen_set.copy()
            best_size = len(chosen_set)
    
    # Random greedy passes
    random.seed(99999)
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        chosen_set = set()
        counts = {}
        for x in order:
            if is_valid_add(chosen_set, counts, x):
                for y in chosen_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen_set.add(x)
        if len(chosen_set) > best_size:
            best_chosen = chosen_set.copy()
            best_size = len(chosen_set)
    
    # Phase 2: Try to add more elements to the best solution
    chosen_set = best_chosen.copy()
    counts = build_counts(list(chosen_set))
    
    # Try to add any element that can be added
    added = True
    while added and time.time() < deadline - 3:
        added = False
        for x in range(1, N + 1):
            if x not in chosen_set and is_valid_add(chosen_set, counts, x):
                for y in chosen_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen_set.add(x)
                added = True
                break
    
    best_size = len(chosen_set)
    best_chosen = chosen_set.copy()
    
    # Phase 3: Local search - try to improve by swapping
    if time.time() < deadline - 1:
        counts = build_counts(list(best_chosen))
        
        for _ in range(200):
            if time.time() > deadline - 1:
                break
            
            if len(best_chosen) == 0:
                break
            
            # Pick a random element to remove
            old_x = random.choice(list(best_chosen))
            
            # Temporarily remove it
            for y in best_chosen:
                if y == old_x:
                    continue
                s = old_x + y
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
            counts[2 * old_x] -= 1
            if counts[2 * old_x] == 0:
                del counts[2 * old_x]
            best_chosen.discard(old_x)
            
            # Try to add a different element
            found = False
            candidates = [x for x in range(1, N + 1) if x != old_x and x not in best_chosen]
            random.shuffle(candidates)
            
            for x in candidates[:100]:
                if is_valid_add(best_chosen, counts, x):
                    for y in best_chosen:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    best_chosen.add(x)
                    found = True
                    break
            
            if not found:
                # Put old_x back
                for y in best_chosen:
                    s = old_x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * old_x] = counts.get(2 * old_x, 0) + 1
                best_chosen.add(old_x)
            
            # Update best if improved
            if len(best_chosen) > best_size:
                best_size = len(best_chosen)
    
    return list(best_chosen)
