import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    prescribed = [1, 2, 5, 11]
    best_size = 0
    best_set = set(prescribed)
    
    def get_sums(s):
        sums = set()
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sums.add(lst[i] + lst[j])
        return sums

    def is_valid(s):
        sums = get_sums(s)
        n = len(s)
        return len(sums) == n * (n + 1) // 2

    candidates = list(range(12, 285))
    
    # 1. Greedy Initialization
    current_set = set(prescribed)
    current_sums = get_sums(current_set)
    
    for x in candidates:
        if x in current_set:
            continue
        new_sums_part = set()
        conflict = False
        for y in current_set:
            s = x + y
            if s in current_sums:
                conflict = True
                break
            new_sums_part.add(s)
        if 2 * x in current_sums:
            conflict = True
        if not conflict:
            current_set.add(x)
            current_sums |= new_sums_part
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)

    # 2. Local Search with Randomized Steps
    max_attempts = 5000
    attempt = 0
    sample_size_add = 20  # How many random candidates to try for adding
    sample_size_swap = 20 # How many random candidates to try for swapping
    
    while time.time() < deadline and attempt < max_attempts:
        attempt += 1
        
        if attempt % 10 == 0:
            # Random Greedy Restart
            current_set = set(prescribed)
            current_sums = get_sums(current_set)
            rand_candidates = candidates[:]
            random.shuffle(rand_candidates)
            for x in rand_candidates:
                if x in current_set:
                    continue
                new_sums_part = set()
                conflict = False
                for y in current_set:
                    s = x + y
                    if s in current_sums:
                        conflict = True
                        break
                    new_sums_part.add(s)
                if 2 * x in current_sums:
                    conflict = True
                if not conflict:
                    current_set.add(x)
                    current_sums |= new_sums_part
        else:
            # Refine best
            current_set = set(best_set)
            current_sums = get_sums(current_set)

        # Local Search Loop
        local_iterations = 0
        max_local_iterations = 100
        
        while len(current_set) < 20 and local_iterations < max_local_iterations:
            local_iterations += 1
            
            # Try to add an element using random sampling
            added = False
            # Get a random sample of candidates not in current_set
            # To be fast, we just pick random indices from the full candidate list
            # and check if they are in current_set
            for _ in range(sample_size_add):
                x = random.choice(candidates)
                if x in current_set:
                    continue
                new_sums_part = set()
                conflict = False
                for y in current_set:
                    s = x + y
                    if s in current_sums:
                        conflict = True
                        break
                    new_sums_part.add(s)
                if 2 * x in current_sums:
                    conflict = True
                if not conflict:
                    current_set.add(x)
                    current_sums |= new_sums_part
                    added = True
                    break
            
            if added:
                continue
            
            # If we couldn't add, try a swap
            removable = [x for x in current_set if x not in prescribed]
            if not removable:
                break
                
            x_remove = random.choice(removable)
            
            swap_found = False
            for _ in range(sample_size_swap):
                y_add = random.choice(candidates)
                if y_add in current_set or y_add == x_remove:
                    continue
                    
                new_set = set(current_set)
                new_set.remove(x_remove)
                new_set.add(y_add)
                
                if is_valid(new_set):
                    current_set = new_set
                    current_sums = get_sums(current_set)
                    swap_found = True
                    break
            
            if not swap_found:
                break

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            if best_size >= 20:
                break

    return sorted(list(best_set))
