import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    
    def compute_sums(arr):
        sums = set()
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                sums.add(arr[i] + arr[j])
        return sums

    def can_add(x, arr, sums):
        # Check if x can be added to arr without creating duplicate sums
        # Sums to check: x + y for all y in arr, and 2*x
        for y in arr:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def add_sums(x, arr, sums):
        new_sums = sums.copy()
        for y in arr:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums

    def remove_sums(x, arr, sums):
        # Remove sums involving x
        new_sums = set()
        for s in sums:
            # If s == x + y for some y in arr, or s == 2x, it might be removed.
            # But other pairs might also sum to s.
            # Safer to recompute, but expensive.
            # For local search, we usually keep a list and recompute sums on swap.
            pass
        # Recomputing is safer and fast enough for size <= 20
        new_arr = [y for y in arr if y != x]
        return compute_sums(new_arr)

    best_set = prescribed[:]
    best_size = len(best_set)
    
    rng = random.Random(42)

    # Phase 1: Multiple Greedy Starts
    # We try to build a set greedily from scratch multiple times.
    
    def greedy_build(seed_offset):
        local_rng = random.Random(seed_offset)
        arr = prescribed[:]
        sums = compute_sums(arr)
        candidates = list(range(1, 285))
        local_rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in arr:
                continue
            if can_add(x, arr, sums):
                arr.append(x)
                sums = add_sums(x, arr[:-1], sums)
                if len(arr) >= 20:
                    break
        return arr, sums

    # Run several greedy builds
    for i in range(50):
        if time.time() > deadline:
            break
        arr, sums = greedy_build(1000 + i)
        if len(arr) > best_size:
            best_size = len(arr)
            best_set = arr[:]
            if best_size >= 20:
                break

    # Phase 2: Local Search on the best set found
    # We use a simulated annealing-like approach but with specific move types.
    
    current_set = best_set[:]
    current_sums = compute_sums(current_set)
    
    T = 1.0
    T_min = 0.01
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if best_size >= 20:
            break
            
        move_type = rng.random()
        
        if move_type < 0.3:
            # Add
            x = rng.randint(1, 284)
            if x not in current_set:
                if can_add(x, current_set, current_sums):
                    new_set = current_set + [x]
                    new_sums = add_sums(x, current_set, current_sums)
                    current_set = new_set
                    current_sums = new_sums
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
        elif move_type < 0.6:
            # Swap: Remove one non-prescribed, add one
            removable = [x for x in current_set if x not in prescribed_set]
            if removable:
                x_remove = rng.choice(removable)
                temp_set = [y for y in current_set if y != x_remove]
                temp_sums = compute_sums(temp_set)
                
                # Try to find a valid add
                # Random candidate
                x_add = rng.randint(1, 284)
                if x_add not in temp_set:
                    if can_add(x_add, temp_set, temp_sums):
                        new_set = temp_set + [x_add]
                        new_sums = add_sums(x_add, temp_set, temp_sums)
                        # Accept swap if size is same or larger
                        # Since we removed one and added one, size is same.
                        # We accept to explore.
                        current_set = new_set
                        current_sums = new_sums
                        no_improve_count += 1
        elif move_type < 0.8:
            # Shift: Move a non-prescribed element to a new random position
            removable = [x for x in current_set if x not in prescribed_set]
            if removable:
                x_old = rng.choice(removable)
                temp_set = [y for y in current_set if y != x_old]
                temp_sums = compute_sums(temp_set)
                
                x_new = rng.randint(1, 284)
                if x_new not in temp_set:
                    if can_add(x_new, temp_set, temp_sums):
                        new_set = temp_set + [x_new]
                        new_sums = add_sums(x_new, temp_set, temp_sums)
                        current_set = new_set
                        current_sums = new_sums
                        no_improve_count += 1
        else:
            # Kick: Remove 2 random non-prescribed elements and re-greedy fill
            removable = [x for x in current_set if x not in prescribed_set]
            if len(removable) >= 2:
                to_remove = rng.sample(removable, 2)
                start_set = [x for x in current_set if x not in to_remove]
                start_sums = compute_sums(start_set)
                
                # Greedy fill from this state
                candidates = list(range(1, 285))
                rng.shuffle(candidates)
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if x in start_set:
                        continue
                    if can_add(x, start_set, start_sums):
                        start_set.append(x)
                        start_sums = add_sums(x, start_set[:-1], start_sums)
                        if len(start_set) >= 20:
                            break
                
                current_set = start_set
                current_sums = start_sums
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        # Cooling
        T *= 0.99
        if T < T_min:
            T = 1.0
            # Reset no_improve to encourage more kicks if stuck?
            # Actually, if we are stuck, we should kick more often.
            
        if no_improve_count > max_no_improve:
            # Force a kick
            removable = [x for x in current_set if x not in prescribed_set]
            if removable:
                to_remove = rng.sample(removable, min(3, len(removable)))
                start_set = [x for x in current_set if x not in to_remove]
                start_sums = compute_sums(start_set)
                
                candidates = list(range(1, 285))
                rng.shuffle(candidates)
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if x in start_set:
                        continue
                    if can_add(x, start_set, start_sums):
                        start_set.append(x)
                        start_sums = add_sums(x, start_set[:-1], start_sums)
                        if len(start_set) >= 20:
                            break
                current_set = start_set
                current_sums = start_sums
                no_improve_count = 0

    # Final validation
    final_sums = set()
    lst = sorted(best_set)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            final_sums.add(lst[i] + lst[j])
    
    expected = len(lst) * (len(lst) + 1) // 2
    if len(final_sums) != expected:
        return sorted(prescribed)
        
    return lst
