import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    elements = list(range(1, N + 1))
    
    # Precompute neighbors
    neighbors = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # Initial solution: all odd numbers
    current_set = set(range(1, N + 1, 2))
    best_set = set(current_set)
    
    counts = [0] * (N + 1)
    for x in range(1, N + 1):
        if x not in current_set:
            counts[x] = len(neighbors[x] & current_set)
            
    def add_element(x, current_set, counts):
        current_set.add(x)
        for y in neighbors[x]:
            if y not in current_set:
                counts[y] += 1
                
    def remove_element(x, current_set, counts):
        current_set.remove(x)
        for y in neighbors[x]:
            if y not in current_set:
                counts[y] -= 1
                
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add elements
        candidates = [x for x in range(1, N + 1) if x not in current_set and counts[x] == 0]
        if candidates:
            # Greedy add
            while candidates:
                # Sort by degree in remaining graph? Or just random
                random.shuffle(candidates)
                x = candidates[0]
                if counts[x] == 0:
                    add_element(x, current_set, counts)
                    candidates.remove(x)
                    improved = True
                    # Update candidates
                    candidates = [c for c in candidates if counts[c] == 0]
                else:
                    candidates.remove(x)
                    
            if len(current_set) > len(best_set):
                best_set = set(current_set)
            continue
            
        # 2. Try swaps
        # For each y in current_set, see if we can remove it and add an x
        # x must satisfy: N(x) & (current_set - {y}) == empty
        # i.e., N(x) & current_set is a subset of {y}
        # This means counts[x] <= 1 and if counts[x]==1, the only conflict is y.
        
        # We can iterate over x not in current_set
        # For each x, if counts[x] == 0, we could have added it in step 1.
        # So we look for x with counts[x] == 1.
        # Let y be the unique element in current_set that conflicts with x.
        # If we remove y, x becomes addable.
        # So the swap is: remove y, add x.
        # This is valid if x is not in current_set.
        
        swaps = []
        for x in range(1, N + 1):
            if x in current_set:
                continue
            if counts[x] == 1:
                # Find the unique y
                # We can just check neighbors[x] & current_set
                conflicts = neighbors[x] & current_set
                if len(conflicts) == 1:
                    y = next(iter(conflicts))
                    # Check if swapping y for x is beneficial?
                    # It maintains size. But it might open up more possibilities.
                    # Let's record this potential swap.
                    swaps.append((x, y))
                    
        if swaps:
            # Try a random swap
            random.shuffle(swaps)
            x, y = swaps[0]
            # Perform swap
            remove_element(y, current_set, counts)
            if counts[x] == 0: # Should be 0 after removing y
                add_element(x, current_set, counts)
                improved = True
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
            else:
                # Something went wrong, revert
                add_element(y, current_set, counts)
                remove_element(x, current_set, counts) # x wasn't added, so this is wrong logic
                # Actually, if counts[x] != 0, we shouldn't have added it.
                # So we just revert the removal of y.
                add_element(y, current_set, counts)
            continue
            
        # 3. Perturbation
        # Remove k random elements
        k = random.randint(1, 4)
        removed = []
        for _ in range(k):
            if not current_set:
                break
            y = random.choice(list(current_set))
            remove_element(y, current_set, counts)
            removed.append(y)
            
        # Try to re-add
        random.shuffle(removed)
        for y in removed:
            if counts[y] == 0:
                add_element(y, current_set, counts)
                
        # Try to add new
        candidates = [x for x in range(1, N + 1) if x not in current_set and counts[x] == 0]
        if candidates:
            random.shuffle(candidates)
            for x in candidates:
                if counts[x] == 0:
                    add_element(x, current_set, counts)
                    candidates = [c for c in candidates if counts[c] == 0]
                    
        if len(current_set) > len(best_set):
            best_set = set(current_set)
            improved = True
        else:
            # If we didn't improve, maybe restart from best_set?
            # To avoid getting stuck in local minima.
            if random.random() < 0.1:
                current_set = set(best_set)
                counts = [0] * (N + 1)
                for x in range(1, N + 1):
                    if x not in current_set:
                        counts[x] = len(neighbors[x] & current_set)

    return sorted(list(best_set))
