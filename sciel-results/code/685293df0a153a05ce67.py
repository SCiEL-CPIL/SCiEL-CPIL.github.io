import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = 200
    max_val = 100
    all_elements = list(range(1, max_val + 1))
    
    def compute_mask(elements):
        mask = 1
        for x in elements:
            if x > target:
                continue
            mask = mask | (mask << x)
            mask &= (1 << (target + 1)) - 1
        return mask

    def is_valid(elements):
        mask = compute_mask(elements)
        return not ((mask >> target) & 1)

    def find_conflict(elements):
        # Returns a subset summing to target, or None
        reachable = [False] * (target + 1)
        parent = [None] * (target + 1)
        reachable[0] = True
        for x in elements:
            if x > target:
                continue
            for s in range(target, x - 1, -1):
                if not reachable[s] and reachable[s - x]:
                    reachable[s] = True
                    parent[s] = (s - x, x)
        if not reachable[target]:
            return None
        subset = []
        s = target
        while s > 0:
            prev_s, x = parent[s]
            subset.append(x)
            s = prev_s
        return subset

    # Start with a strong heuristic candidate
    # We know {66..100} is valid (size 35)
    # Let's try to beat it.
    
    best_set = list(range(66, 101))
    if not is_valid(best_set):
        # Fallback if my manual analysis was wrong
        best_set = list(range(1, 101))
        while not is_valid(best_set) and time.time() < deadline - 5.0:
            c = find_conflict(best_set)
            if c:
                # Remove a random element from the conflict
                idx = random.randint(0, len(c)-1)
                best_set.remove(c[idx])
    
    best_size = len(best_set)
    
    # Local Search: Try to add elements from 1..65
    # If adding fails, try swapping? 
    # For now, just try adding to the best set found so far.
    # We also run a few random restarts to be safe.
    
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            current_set = best_set[:]
        else:
            # Random valid set generation
            current_set = []
            random.seed(restart * 100)
            candidates = all_elements[:]
            random.shuffle(candidates)
            mask = 1
            for x in candidates:
                new_mask = mask | (mask << x)
                new_mask &= (1 << (target + 1)) - 1
                if not ((new_mask >> target) & 1):
                    current_set.append(x)
                    mask = new_mask
        
        # Try to add remaining elements
        current_mask = compute_mask(current_set)
        for x in range(1, 101):
            if x in current_set:
                continue
            new_mask = current_mask | (current_mask << x)
            new_mask &= (1 << (target + 1)) - 1
            if not ((new_mask >> target) & 1):
                current_set.append(x)
                current_mask = new_mask
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Best size: {len(res)}")
    print(f"Set: {sorted(res)}")
    # Verify
    mask = 1
    for x in res:
        mask = mask | (mask << x)
        mask &= (1 << 201) - 1
    print(f"Valid (no sum 200): {not ((mask >> 200) & 1)}")
