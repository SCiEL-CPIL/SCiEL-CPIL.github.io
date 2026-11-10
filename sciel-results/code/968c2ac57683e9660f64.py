import random
import time

def construct():
    start_time = time.time()
    deadline = 85.0
    
    # Initial solution: Powers of 2
    a = [1 << i for i in range(12)]
    best_a = a[:]
    best_max = max(a)
    
    def get_sums(arr):
        sums = [0]
        for x in arr:
            sums += [s + x for s in sums]
        return sums
        
    def is_valid(arr):
        sums = get_sums(arr)
        return len(sums) == len(set(sums))
        
    # Local Search
    current_a = a[:]
    
    while time.time() < deadline:
        # Try to reduce the largest element
        # Pick a random index, biased towards the end
        idx = random.randint(0, 11)
        # Try to reduce a[idx]
        # Ensure it stays positive and distinct
        min_val = 1
        # If idx is the max, we can try to reduce it more aggressively
        if current_a[idx] == max(current_a):
            # Try values from max-1 down to min_val
            # But random search is better for exploration
            new_val = random.randint(min_val, current_a[idx] - 1)
        else:
            new_val = random.randint(min_val, max(current_a) - 1)
            
        new_a = current_a[:]
        new_a[idx] = new_val
        new_a.sort()
        
        # Check distinctness of elements
        if len(set(new_a)) < 12:
            continue
            
        # Check subset sums
        if is_valid(new_a):
            new_max = max(new_a)
            if new_max < best_max:
                best_max = new_max
                best_a = new_a
                current_a = new_a
            elif new_max == best_max and random.random() < 0.1:
                current_a = new_a
            # If new_max > best_max, we might accept with low probability to escape local minima
            # But we want to minimize max, so rejecting is usually better.
            
    return best_a
