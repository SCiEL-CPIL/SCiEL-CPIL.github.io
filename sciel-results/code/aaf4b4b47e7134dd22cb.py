import random
import time

def _all_subset_sums_distinct(arr):
    """Check if all subset sums of arr are distinct."""
    sums = {0}
    for x in arr:
        new_sums = {s + x for s in sums}
        if sums & new_sums:
            return False
        sums |= new_sums
    return True

def _max_element(arr):
    return max(arr)

def construct():
    # Start with powers of 2
    arr = [1 << i for i in range(12)]
    best_arr = arr[:]
    best_max = max(arr)
    
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    
    current_max = best_max
    
    while time.time() < deadline:
        # Pick a random index to change
        i = random.randint(0, 11)
        old_val = arr[i]
        
        # Try a new value
        # We want to keep values positive and distinct
        # Let's try values in a range around old_val
        # To minimize max, we might want to reduce large values
        if i == 11:
            # For the largest element, try reducing it
            new_val = max(1, old_val - random.randint(1, 50))
        else:
            # For other elements, try small changes
            new_val = max(1, old_val + random.randint(-10, 10))
        
        # Ensure distinctness of the set itself
        if new_val in arr and new_val != old_val:
            continue
            
        arr[i] = new_val
        
        if _all_subset_sums_distinct(arr):
            new_max = max(arr)
            if new_max < current_max:
                current_max = new_max
                best_arr = arr[:]
                best_max = new_max
            else:
                # Accept with probability exp(-(new_max - current_max) / T)
                delta = new_max - current_max
                if delta < 0:
                    arr[i] = new_val
                    current_max = new_max
                else:
                    prob = pow(2.71828, -delta / T)
                    if random.random() < prob:
                        arr[i] = new_val
                        current_max = new_max
                    else:
                        arr[i] = old_val
        else:
            arr[i] = old_val
            
        T *= cooling
        
    return best_arr
