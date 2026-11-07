import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def is_valid(v, current_sums):
        # Check distinctness
        # New sums: {s} U {s+v for s in current_sums}
        # We can check if v is a difference of two elements in current_sums
        # But we also need the set for divisibility.
        
        # Generate new sums
        # Optimization: If len(current_sums) is large, this is the bottleneck.
        # But max size is small (~10-15).
        
        new_sums = set()
        for s in current_sums:
            new_sums.add(s)
            new_sums.add(s + v)
            
        # If size didn't double, there was a collision
        if len(new_sums) != 2 * len(current_sums):
            return False, None

        # Divisibility check
        # Remove 0
        non_empty = list(new_sums - {0})
        if not non_empty:
            return True, new_sums
            
        max_sum = max(non_empty)
        sum_set = set(non_empty)
        
        # Check if any x divides y
        # Iterate x
        # To speed up, we can sort and use two pointers? No, multiples are sparse.
        # Just check multiples.
        
        for x in non_empty:
            # If x is large, no multiple fits
            if x > max_sum // 2:
                continue
            m = x + x
            while m <= max_sum:
                if m in sum_set:
                    return False, None
                m += x
                
        return True, new_sums

    def run_greedy(order):
        curr_sums = {0}
        curr_set = []
        for v in order:
            if time.time() > deadline:
                break
            valid, new_sums = is_valid(v, curr_sums)
            if valid:
                curr_set.append(v)
                curr_sums = new_sums
        return curr_set

    # Try different orders
    candidates = [
        list(range(1, 257)),       # Ascending
        list(range(256, 0, -1)),   # Descending
    ]
    
    # Add a few random shuffles
    for _ in range(5):
        rand_order = list(range(1, 257))
        random.shuffle(rand_order)
        candidates.append(rand_order)
        
    best_set = []
    for order in candidates:
        if time.time() > deadline:
            break
        s = run_greedy(order)
        if len(s) > len(best_set):
            best_set = s
            
    return best_set
