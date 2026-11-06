"""Reciprocal sets with all subset sums distinct."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    reciprocals = [1.0 / i for i in range(1, 23)]
    
    def check_valid(subset):
        """Check if all 2^k subset sums are distinct."""
        k = len(subset)
        if k > 22:
            return False
        sums = [0.0]
        for num in subset:
            r = reciprocals[num - 1]
            sums = sums + [s + r for s in sums]
        return len(sums) == len(set(sums))
    
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Greedy: try to build larger set by adding elements in various orders
    def greedy(order):
        current = []
        sums = [0.0]
        for num in order:
            r = reciprocals[num - 1]
            new_sums = [s + r for s in sums]
            all_sums = sums + new_sums
            if len(all_sums) == len(set(all_sums)):
                current.append(num)
                sums = all_sums
        return current
    
    # Try several orderings
    orderings = [
        list(range(22, 0, -1)),
        list(range(1, 23)),
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 4, 8, 16, 3, 6, 12, 5, 10, 20, 7, 14, 11, 22, 13, 17, 19, 15, 18, 21, 9],
    ]
    
    for order in orderings:
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
        if time.time() > deadline:
            break
    
    # Backtracking search for larger sets
    def backtrack(target_size, current_set, current_sums, start_idx):
        if time.time() > deadline:
            return None
        if len(current_set) == target_size:
            return list(current_set)
        
        # Pruning: not enough elements remaining
        if len(current_set) + (22 - start_idx) < target_size:
            return None
        
        for i in range(start_idx, 23):
            num = i + 1
            r = reciprocals[i]
            new_sums = [s + r for s in current_sums]
            
            # Check for collisions
            combined = current_sums + new_sums
            if len(combined) != len(set(combined)):
                continue
            
            current_set.append(num)
            result = backtrack(target_size, current_set, combined, i + 1)
            if result is not None:
                return result
            current_set.pop()
        
        return None
    
    # Try to find sets larger than best_size
    for target_size in range(best_size + 1, 23):
        if time.time() > deadline:
            break
        result = backtrack(target_size, [], [0.0], 0)
        if result is not None:
            best_set = result
            best_size = target_size
    
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
