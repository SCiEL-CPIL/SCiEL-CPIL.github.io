import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10-second safety margin
    
    best_set = set()
    
    # Helper function to check if adding x to current_set is valid
    def can_add(x, current_set, counts):
        # Check 2x
        if counts.get(2*x, 0) >= 2:
            return False
            
        # Check x+y for y in current_set
        for y in current_set:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def update_counts(x, current_set, counts):
        counts[2*x] = counts.get(2*x, 0) + 1
        for y in current_set:
            s = x + y
            counts[s] = counts.get(s, 0) + 1

    def revert_counts(x, current_set, counts):
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        for y in current_set:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]

    def local_improve(current_set, counts, deadline):
        """Try to improve the set by swapping elements or adding new ones."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # First try to add any unselected element
            for x in range(1, 1001):
                if x not in current_set and can_add(x, current_set, counts):
                    current_set.add(x)
                    update_counts(x, current_set, counts)
                    improved = True
                    break
            
            # If no addition possible, try swaps: remove one, try to add another
            if not improved:
                for remove_x in list(current_set):
                    if time.time() > deadline:
                        return current_set, counts
                    # Temporarily remove remove_x
                    revert_counts(remove_x, current_set - {remove_x}, counts)
                    current_set.discard(remove_x)
                    
                    # Try to add a different element
                    for add_x in range(1, 1001):
                        if add_x == remove_x or add_x in current_set:
                            continue
                        if can_add(add_x, current_set, counts):
                            current_set.add(add_x)
                            update_counts(add_x, current_set, counts)
                            improved = True
                            break
                    
                    # If we didn't find a swap, put remove_x back
                    if not improved:
                        current_set.add(remove_x)
                        update_counts(remove_x, current_set, counts)
                    
                    if improved:
                        break
        
        return current_set, counts

    candidates = list(range(1, 1001))
    
    # Try multiple random restarts
    while time.time() < deadline:
        random.shuffle(candidates)
        current_set = set()
        counts = {}
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, current_set, counts):
                current_set.add(x)
                update_counts(x, current_set, counts)
        
        # Local improvement phase
        if time.time() < deadline - 5:  # Leave some time for other restarts
            current_set, counts = local_improve(current_set, counts, deadline)
                
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
    return list(best_set)
