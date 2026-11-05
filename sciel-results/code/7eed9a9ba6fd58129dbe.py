import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    prescribed = [1, 2, 5, 11]
    prescribed_set = frozenset(prescribed)
    best_size = 0
    best_set = set(prescribed_set)
    
    # Pre-compute sums for prescribed set
    def get_sums(s):
        sums = set()
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sums.add(lst[i] + lst[j])
        return sums

    initial_sums = get_sums(prescribed_set)
    
    # Helper to check if x can be added to a set with given sums
    def can_add(x, s_set, s_sums):
        if x in s_set:
            return False
        # Check x + y for all y in s_set
        for y in s_set:
            if x + y in s_sums:
                return False
        # Check 2x
        if 2 * x in s_sums:
            return False
        return True

    # Helper to update sums when adding x
    def add_to_sums(x, s_set, s_sums):
        new_sums = set(s_sums)
        for y in s_set:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums

    def greedy_fill(current_set, current_sums, candidates):
        """Greedily add elements from candidates to current_set."""
        for x in candidates:
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                current_set.add(x)
                current_sums = add_to_sums(x, current_set, current_sums)
        return current_set, current_sums

    def kick_search(current_set, current_sums, time_limit=5.0):
        """
        Perform kick-based local search.
        Removes k random elements and greedily refills.
        """
        best_local_set = set(current_set)
        best_local_sums = current_sums
        start = time.time()
        
        all_candidates = list(range(1, 285))
        
        while time.time() - start < time_limit:
            # Decide kick size: 1 or 2
            kick_size = 1 if len(current_set) < 15 else random.choice([1, 2])
            
            removable = [x for x in current_set if x not in prescribed_set]
            if len(removable) < kick_size:
                break
                
            # Select elements to remove
            to_remove = random.sample(removable, kick_size)
            
            # Remove them
            temp_set = set(current_set)
            for x in to_remove:
                temp_set.remove(x)
            temp_sums = get_sums(temp_set)
            
            # Greedily add back from all candidates not in temp_set
            # Shuffle candidates to randomize the fill order
            candidates = [x for x in all_candidates if x not in temp_set]
            random.shuffle(candidates)
            
            temp_set, temp_sums = greedy_fill(temp_set, temp_sums, candidates)
            
            # Update local best
            if len(temp_set) > len(best_local_set):
                best_local_set = set(temp_set)
                best_local_sums = temp_sums
                current_set = set(temp_set)
                current_sums = temp_sums
            elif len(temp_set) == len(best_local_set) and random.random() < 0.3:
                # Accept same-size solution with some probability to explore
                current_set = set(temp_set)
                current_sums = temp_sums
        
        return best_local_set, best_local_sums

    # Main search loop
    num_restarts = 0
    while time.time() < deadline - 10:  # Leave 10s for final polish
        num_restarts += 1
        
        # 1. Random Greedy Construction
        current_set = set(prescribed_set)
        current_sums = get_sums(current_set)
        
        candidates_list = list(range(12, 285))
        random.shuffle(candidates_list)
        
        current_set, current_sums = greedy_fill(current_set, current_sums, candidates_list)
        
        # 2. Kick-based Local Search
        # Allocate time for kick search
        remaining = deadline - time.time()
        kick_time = min(5.0, max(0.5, remaining * 0.1))
        current_set, current_sums = kick_search(current_set, current_sums, kick_time)
        
        # 3. Update Best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
            # If we hit 20, it's likely optimal.
            if best_size >= 20:
                break

    # Final polish: try all single swaps on the best set
    if time.time() < deadline - 2:
        best_sums = get_sums(best_set)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            removable = [x for x in best_set if x not in prescribed_set]
            for x_remove in removable:
                if time.time() >= deadline - 2:
                    break
                temp_set = set(best_set)
                temp_set.remove(x_remove)
                temp_sums = get_sums(temp_set)
                
                # Try all possible additions
                for y_add in range(1, 285):
                    if y_add in temp_set:
                        continue
                    if can_add(y_add, temp_set, temp_sums):
                        # Check if this swap actually improves (it should, since we're adding)
                        new_set = set(temp_set)
                        new_set.add(y_add)
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            best_sums = get_sums(best_set)
                            improved = True
                            break
                if improved:
                    break

    # Final validation before returning
    final_sums = get_sums(best_set)
    if len(final_sums) != len(best_set) * (len(best_set) + 1) // 2:
        # Fallback to prescribed set if something went wrong
        return sorted(prescribed)
        
    return sorted(list(best_set))
