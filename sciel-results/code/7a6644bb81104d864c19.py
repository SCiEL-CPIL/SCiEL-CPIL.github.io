import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    fixed = {1, 2, 5, 11}
    best_set = list(fixed)
    best_size = len(fixed)
    
    def get_sums(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums

    def is_valid_addition(x, current_set, current_sums):
        # Check if x can be added to current_set
        for y in current_set:
            if x + y in current_sums:
                return False
        if x + x in current_sums:
            return False
        return True

    def add_to_set(x, current_set, current_sums):
        for y in current_set:
            current_sums.add(x + y)
        current_sums.add(x + x)
        current_set.append(x)
        current_set.sort()

    def remove_from_set(x, current_set, current_sums):
        for y in current_set:
            if y != x:
                current_sums.discard(x + y)
        current_sums.discard(x + x)
        current_set.remove(x)

    def try_greedy_fill(current_set, current_sums, candidates_list):
        """Try to add as many candidates as possible in the given order."""
        added = 0
        for x in candidates_list:
            if time.time() > deadline:
                break
            if is_valid_addition(x, current_set, current_sums):
                add_to_set(x, current_set, current_sums)
                added += 1
        return added

    # Initial greedy fill with multiple random orderings
    current_set = list(fixed)
    current_sums = get_sums(current_set)
    
    candidates = [x for x in range(1, 285) if x not in fixed]
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if is_valid_addition(x, current_set, current_sums):
            add_to_set(x, current_set, current_sums)
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = list(current_set)
        
    # Local search loop with swap moves
    attempts = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        attempts += 1
        current_size = len(current_set)
        
        # Destroy: Remove 1 to 4 random non-fixed elements
        removable = [x for x in current_set if x not in fixed]
        if not removable:
            break
            
        num_remove = min(len(removable), random.randint(1, 4))
        to_remove = random.sample(removable, num_remove)
        
        for x in to_remove:
            remove_from_set(x, current_set, current_sums)
            
        # Repair: Try multiple strategies
        # Strategy 1: Random order
        all_candidates = [x for x in range(1, 285) if x not in current_set]
        random.shuffle(all_candidates)
        try_greedy_fill(current_set, current_sums, all_candidates)
        
        # Strategy 2: Ascending order (sometimes finds denser solutions)
        if time.time() < deadline - 5:
            # Reset to the state before strategy 2 by trying from current state
            # Actually, let's just try ascending from current state
            asc_candidates = [x for x in range(1, 285) if x not in current_set]
            try_greedy_fill(current_set, current_sums, asc_candidates)
            
        # Strategy 3: Descending order
        if time.time() < deadline - 5:
            desc_candidates = [x for x in range(1, 285) if x not in current_set]
            desc_candidates.sort(reverse=True)
            try_greedy_fill(current_set, current_sums, desc_candidates)
        
        # Check for improvement
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            no_improve_count = 0
            
            if best_size >= 20:
                break
        else:
            no_improve_count += 1
            
        # If we didn't improve, try a swap move: remove one element and see if we can add two
        if len(current_set) > 4 and time.time() < deadline - 10:
            removable = [x for x in current_set if x not in fixed]
            if removable:
                x_to_swap = random.choice(removable)
                remove_from_set(x_to_swap, current_set, current_sums)
                
                # Try to add back x_to_swap plus potentially more
                all_candidates = [x for x in range(1, 285) if x not in current_set]
                # Prioritize x_to_swap and nearby values
                random.shuffle(all_candidates)
                try_greedy_fill(current_set, current_sums, all_candidates)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                    no_improve_count = 0
                    if best_size >= 20:
                        break
                else:
                    no_improve_count += 1
    
    # Final polish: Try to add any remaining valid elements in a systematic way
    if time.time() < deadline - 2:
        # Try multiple random orderings for final polish
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            all_candidates = [x for x in range(1, 285) if x not in best_set]
            random.shuffle(all_candidates)
            
            # Work on a copy to see if we can improve
            test_set = list(best_set)
            test_sums = get_sums(test_set)
            
            for x in all_candidates:
                if time.time() > deadline - 2:
                    break
                if is_valid_addition(x, test_set, test_sums):
                    add_to_set(x, test_set, test_sums)
            
            if len(test_set) > best_size:
                best_size = len(test_set)
                best_set = list(test_set)
                if best_size >= 20:
                    break
    
    return best_set
