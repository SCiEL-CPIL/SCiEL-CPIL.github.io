import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds margin
    
    # Helper to check if a set of sums is valid (no divisibility)
    # Uses boolean array for speed
    def is_divisibility_free(sums):
        if not sums:
            return True
        max_sum = max(sums)
        present = [False] * (max_sum + 1)
        for s in sums:
            present[s] = True
            
        for s in sums:
            if s > max_sum // 2:
                continue
            multiple = 2 * s
            while multiple <= max_sum:
                if present[multiple]:
                    return False
                multiple += s
        return True

    # Fast check for adding x to S with current_sums
    def can_add(S, x, current_sums):
        # 1. Generate new sums
        new_sums = set()
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)
            
        # 2. Check for duplicate sums
        if current_sums & new_sums:
            return False, None

        # 3. Check divisibility on the union
        all_sums = current_sums | new_sums
        
        max_sum = max(all_sums)
        present = [False] * (max_sum + 1)
        for s in all_sums:
            present[s] = True
            
        for s in all_sums:
            if s > max_sum // 2:
                continue
            multiple = 2 * s
            while multiple <= max_sum:
                if present[multiple]:
                    return False, None
                multiple += s
                
        return True, new_sums

    # Initial best set: a known valid set of size 3
    best_set = [11, 13, 17]
    best_size = len(best_set)
    
    candidates = list(range(1, 257))
    
    # Strategy 1: Greedy with specific orderings
    def run_greedy(ordered_candidates):
        current_set = []
        current_sums = set()
        
        for x in ordered_candidates:
            if time.time() > deadline:
                break
            valid, new_sums = can_add(current_set, x, current_sums)
            if valid:
                current_set.append(x)
                current_sums = current_sums.union(new_sums)
                
        return current_set, current_sums

    # Try descending order (often best for this problem)
    ordered_desc = sorted(candidates, reverse=True)
    set_desc, sums_desc = run_greedy(ordered_desc)
    if len(set_desc) > best_size:
        best_size = len(set_desc)
        best_set = set_desc.copy()

    # Try ascending order
    ordered_asc = sorted(candidates)
    set_asc, sums_asc = run_greedy(ordered_asc)
    if len(set_asc) > best_size:
        best_size = len(set_asc)
        best_set = set_asc.copy()

    # Try random restarts
    max_restarts = 2000
    for _ in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        random.shuffle(candidates)
        # Bias towards larger numbers in random restarts?
        # Simple shuffle first
        current_set, current_sums = run_greedy(candidates)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
        # If we found a good set, try to extend it with remaining candidates in random order
        if len(current_set) >= best_size and len(current_set) > 0:
            used = set(current_set)
            remaining = [x for x in range(1, 257) if x not in used]
            random.shuffle(remaining)
            
            extended = current_set.copy()
            ext_sums = current_sums.copy()
            
            for x in remaining:
                if time.time() > deadline - 2.0:
                    break
                valid, new_sums = can_add(extended, x, ext_sums)
                if valid:
                    extended.append(x)
                    ext_sums = ext_sums.union(new_sums)
                    if len(extended) > best_size:
                        best_size = len(extended)
                        best_set = extended.copy()

    # Local Search: Try to swap elements to allow more additions
    # This is a bit complex, so we'll do a simpler "try to add any remaining number" loop
    # on the best set found, trying all remaining numbers in a few orders.
    
    if time.time() < deadline - 10.0:
        # Re-evaluate best_set to get its sums
        current_sums = set()
        for x in best_set:
            new_sums = set()
            new_sums.add(x)
            for s in current_sums:
                new_sums.add(s + x)
            current_sums = current_sums.union(new_sums)
        
        used = set(best_set)
        remaining = [x for x in range(1, 257) if x not in used]
        
        # Try adding in descending order
        for x in sorted(remaining, reverse=True):
            if time.time() > deadline - 5.0:
                break
            valid, new_sums = can_add(best_set, x, current_sums)
            if valid:
                best_set.append(x)
                current_sums = current_sums.union(new_sums)
                best_size = len(best_set)
                
        # Try adding in random order (a few times)
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            random.shuffle(remaining)
            # Reset best_set to its state before this round? 
            # Actually, we modified best_set above. Let's keep the improvements.
            # But 'remaining' list is stale.
            used = set(best_set)
            remaining = [x for x in range(1, 257) if x not in used]
            if not remaining:
                break
            random.shuffle(remaining)
            
            # Recalculate sums for current best_set
            current_sums = set()
            for x in best_set:
                new_sums = set()
                new_sums.add(x)
                for s in current_sums:
                    new_sums.add(s + x)
                current_sums = current_sums.union(new_sums)
                
            for x in remaining:
                if time.time() > deadline - 5.0:
                    break
                valid, new_sums = can_add(best_set, x, current_sums)
                if valid:
                    best_set.append(x)
                    current_sums = current_sums.union(new_sums)
                    best_size = len(best_set)
                    # Update remaining
                    remaining.remove(x)
                    if not remaining:
                        break

    # Final verification
    if best_size >= 2:
        final_sums = set()
        for x in best_set:
            new_sums = set()
            new_sums.add(x)
            for s in final_sums:
                new_sums.add(s + x)
            final_sums = final_sums.union(new_sums)
        
        if not is_divisibility_free(final_sums):
            # Fallback
            best_set = [11, 13, 17]
            best_size = 3

    return [int(x) for x in best_set]
