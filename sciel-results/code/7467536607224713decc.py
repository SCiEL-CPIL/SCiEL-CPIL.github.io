import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    N = 1000
    max_sum = 2 * N
    sum_counts = [0] * (max_sum + 1)
    
    def can_add(x, chosen_list):
        """Check if x can be added to chosen_list without violating B2[2]."""
        if sum_counts[2 * x] >= 2:
            return False
        for y in chosen_list:
            if sum_counts[x + y] >= 2:
                return False
        return True

    def add(x, chosen_list, chosen_set):
        """Add x to chosen_list and update sum_counts."""
        sum_counts[2 * x] += 1
        for y in chosen_list:
            sum_counts[x + y] += 1
        chosen_list.append(x)
        chosen_set.add(x)

    def remove(x, chosen_list, chosen_set):
        """Remove x from chosen_list and update sum_counts."""
        sum_counts[2 * x] -= 1
        # Find index to pop efficiently
        idx = chosen_list.index(x)
        for j in range(len(chosen_list)):
            if j == idx:
                continue
            y = chosen_list[j]
            sum_counts[x + y] -= 1
        chosen_list.pop(idx)
        chosen_set.discard(x)

    def run_greedy(seed):
        """Run a randomized greedy construction."""
        rng = random.Random(seed)
        # Reset counts
        for i in range(max_sum + 1):
            sum_counts[i] = 0
        
        chosen_list = []
        chosen_set = set()
        
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, chosen_list):
                add(x, chosen_list, chosen_set)
                
        return list(chosen_set), dict(sum_counts) # Return copy of counts? No, we manage global state. 
        # Actually, better to return list and let caller rebuild or use global.
        # Let's return list only and rebuild counts if needed, or keep global.
        # Since we mutate global sum_counts, we need to be careful.
        # Let's make run_greedy self-contained with local counts for safety during restarts, 
        # then load the best one into global for local search.

    # Phase 1: Find a good starting set via multiple greedy restarts
    best_start = []
    best_start_size = 0
    best_counts_copy = None
    
    # We need a local version of counts for the restarts to not interfere
    # Or we can just do it sequentially and save the best state.
    
    for s in range(20):
        if time.time() > deadline - 10:
            break
        # Use a local sum_counts for the greedy run to allow parallelism in thought, 
        # but here it's sequential. 
        local_counts = [0] * (max_sum + 1)
        chosen_list = []
        chosen_set = set()
        
        rng = random.Random(s)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            # Check with local_counts
            ok = True
            if local_counts[2 * x] >= 2:
                ok = False
            else:
                for y in chosen_list:
                    if local_counts[x + y] >= 2:
                        ok = False
                        break
            
            if ok:
                local_counts[2 * x] += 1
                for y in chosen_list:
                    local_counts[x + y] += 1
                chosen_list.append(x)
                chosen_set.add(x)
        
        if len(chosen_list) > best_start_size:
            best_start_size = len(chosen_list)
            best_start = chosen_list[:]
            best_counts_copy = local_counts[:]
            
    # Initialize global state with best start
    sum_counts[:] = best_counts_copy
    current_list = best_start[:]
    current_set = set(best_start)
    
    best_size = len(current_list)
    
    # Phase 2: Local Search
    # Strategy: Try to remove one element and add two others.
    # This increases size by 1.
    
    rng = random.Random(42)
    
    # Pre-compute candidates list for efficiency
    all_candidates = list(range(1, N + 1))
    
    # We will try to improve by removing a random element and trying to fill with 2 new ones.
    # To speed up, we can shuffle candidates once and try to add them greedily after removal.
    
    iterations = 0
    max_iterations = 50000
    stall_count = 0
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        if len(current_list) <= 1:
            break
            
        # Pick a random element to remove
        # Bias towards "expensive" elements? Or just random.
        idx_to_remove = rng.randint(0, len(current_list) - 1)
        u = current_list[idx_to_remove]
        
        # Save state
        saved_list = current_list[:]
        saved_set = set(current_set)
        saved_counts = sum_counts[:]
        
        # Remove u
        remove(u, current_list, current_set)
        
        # Try to add two new elements
        # Shuffle candidates not in current_set
        candidates = [x for x in all_candidates if x not in current_set]
        rng.shuffle(candidates)
        
        added_count = 0
        added_elements = []
        
        # We need to add 2 elements. 
        # Try to add first element a
        a_added = False
        a_val = None
        for x in candidates:
            if can_add(x, current_list):
                add(x, current_list, current_set)
                added_count += 1
                added_elements.append(x)
                a_val = x
                a_added = True
                break
        
        if not a_added:
            # Revert
            current_list[:] = saved_list
            current_set = saved_set
            sum_counts[:] = saved_counts
            stall_count += 1
            continue
            
        # Try to add second element b
        # Update candidates list to exclude a_val and current elements
        # Note: current_list now includes a_val
        candidates_b = [x for x in candidates if x != a_val and x not in current_set]
        rng.shuffle(candidates_b)
        
        b_added = False
        for x in candidates_b:
            if can_add(x, current_list):
                add(x, current_list, current_set)
                added_count += 1
                added_elements.append(x)
                b_added = True
                break
        
        if b_added:
            # Success! Size increased by 1 (removed 1, added 2)
            best_size = len(current_list)
            stall_count = 0
            # Do not revert
        else:
            # Failed to add 2nd element. 
            # We could accept if we added 1 (size stays same, but set changed) - usually not useful for maximizing size.
            # Revert
            current_list[:] = saved_list
            current_set = saved_set
            sum_counts[:] = saved_counts
            stall_count += 1
            
            # If we stall too much, try a different strategy or break
            if stall_count > 500:
                # Try a simple 1-for-1 swap or just break if no progress
                # For now, just continue, maybe randomize the removal index differently
                if stall_count > 2000:
                    break

    return current_list

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    # Verify
    from collections import defaultdict
    counts = defaultdict(int)
    n = len(result)
    res = sorted(result)
    for i in range(n):
        for j in range(i, n):
            counts[res[i] + res[j]] += 1
    max_c = max(counts.values()) if counts else 0
    print(f"Max representation count: {max_c}")
    assert max_c <= 2, "Invalid B2[2] set!"
