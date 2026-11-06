import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    N = 1000
    max_sum = 2 * N
    sum_counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    def can_add_fast(x, chosen_list):
        """Check if x can be added without violating B2[2] using numpy."""
        if sum_counts[2 * x] >= 2:
            return False
        if len(chosen_list) == 0:
            return True
        sums = x + np.array(chosen_list, dtype=np.int32)
        return not np.any(sum_counts[sums] >= 2)

    def add_fast(x, chosen_list):
        """Add x and update sum_counts using numpy."""
        sum_counts[2 * x] += 1
        if chosen_list:
            sums = x + np.array(chosen_list, dtype=np.int32)
            sum_counts[sums] += 1
        chosen_list.append(x)

    def remove_fast(x, chosen_list):
        """Remove x and update sum_counts using numpy."""
        sum_counts[2 * x] -= 1
        idx = chosen_list.index(x)
        if idx > 0:
            sums = x + np.array(chosen_list[:idx], dtype=np.int32)
            sum_counts[sums] -= 1
        if idx < len(chosen_list) - 1:
            sums = x + np.array(chosen_list[idx+1:], dtype=np.int32)
            sum_counts[sums] -= 1
        chosen_list.pop(idx)

    # Phase 1: Find a good starting set via multiple greedy restarts with numpy
    best_start = []
    best_start_size = 0
    best_counts_copy = None
    
    for s in range(30):
        if time.time() > deadline - 15:
            break
        local_counts = np.zeros(max_sum + 1, dtype=np.int32)
        chosen_list = []
        
        rng = random.Random(s)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if local_counts[2 * x] >= 2:
                continue
            if chosen_list:
                sums = x + np.array(chosen_list, dtype=np.int32)
                if np.any(local_counts[sums] >= 2):
                    continue
            local_counts[2 * x] += 1
            if chosen_list:
                sums = x + np.array(chosen_list, dtype=np.int32)
                local_counts[sums] += 1
            chosen_list.append(x)
        
        if len(chosen_list) > best_start_size:
            best_start_size = len(chosen_list)
            best_start = chosen_list[:]
            best_counts_copy = local_counts.copy()
    
    # Initialize with best start
    sum_counts[:] = best_counts_copy
    current_list = best_start[:]
    current_set = set(best_start)
    best_size = len(current_list)
    
    # Phase 2: Local Search with improved heuristics
    rng = random.Random(42)
    all_candidates = list(range(1, N + 1))
    
    iterations = 0
    max_iterations = 200000
    stall_count = 0
    consecutive_stalls = 0
    
    # Precompute which elements are "expensive" (block many additions)
    # We'll track this dynamically
    
    while time.time() < deadline - 5 and iterations < max_iterations:
        iterations += 1
        
        if len(current_list) <= 1:
            break
            
        # Smart removal: pick element that blocks the most potential additions
        # Or just random with some bias
        if rng.random() < 0.3 and len(current_list) > 5:
            # Try to remove an element that's causing many conflicts
            # Pick random and check how many candidates it blocks
            idx_to_remove = rng.randint(0, len(current_list) - 1)
        else:
            idx_to_remove = rng.randint(0, len(current_list) - 1)
        
        u = current_list[idx_to_remove]
        
        # Save state
        saved_list = current_list[:]
        saved_set = set(current_set)
        saved_counts = sum_counts.copy()
        
        # Remove u
        remove_fast(u, current_list)
        current_set.discard(u)
        
        # Try to add two new elements - try multiple strategies
        candidates = [x for x in all_candidates if x not in current_set]
        rng.shuffle(candidates)
        
        success = False
        added_elements = []
        
        # Strategy 1: Greedy - add first feasible, then first feasible for second
        a_added = False
        a_val = None
        for x in candidates[:200]:  # Limit to first 200 for speed
            if can_add_fast(x, current_list):
                add_fast(x, current_list)
                current_set.add(x)
                added_elements.append(x)
                a_val = x
                a_added = True
                break
        
        if a_added:
            candidates_b = [x for x in candidates if x != a_val and x not in current_set]
            rng.shuffle(candidates_b)
            
            b_added = False
            for x in candidates_b[:200]:
                if can_add_fast(x, current_list):
                    add_fast(x, current_list)
                    current_set.add(x)
                    added_elements.append(x)
                    b_added = True
                    break
            
            if b_added:
                success = True
                best_size = len(current_list)
                stall_count = 0
                consecutive_stalls = 0
            else:
                # Revert
                current_list[:] = saved_list
                current_set = saved_set
                sum_counts[:] = saved_counts
                stall_count += 1
                consecutive_stalls += 1
        else:
            # Revert
            current_list[:] = saved_list
            current_set = saved_set
            sum_counts[:] = saved_counts
            stall_count += 1
            consecutive_stalls += 1
        
        if not success:
            if stall_count > 1000:
                # Try a different approach: 1-for-1 swap
                if consecutive_stalls > 500:
                    break
                # Try swapping: remove u, add one better element
                remove_fast(u, current_list)
                current_set.discard(u)
                
                # Find the best single element to add
                best_x = None
                best_score = -1
                for x in candidates[:100]:
                    if can_add_fast(x, current_list):
                        # Score by how many future additions it enables
                        # Simple heuristic: prefer smaller values
                        score = -x
                        if score > best_score:
                            best_score = score
                            best_x = x
                
                if best_x is not None:
                    add_fast(best_x, current_list)
                    current_set.add(best_x)
                    # Keep this swap even if size doesn't change
                    stall_count = 0
                else:
                    current_list[:] = saved_list
                    current_set = saved_set
                    sum_counts[:] = saved_counts
        
        # Periodic restart if stuck too long
        if stall_count > 3000:
            # Try a fresh greedy from a different seed
            if time.time() < deadline - 10:
                local_counts = np.zeros(max_sum + 1, dtype=np.int32)
                fresh_list = []
                fresh_rng = random.Random(int(time.time() * 1000) % 10000)
                fresh_candidates = list(range(1, N + 1))
                fresh_rng.shuffle(fresh_candidates)
                
                for x in fresh_candidates:
                    if local_counts[2 * x] >= 2:
                        continue
                    if fresh_list:
                        sums = x + np.array(fresh_list, dtype=np.int32)
                        if np.any(local_counts[sums] >= 2):
                            continue
                    local_counts[2 * x] += 1
                    if fresh_list:
                        sums = x + np.array(fresh_list, dtype=np.int32)
                        local_counts[sums] += 1
                    fresh_list.append(x)
                
                if len(fresh_list) > len(current_list):
                    sum_counts[:] = local_counts
                    current_list = fresh_list
                    current_set = set(fresh_list)
                    best_size = len(current_list)
                    stall_count = 0
                    consecutive_stalls = 0
    
    # Phase 3: Final polish - try to add any remaining feasible elements
    for x in sorted(set(range(1, N + 1)) - current_set):
        if time.time() > deadline - 2:
            break
        if can_add_fast(x, current_list):
            add_fast(x, current_list)
            current_set.add(x)
    
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
