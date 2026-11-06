import time
import numpy as np

def construct():
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    n_max = 4096
    max_sum = 3 * n_max + 1
    
    def try_greedy(candidate_order, deadline):
        """Try to build a valid set by adding elements in the given order."""
        bitset = np.zeros(max_sum + 1, dtype=np.uint8)
        selected = []
        
        for x in candidate_order:
            if time.time() > deadline:
                break
            if len(selected) >= 80:
                break
            if x > n_max:
                continue
                
            sel = np.array(selected, dtype=np.int32)
            n = len(sel)
            
            # Generate all new triple sums involving x
            total_size = 1 + n + n + n * (n - 1) // 2
            if total_size == 0:
                continue
                
            new_sums = np.empty(total_size, dtype=np.int32)
            
            idx = 0
            new_sums[idx] = 3 * x
            idx += 1
            
            new_sums[idx:idx + n] = 2 * x + sel
            idx += n
            
            new_sums[idx:idx + n] = x + 2 * sel
            idx += n
            
            if n >= 2:
                ii, jj = np.triu_indices(n, k=1)
                new_sums[idx:] = x + sel[ii] + sel[jj]
            
            # Check for internal duplicates
            sorted_sums = np.sort(new_sums)
            if np.any(sorted_sums[1:] == sorted_sums[:-1]):
                continue
                
            # Check against existing sums
            if np.any(bitset[new_sums]):
                continue
                
            # Valid - add x
            selected.append(x)
            bitset[new_sums] = 1
        
        return selected
    
    # Strategy 1: Start with powers of 4 skeleton, then greedily add in increasing order
    skeleton = [1, 4, 16, 64, 256, 1024, 4096]
    skeleton_set = set(skeleton)
    
    # Build candidate order: skeleton first, then remaining in increasing order
    candidates_1 = list(skeleton) + [x for x in range(1, n_max + 1) if x not in skeleton_set]
    result1 = try_greedy(candidates_1, deadline)
    
    # Strategy 2: Randomized greedy - try multiple random orderings
    best_result = result1
    best_size = len(best_result)
    
    # Reserve some time for multiple randomized attempts
    time_per_attempt = 5.0
    remaining_time = deadline - time.time()
    num_attempts = max(1, int(remaining_time / time_per_attempt))
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Create a random candidate order
        # Mix of small numbers (more likely to fit) and random numbers
        small_nums = list(range(1, 200))
        random_nums = np.random.choice(range(200, n_max + 1), size=min(300, n_max - 200), replace=False)
        
        # Shuffle both groups and interleave
        np.random.shuffle(small_nums)
        random_list = list(random_nums)
        np.random.shuffle(random_list)
        
        # Interleave: take 1 from small, 2 from random, etc.
        candidate_order = []
        si, ri = 0, 0
        while si < len(small_nums) or ri < len(random_list):
            if si < len(small_nums):
                candidate_order.append(small_nums[si])
                si += 1
            if ri < len(random_list):
                candidate_order.append(random_list[ri])
                ri += 1
            if ri < len(random_list):
                candidate_order.append(random_list[ri])
                ri += 1
        
        # Add any remaining
        while si < len(small_nums):
            candidate_order.append(small_nums[si])
            si += 1
        while ri < len(random_list):
            candidate_order.append(random_list[ri])
            ri += 1
        
        # Remove duplicates while preserving order
        seen = set()
        unique_order = []
        for x in candidate_order:
            if x not in seen:
                seen.add(x)
                unique_order.append(x)
        
        result = try_greedy(unique_order, deadline)
        
        if len(result) > best_size:
            best_result = result
            best_size = len(best_result)
        
        # If we've hit the max, stop early
        if best_size >= 80:
            break
    
    return best_result
