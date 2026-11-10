import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    target_min = 128
    target_max = 256
    num_targets = 129
    
    best_A = None
    best_max_count = float('inf')
    
    # Helper to compute counts from A
    def calc_counts(A_set):
        counts = np.zeros(num_targets, dtype=np.int32)
        A_list = list(A_set)
        # Optimization: only consider pairs summing to [128, 256]
        # a+b >= 128 and a+b <= 256
        # Since a,b <= 128, a+b <= 256 is always true.
        # So only need a+b >= 128.
        for i in range(len(A_list)):
            a = A_list[i]
            # b must be >= 128 - a
            # And b >= a (for unordered)
            # So b in [max(a, 128-a), 128]
            min_b = max(a, 128 - a)
            if min_b <= 128:
                # Count how many b in A_list are in [min_b, 128]
                # This is O(k) per a. Total O(k^2).
                # For k=50, 2500 checks.
                # Vectorized:
                pass
        # Vectorized approach for calc_counts:
        # Generate all pairs? No, k^2 is small.
        # Just use nested loops in Python? 2500 iterations is fine.
        # Or use numpy broadcasting?
        # A_arr = np.array(A_list)
        # Sums = A_arr[:, None] + A_arr[None, :]
        # Mask = (Sums >= 128) & (Sums <= 256) & (A_arr[:, None] <= A_arr[None, :])
        # Counts = np.bincount(Sums[Mask] - 128, minlength=129)
        # This is very fast.
        
        A_arr = np.array(list(A_set))
        if len(A_arr) == 0:
            return counts
        Sums = A_arr[:, None] + A_arr[None, :]
        # Unordered: a <= b
        Mask = (Sums >= 128) & (Sums <= 256) & (A_arr[:, None] <= A_arr[None, :])
        valid_sums = Sums[Mask]
        if len(valid_sums) > 0:
            counts = np.bincount(valid_sums - 128, minlength=129).astype(np.int32)
        return counts

    # SA Loop
    for restart in range(10):
        if time.time() > deadline - 5: break
        
        # Init
        k = 50
        A = set(np.random.choice(n+1, k, replace=False))
        
        # Ensure validity by adding random elements
        counts = calc_counts(A)
        while np.any(counts < 1) and len(A) < 129:
            # Find missing sums
            missing = np.where(counts < 1)[0] + 128
            # Try to add a random element
            x = np.random.randint(0, n+1)
            if x not in A:
                A.add(x)
                counts = calc_counts(A) # Recalculate. O(k^2). k grows.
                # If k becomes large, this is slow.
                # Better: update counts incrementally.
                pass
        
        # If still invalid, skip
        if np.any(counts < 1):
            continue
            
        curr_max = np.max(counts)
        if curr_max < best_max_count:
            best_max_count = curr_max
            best_A = set(A)
            
        # SA
        T = 2.0
        cooling = 0.999
        curr_A = set(A)
        curr_counts = counts.copy()
        curr_max = curr_max
        curr_zeros = np.sum(curr_counts < 1)
        
        steps = 0
        while steps < 50000 and time.time() < deadline - 2:
            steps += 1
            x = np.random.randint(0, n+1)
            
            if x in curr_A:
                # Remove x
                # Update counts
                # Pairs {x, y} for y in curr_A \ {x}
                # And {x, x}
                
                # Vectorized update
                # Get y in curr_A, y != x
                y_arr = np.array([y for y in curr_A if y != x])
                if len(y_arr) > 0:
                    sums = x + y_arr
                    mask = (sums >= 128) & (sums <= 256)
                    if np.any(mask):
                        curr_counts[sums[mask] - 128] -= 1
                
                s_xx = 2 * x
                if 128 <= s_xx <= 256:
                    curr_counts[s_xx - 128] -= 1
                    
                curr_A.remove(x)
                
            else:
                # Add x
                # Pairs {x, y} for y in curr_A
                y_arr = np.array(list(curr_A))
                if len(y_arr) > 0:
                    sums = x + y_arr
                    mask = (sums >= 128) & (sums <= 256)
                    if np.any(mask):
                        curr_counts[sums[mask] - 128] += 1
                
                s_xx = 2 * x
                if 128 <= s_xx <= 256:
                    curr_counts[s_xx - 128] += 1
                    
                curr_A.add(x)
            
            # Evaluate
            new_max = np.max(curr_counts)
            new_zeros = np.sum(curr_counts < 1)
            
            cost_new = new_max + 100 * new_zeros
            cost_old = curr_max + 100 * curr_zeros
            
            if cost_new < cost_old:
                curr_max = new_max
                curr_zeros = new_zeros
            else:
                # Accept with probability
                delta = cost_new - cost_old
                if np.random.random() < np.exp(-delta / T):
                    curr_max = new_max
                    curr_zeros = new_zeros
                else:
                    # Revert
                    if x in curr_A:
                        # We added x, now remove it
                        y_arr = np.array([y for y in curr_A if y != x])
                        if len(y_arr) > 0:
                            sums = x + y_arr
                            mask = (sums >= 128) & (sums <= 256)
                            if np.any(mask):
                                curr_counts[sums[mask] - 128] -= 1
                        s_xx = 2 * x
                        if 128 <= s_xx <= 256:
                            curr_counts[s_xx - 128] -= 1
                        curr_A.remove(x)
                    else:
                        # We removed x, now add it
                        y_arr = np.array(list(curr_A))
                        if len(y_arr) > 0:
                            sums = x + y_arr
                            mask = (sums >= 128) & (sums <= 256)
                            if np.any(mask):
                                curr_counts[sums[mask] - 128] += 1
                        s_xx = 2 * x
                        if 128 <= s_xx <= 256:
                            curr_counts[s_xx - 128] += 1
                        curr_A.add(x)
                        
            T *= cooling
            
            # Track best
            if new_zeros == 0 and new_max < best_max_count:
                best_max_count = new_max
                best_A = set(curr_A)

    if best_A is None:
        return list(range(64, 129))
    return list(best_A)
