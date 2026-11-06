import random
import time
import numpy as np

N = 64
GRID_SIZE = N * N

# Pre-compute all points
points_arr = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
ALL_POINTS = [tuple(p) for p in points_arr]

# Pre-compute squared distances
xs = points_arr[:, 0].astype(np.int64)
ys = points_arr[:, 1].astype(np.int64)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
DIST = (dx * dx + dy * dy)  # shape (GRID_SIZE, GRID_SIZE)

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # We will work with a binary mask `mask` of size GRID_SIZE
    # where mask[i] = 1 if point i is in the set.
    # Target size K. We start with a random set of size K and try to optimize.
    # If we can find a set of size K with 0 conflicts, we can try K+1, etc.
    
    # However, dynamic size is tricky in vectorized local search.
    # Better strategy: 
    # 1. Find a large valid set using a fast vectorized greedy or SA.
    # 2. The "score" is the size.
    
    # Let's use a Simulated Annealing approach on a fixed size K.
    # We will iterate K from small to large? No, that's slow.
    # Instead, we start with a reasonably large K (e.g., 100) and try to find a valid config.
    # If we find one, we try K+1.
    # To be efficient, we'll run a long SA for a fixed K, and if it reaches 0 conflicts, 
    # we increment K and restart (or continue with the new size).
    
    # Actually, a better structural change:
    # Use the vectorized conflict count to guide a "swap" local search.
    # We maintain a set of indices `current_indices`.
    # We compute the conflict count of this set.
    # We try to swap one point out and one point in to reduce conflicts.
    
    # Helper: Compute conflict count for a given set of indices
    # Conflict exists if there are i, j, k in set such that d(i,j) == d(i,k) or d(i,j) == d(j,k) etc.
    # Actually, the definition is: no three points form an isosceles triangle.
    # This means for any distinct i, j, k in the set:
    # d(i,j) != d(i,k) and d(i,j) != d(j,k) and d(j,k) != d(i,k) is NOT required.
    # Wait. An isosceles triangle has at least two equal sides.
    # So we need: for all distinct i, j, k in S:
    # d(i,j) != d(i,k) AND d(i,j) != d(j,k) AND d(j,k) != d(i,k)
    # This is equivalent to: for each apex i, all distances d(i, j) for j in S\{i} must be distinct.
    
    # So, for a set S, the number of isosceles triangles is the sum over all i in S of:
    # (number of pairs j, k in S\{i} such that d(i,j) == d(i,k))
    
    # Let's vectorize this check.
    # Given a mask `mask` (boolean array of length GRID_SIZE), we want to count conflicts.
    # This is still expensive to do from scratch every step.
    
    # Alternative Vectorized Strategy:
    # Maintain `current_indices` (list of ints).
    # Pre-compute `dist_from_i` for all i? No, too much memory.
    
    # Let's stick to the swap idea but make the validity check fast.
    # The bottleneck in the previous program was the Python loop in `is_valid_addition`.
    # If we can make the "cost" calculation fast, we can do many more swaps.
    
    # Cost Calculation for a set S:
    # For each i in S, count duplicate distances to other points in S.
    # If we represent S as a numpy array of indices `S_idx`:
    # D_sub = DIST[np.ix_(S_idx, S_idx)]  # |S|x|S| matrix
    # For each row i, count how many values are repeated.
    # This is O(|S|^2) but in C (NumPy), which is much faster than Python loops.
    
    def count_conflicts(S_idx):
        m = len(S_idx)
        if m < 3:
            return 0
        D_sub = DIST[S_idx][:, S_idx]
        conflicts = 0
        for i in range(m):
            # Exclude self (0 distance)
            dists = D_sub[i, :].copy()
            dists[i] = -1 # Mark self
            # Count duplicates
            # Sort and count runs
            sorted_dists = np.sort(dists)
            # A duplicate exists if sorted_dists[j] == sorted_dists[j+1]
            # We count the number of pairs (j, k) with j<k and dists[j]==dists[k]
            # This is sum over unique values v of C(count(v), 2)
            # For speed, we can just check if any duplicate exists? 
            # No, we need the count for SA.
            # But for a "valid" check, we just need to know if count > 0.
            # For SA, we need the total count.
            
            # Efficient duplicate count:
            # np.unique with return_counts
            vals, counts = np.unique(dists, return_counts=True)
            # Sum of c*(c-1)/2 for c in counts
            conflicts += np.sum(counts * (counts - 1) // 2)
        return conflicts
    
    # This is still O(m^2 log m) per evaluation. If m=100, 100*100*log(100) ~ 7000 ops.
    # In NumPy, this is very fast.
    
    # Strategy:
    # 1. Start with a random set of size K.
    # 2. Run SA:
    #    - Pick random i in S, random j not in S.
    #    - Compute new_conflicts = count_conflicts(S - i + j)
    #    - Compute old_conflicts (cached)
    #    - If new < old or random() < exp((old-new)/T), accept.
    # 3. If conflicts == 0, we have a valid set of size K. Try K+1.
    
    # To speed up `count_conflicts`, we can update incrementally?
    # Removing i and adding j changes distances for all other points.
    # It's complex. Let's rely on NumPy speed.
    
    best_size = 0
    best_set_indices = []
    
    # Start with a small K and grow
    K = 10
    current_indices = np.random.choice(GRID_SIZE, size=K, replace=False)
    current_conflicts = count_conflicts(current_indices)
    
    # If we start with conflicts, we need to reduce them.
    # SA parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # We will run multiple "rounds" for increasing K
    # But we must be careful with time.
    
    while time.time() < deadline:
        # Try to solve for current K
        # If we find a solution, increment K
        
        # If we haven't found a solution for current K in a while, maybe K is too big?
        # No, we start small.
        
        # Run SA for a fixed amount of time or steps
        sa_start = time.time()
        steps_this_round = 0
        max_steps_per_round = 20000  # Adjust based on speed
        
        while steps_this_round < max_steps_per_round and time.time() < deadline - 10:
            steps_this_round += 1
            m = len(current_indices)
            
            # Pick i to remove
            i_idx = random.randint(0, m - 1)
            i_val = current_indices[i_idx]
            
            # Pick j to add
            # To avoid picking a point already in S, we can pick randomly and check
            # Or maintain a set of available points.
            # For simplicity, pick random j, if in S, retry (unlikely for large grid)
            j_val = random.randint(0, GRID_SIZE - 1)
            if j_val == i_val:
                continue
            if j_val in current_indices:
                # Check if j is in the list
                # Since m is small, linear search is ok, but let's use a set for O(1)
                # We'll maintain a set `S_set` alongside `current_indices`
                pass # We need to manage state carefully.
            
            # Let's restructure state management for speed
            # Use a list `S_list` and a set `S_set`.
            
            # This inner loop is getting messy. Let's write a cleaner SA loop.
            
            # Break out of inner loop to restructure
            break
            
        # Restructured SA Loop
        # We need to track the current set efficiently.
        
        # Let's restart the logic for the main loop to be cleaner.
        break
        
    # --- CLEAN IMPLEMENTATION ---
    
    # State:
    S_list = []
    S_set = set()
    
    def add_point(idx):
        S_list.append(idx)
        S_set.add(idx)
        
    def remove_point(idx):
        S_list.remove(idx) # O(N)
        S_set.remove(idx)
        
    # We'll use a different approach: 
    # Since `count_conflicts` is the bottleneck, let's optimize it further.
    # Actually, for K ~ 50-100, `count_conflicts` takes ~1ms in NumPy.
    # So we can do ~1000 steps/second. In 280s, we can do ~280,000 steps.
    # This is enough to find a valid set if one exists.
    
    # Let's try to find the maximum K.
    # We'll use a "hill climbing" with random restarts if stuck.
    
    best_valid_size = 0
    best_valid_set = []
    
    # Start with K=1
    current_K = 1
    # Initialize with a random point
    current_indices = [random.randint(0, GRID_SIZE - 1)]
    
    # We will iteratively increase K.
    # For each K, we run SA to minimize conflicts.
    # If conflicts reach 0, we accept K and try K+1.
    # If we fail to reach 0 after a time budget, we try a smaller K? 
    # No, if K is too big, no solution might exist. 
    # But we want the max. 
    # Strategy: 
    # 1. Try K=10, 20, 30... 
    # 2. Use binary search on K? 
    # 3. Or just increment K by 1 and if stuck, backtrack?
    
    # Let's try a simpler heuristic:
    # Start with K=50. Try to fix it. If it works, try 60, 70...
    # If it fails, try 40, 30...
    
    # To be safe and robust:
    # We'll run a global SA that allows size changes? 
    # No, "size" is the score. We want max size.
    
    # Let's stick to: Find max K such that a valid set of size K exists.
    # We'll probe K values.
    
    # Probe list: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    # We'll sort them and try to find the highest one that works.
    # But "works" is probabilistic.
    
    # Better: 
    # Start with K=1. 
    # Greedily add points? No, we established that's slow.
    
    # Let's use the vectorized conflict count for a "Swap" local search on a fixed size K.
    # We'll try K = 100. 
    # If we find a valid set, great.
    # If not, we try K = 90, 80...
    
    # To save time, we'll allocate time budget per K attempt.
    
    k_candidates = [120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    
    best_result_size = 0
    best_result_indices = []
    
    for K in k_candidates:
        if time.time() > deadline - 20:
            break
            
        # Try to find a valid set of size K
        # Initialize randomly
        curr_indices = np.random.choice(GRID_SIZE, size=K, replace=False)
        curr_conflicts = count_conflicts(curr_indices)
        
        if curr_conflicts == 0:
            best_result_size = K
            best_result_indices = curr_indices.tolist()
            continue # Try larger K
        
        # Run SA
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        max_steps = 50000 # Adjusted
        
        # We need a fast "swap" evaluation.
        # We will maintain `curr_indices` as a list for easy modification.
        # `count_conflicts` takes a numpy array.
        
        for step in range(max_steps):
            if time.time() > deadline - 20:
                break
                
            if curr_conflicts == 0:
                best_result_size = K
                best_result_indices = curr_indices.tolist()
                break # Try larger K
                
            # Pick i to remove
            i_idx = random.randint(0, K - 1)
            i_val = curr_indices[i_idx]
            
            # Pick j to add
            # Random j
            j_val = random.randint(0, GRID_SIZE - 1)
            
            # Check if j is already in set
            # To make this O(1), we should use a set.
            # But converting list to set every step is O(K).
            # Let's maintain a set.
            
            # This is getting complicated with state management in the loop.
            # Let's use a class or just careful variable handling.
            
            # Optimization: 
            # Instead of full recount, can we update?
            # No, let's rely on NumPy speed.
            
            # To speed up the "is j in S" check, we'll use a boolean array `in_S`
            in_S = np.zeros(GRID_SIZE, dtype=bool)
            in_S[curr_indices] = True
            
            if in_S[j_val]:
                continue
                
            # Perform swap
            new_indices = curr_indices.copy()
            new_indices[i_idx] = j_val
            
            new_conflicts = count_conflicts(new_indices)
            
            # SA Acceptance
            delta = new_conflicts - curr_conflicts
            if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
                curr_indices = new_indices
                curr_conflicts = new_conflicts
                in_S = np.zeros(GRID_SIZE, dtype=bool)
                in_S[curr_indices] = True
            else:
                # Reject
                pass
                
            T *= cooling
            
        # After SA, if we found a valid set, update best
        if curr_conflicts == 0:
            if K > best_result_size:
                best_result_size = K
                best_result_indices = curr_indices.tolist()
            # If we found a valid set for K, we can try larger K immediately
            # But the loop structure tries larger K first.
            # So if we succeed here, we should break and let the outer loop try larger?
            # No, the outer loop iterates k_candidates in descending order.
            # So if K=120 fails, we try K=110.
            # If K=110 succeeds, we record it.
            # We should continue to see if K=120 was a fluke? No, 120 was tried and failed.
            # So we can stop? No, maybe 120 failed due to bad luck, but 110 worked.
            # We want the MAX. So if 110 works, we have a valid 110.
            # But maybe 120 is possible. 
            # Since we tried 120 first and failed, we assume 120 is hard/impossible.
            # We'll keep the best found so far.
            
    # Return the best set found
    return [ALL_POINTS[i] for i in best_result_indices]

# Wait, the above code has a flaw: `in_S` is recreated every step inside the SA loop.
# That's O(N) per step. 280000 steps * 4096 size = too slow?
# 280000 * 4096 ~ 1e9 ops. Might be too slow in Python/NumPy overhead.
# We need to maintain `in_S` incrementally.

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    N = 64
    GRID_SIZE = N * N
    points_arr = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    ALL_POINTS = [tuple(p) for p in points_arr]
    
    xs = points_arr[:, 0].astype(np.int64)
    ys = points_arr[:, 1].astype(np.int64)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    DIST = (dx * dx + dy * dy)
    
    def count_conflicts(S_idx):
        m = len(S_idx)
        if m < 3:
            return 0
        D_sub = DIST[S_idx][:, S_idx]
        conflicts = 0
        for i in range(m):
            dists = D_sub[i, :].copy()
            dists[i] = -1
            vals, counts = np.unique(dists, return_counts=True)
            conflicts += np.sum(counts * (counts - 1) // 2)
        return conflicts

    best_result_size = 0
    best_result_indices = []
    
    # Try sizes in descending order
    # We'll allocate time based on remaining time.
    
    k_candidates = [150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    
    for K in k_candidates:
        if time.time() > deadline - 15:
            break
            
        # Time budget for this K
        # If K is large, give it more time? Or less?
        # Let's give each K a fixed number of steps, but capped by time.
        
        # Initialize
        curr_indices = np.random.choice(GRID_SIZE, size=K, replace=False)
        curr_conflicts = count_conflicts(curr_indices)
        
        if curr_conflicts == 0:
            if K > best_result_size:
                best_result_size = K
                best_result_indices = curr_indices.tolist()
            continue
        
        # SA Loop
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        steps = 0
        max_steps = 100000
        
        # Maintain in_S array
        in_S = np.zeros(GRID_SIZE, dtype=bool)
        in_S[curr_indices] = True
        
        while steps < max_steps:
            if time.time() > deadline - 15:
                break
                
            if curr_conflicts == 0:
                if K > best_result_size:
                    best_result_size = K
                    best_result_indices = curr_indices.tolist()
                break
                
            steps += 1
            
            # Pick i to remove
            i_idx = random.randint(0, K - 1)
            i_val = curr_indices[i_idx]
            
            # Pick j to add
            # To avoid bias, we can pick from a random subset of available points?
            # Or just random.
            j_val = random.randint(0, GRID_SIZE - 1)
            
            if in_S[j_val]:
                continue
                
            # Swap
            # Update in_S
            in_S[i_val] = False
            in_S[j_val] = True
            
            # Update indices
            new_indices = curr_indices.copy()
            new_indices[i_idx] = j_val
            
            new_conflicts = count_conflicts(new_indices)
            
            delta = new_conflicts - curr_conflicts
            
            if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
                curr_indices = new_indices
                curr_conflicts = new_conflicts
                # in_S is already updated
            else:
                # Reject: revert
                in_S[i_val] = True
                in_S[j_val] = False
                # curr_indices unchanged
                
            T *= cooling
            
        # End SA for this K
        
    return [ALL_POINTS[i] for i in best_result_indices]
