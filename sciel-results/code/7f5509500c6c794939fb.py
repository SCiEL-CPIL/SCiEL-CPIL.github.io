import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    target_min = 128
    target_max = 256
    target_sums = range(target_min, target_max + 1)
    
    # We will represent A as a boolean array of size 129
    # and maintain the count array N of size 257
    
    def calculate_counts(A_mask):
        # A_mask is a numpy boolean array of size 129
        # Convolution
        # Y = np.convolve(A_mask, A_mask, mode='full')
        # But we only need [128, 256]
        # And we need to convert ordered to unordered counts.
        N = np.zeros(257, dtype=np.int32)
        # Efficient loop
        # A_indices = np.where(A_mask)[0]
        # for i_idx in range(len(A_indices)):
        #    for j_idx in range(i_idx, len(A_indices)):
        #        s = A_indices[i_idx] + A_indices[j_idx]
        #        if target_min <= s <= target_max:
        #            N[s] += 1
        # This is O(K^2). K~30 -> 900 ops. Fast enough for init.
        A_indices = np.where(A_mask)[0]
        for i_idx in range(len(A_indices)):
            a = A_indices[i_idx]
            for j_idx in range(i_idx, len(A_indices)):
                b = A_indices[j_idx]
                s = a + b
                if target_min <= s <= target_max:
                    N[s] += 1
        return N

    def get_objective(N):
        # Returns (missing, peak, sum_squares)
        target_N = N[target_min:target_max+1]
        missing = np.sum(target_N == 0)
        if missing > 0:
            return (missing, float('inf'), float('inf'))
        peak = np.max(target_N)
        sum_sq = np.sum(target_N**2)
        return (0, peak, sum_sq)

    # Greedy initialization
    def greedy_init():
        A_mask = np.zeros(129, dtype=bool)
        N = np.zeros(257, dtype=np.int32)
        
        # Start with some random elements to break symmetry?
        # Or just start empty and add one by one.
        # To cover [128, 256], we need elements that sum to these.
        # Let's try a random start and repair.
        
        # Random start
        k = random.randint(25, 35)
        init_set = set(random.sample(range(129), k))
        A_mask = np.zeros(129, dtype=bool)
        for x in init_set:
            A_mask[x] = True
        
        N = calculate_counts(A_mask)
        
        # Repair: add elements to cover missing sums
        while True:
            missing_sums = [s for s in target_sums if N[s] == 0]
            if not missing_sums:
                break
            
            # Find best element to add
            best_x = None
            best_score = -1
            best_new_N = None
            
            candidates = [x for x in range(129) if not A_mask[x]]
            # Shuffle to avoid bias
            random.shuffle(candidates)
            
            for x in candidates:
                # Simulate adding x
                new_N = N.copy()
                # Update counts with x
                A_indices = np.where(A_mask)[0]
                for a in A_indices:
                    s = x + a
                    if target_min <= s <= target_max:
                        new_N[s] += 1
                if 2*x in target_sums:
                    new_N[2*x] += 1
                
                # Score: number of missing sums covered
                covered = 0
                for s in missing_sums:
                    if new_N[s] > 0:
                        covered += 1
                # Secondary: minimize peak increase?
                # Just maximize coverage for now
                if covered > best_score:
                    best_score = covered
                    best_x = x
                    best_new_N = new_N
            
            if best_x is None or best_score == 0:
                # Stuck? Should not happen if we can add any element.
                # If we are stuck, break.
                break
                
            A_mask[best_x] = True
            N = best_new_N
            
        return A_mask, N

    best_peak = float('inf')
    best_A_mask = None
    best_sum_sq = float('inf')
    
    # Local Search
    def local_search(A_mask, N, time_limit):
        A_mask = A_mask.copy()
        N = N.copy()
        current_obj = get_objective(N)
        current_peak = current_obj[1]
        current_sum_sq = current_obj[2]
        
        if current_peak == float('inf'):
            return A_mask, N, current_obj
            
        A_indices = np.where(A_mask)[0]
        
        while time.time() < time_limit:
            improved = False
            # Try random swaps
            # Number of swaps to try per iteration
            num_swaps = 100
            
            for _ in range(num_swaps):
                if time.time() > time_limit:
                    break
                
                # Pick random a in A
                a_idx = random.randint(0, len(A_indices)-1)
                a = A_indices[a_idx]
                
                # Pick random b not in A
                # Efficient way to pick b:
                # Generate random b, check if in A.
                b = random.randint(0, 128)
                if A_mask[b]:
                    continue
                    
                # Simulate swap: remove a, add b
                # Update N
                new_N = N.copy()
                
                # Remove a
                # Decrement counts for pairs involving a
                # Pairs (a, a) and (a, u) for u in A, u != a
                # Note: A_indices includes a.
                # We need to iterate over A \ {a}
                
                # Optimization: Pre-compute neighbors?
                # For now, just iterate.
                for u_idx in range(len(A_indices)):
                    u = A_indices[u_idx]
                    if u == a:
                        continue
                    s = a + u
                    if target_min <= s <= target_max:
                        new_N[s] -= 1
                s = 2*a
                if target_min <= s <= target_max:
                    new_N[s] -= 1
                    
                # Add b
                # Increment counts for pairs involving b
                # Pairs (b, b) and (b, u) for u in A (original A, so includes a? No, A \ {a} + {b})
                # The new set is (A \ {a}) U {b}
                # So pairs are (b, b) and (b, u) for u in (A \ {a})
                
                for u_idx in range(len(A_indices)):
                    u = A_indices[u_idx]
                    if u == a:
                        continue
                    s = b + u
                    if target_min <= s <= target_max:
                        new_N[s] += 1
                s = 2*b
                if target_min <= s <= target_max:
                    new_N[s] += 1

                # Evaluate new state
                new_obj = get_objective(new_N)
                new_peak = new_obj[1]
                new_sum_sq = new_obj[2]
                
                if new_peak < current_peak:
                    # Accept
                    A_mask[a] = False
                    A_mask[b] = True
                    # Update A_indices
                    A_indices[a_idx] = b
                    # Keep A_indices sorted? Not strictly necessary for logic, but good for determinism?
                    # We just swap values.
                    N = new_N
                    current_peak = new_peak
                    current_sum_sq = new_sum_sq
                    improved = True
                    break # Restart search with new state
                    
                elif new_peak == current_peak:
                    if new_sum_sq < current_sum_sq:
                        # Accept if it balances the counts
                        A_mask[a] = False
                        A_mask[b] = True
                        A_indices[a_idx] = b
                        N = new_N
                        current_sum_sq = new_sum_sq
                        improved = True
                        # Do not break, keep searching for peak reduction
                        # Or break to stabilize? Let's not break, might find peak reduction.
                        
            if not improved:
                # Try a larger perturbation? Or break?
                # If stuck, break to try new init
                break
                
        return A_mask, N, (0, current_peak, current_sum_sq)

    # Main Loop
    for i in range(50):
        if time.time() > deadline:
            break
            
        # Greedy Init
        A_mask, N = greedy_init()
        
        # Check if valid
        obj = get_objective(N)
        if obj[0] > 0:
            continue
            
        peak = obj[1]
        sum_sq = obj[2]
        
        if peak < best_peak or (peak == best_peak and sum_sq < best_sum_sq):
            best_peak = peak
            best_sum_sq = sum_sq
            best_A_mask = A_mask.copy()
            
        # Local Search
        time_left = deadline - time.time()
        if time_left > 1:
            A_opt, N_opt, obj_opt = local_search(A_mask, N, time_left - 1)
            if obj_opt[0] == 0:
                if obj_opt[1] < best_peak or (obj_opt[1] == best_peak and obj_opt[2] < best_sum_sq):
                    best_peak = obj_opt[1]
                    best_sum_sq = obj_opt[2]
                    best_A_mask = A_opt.copy()
                    
        if best_peak <= 4:
             # We found a very good solution.
             # Can we get 3?
             # If we have time, try to force 3 with more aggressive search?
             # For now, just continue to see if we can improve sum_sq or find 3.
             pass

    if best_A_mask is None:
        # Fallback
        return list(range(64, 129))
        
    # Convert mask to list
    A_list = [i for i in range(129) if best_A_mask[i]]
    return A_list
