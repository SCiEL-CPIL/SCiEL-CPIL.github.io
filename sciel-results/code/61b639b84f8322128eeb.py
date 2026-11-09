import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    N_LOW = 128
    N_HIGH = 256
    MAX_VAL = 128
    
    # Precompute valid sums range
    # sums are in [0, 256]
    # We care about [128, 256]
    
    def calc_reps_from_set(A_set):
        """Calculate reps for a given set A. Returns array of length 257."""
        reps = np.zeros(257, dtype=np.int32)
        A_list = list(A_set)
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            for j in range(i, n):
                b = A_list[j]
                s = a + b
                reps[s] += 1
        return reps

    def get_objective(reps):
        """Calculate objective: penalty for uncovered + peak."""
        target_reps = reps[N_LOW:N_HIGH+1]
        uncovered = np.sum(target_reps == 0)
        peak = np.max(target_reps) if len(target_reps) > 0 else 0
        return uncovered * 10000 + int(peak)

    def is_valid(reps):
        return np.all(reps[N_LOW:N_HIGH+1] > 0)

    def get_peak(reps):
        return int(np.max(reps[N_LOW:N_HIGH+1]))

    best_A = None
    best_obj = float('inf')
    best_peak = float('inf')
    
    # Initialize with a reasonable random set
    # Size estimation: 
    # We need enough pairs to cover 129 sums.
    # If size K, pairs ~ K^2/2.
    # K=30 -> 450 pairs. Avg 3.5 per sum.
    # K=40 -> 800 pairs. Avg 6.2 per sum.
    # Let's try K in [35, 55]
    
    random.seed(42)
    
    # Phase 1: Find a valid set with low peak using SA
    # We will run multiple restarts with different initial sizes
    
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        # Choose initial size
        init_size = 35 + restart * 2
        if init_size > 60:
            init_size = 60
            
        # Initialize A
        A = set(random.sample(range(0, 129), init_size))
        # Ensure 0 and 128 are likely included to help extremes?
        # 128 is needed for 256. 0 is good for 128.
        # Let's force 128 in.
        A.add(128)
        # If size exceeded, remove a random one
        if len(A) > init_size:
            A = set(random.sample(list(A), init_size))
        
        reps = calc_reps_from_set(A)
        curr_obj = get_objective(reps)
        
        if curr_obj < best_obj:
            best_obj = curr_obj
            best_A = set(A)
            if is_valid(reps):
                best_peak = get_peak(reps)
        
        T = 5.0
        cooling = 0.999
        min_T = 0.01
        
        # Local search / SA loop
        while time.time() < deadline - 10 and T > min_T:
            # Select move
            move_type = random.random()
            
            if len(A) == 0:
                # Add a random element
                v = random.randint(0, 128)
                new_A = set(A)
                new_A.add(v)
                new_reps = calc_reps_from_set(new_A)
                new_obj = get_objective(new_reps)
                
                if new_obj < curr_obj or random.random() < np.exp(-(new_obj - curr_obj) / T):
                    A = new_A
                    reps = new_reps
                    curr_obj = new_obj
                    if curr_obj < best_obj:
                        best_obj = curr_obj
                        best_A = set(A)
                        if is_valid(reps):
                            best_peak = get_peak(reps)
            else:
                # Choose u to remove
                u = random.choice(list(A))
                
                # Choose v to add (or just remove)
                if move_type < 0.8:
                    # Swap
                    # Choose v not in A
                    # Optimization: pick v from a small candidate pool or random
                    # To ensure we explore well, pick random
                    v = random.randint(0, 128)
                    if v == u:
                        T *= cooling
                        continue
                    if v in A:
                        # If v is in A, this is effectively just removing u (if we don't re-add) 
                        # But swap implies v not in A. If v in A, we can't swap.
                        # Let's just try removing u.
                        new_A = set(A)
                        new_A.remove(u)
                        new_reps = calc_reps_from_set(new_A)
                        new_obj = get_objective(new_reps)
                        
                        if new_obj < curr_obj or random.random() < np.exp(-(new_obj - curr_obj) / T):
                            A = new_A
                            reps = new_reps
                            curr_obj = new_obj
                            if curr_obj < best_obj:
                                best_obj = curr_obj
                                best_A = set(A)
                                if is_valid(reps):
                                    best_peak = get_peak(reps)
                    else:
                        # Perform swap u -> v
                        # Update reps incrementally for speed?
                        # For simplicity in this rewrite, we recalculate if set is small.
                        # |A| ~ 40. Recalc is O(1600). 100k steps = 1.6e8 ops. Might be slow in Python.
                        # Let's implement incremental update.
                        
                        # Remove u
                        # For each w in A (excluding u): reps[u+w] -= 1
                        # reps[2u] -= 1 (if u was in A, pair (u,u) existed)
                        
                        # We need to be careful with the set A during update.
                        # Let's just recalculate for correctness first, then optimize if needed.
                        # Given 60s limit, and Python speed ~ 10^7 ops/s, 
                        # 1600 ops per step -> 6000 steps/s. 
                        # 60s -> 360,000 steps. This is reasonable.
                        
                        new_A = set(A)
                        new_A.remove(u)
                        new_A.add(v)
                        
                        new_reps = calc_reps_from_set(new_A)
                        new_obj = get_objective(new_reps)
                        
                        if new_obj < curr_obj or random.random() < np.exp(-(new_obj - curr_obj) / T):
                            A = new_A
                            reps = new_reps
                            curr_obj = new_obj
                            if curr_obj < best_obj:
                                best_obj = curr_obj
                                best_A = set(A)
                                if is_valid(reps):
                                    best_peak = get_peak(reps)
                else:
                    # Just remove u
                    new_A = set(A)
                    new_A.remove(u)
                    new_reps = calc_reps_from_set(new_A)
                    new_obj = get_objective(new_reps)
                    
                    if new_obj < curr_obj or random.random() < np.exp(-(new_obj - curr_obj) / T):
                        A = new_A
                        reps = new_reps
                        curr_obj = new_obj
                        if curr_obj < best_obj:
                            best_obj = curr_obj
                            best_A = set(A)
                            if is_valid(reps):
                                best_peak = get_peak(reps)
            
            T *= cooling
    
    # Phase 2: Polish the best valid set found
    if best_A is not None:
        # Try to reduce peak further
        # Local search on peak only
        A = set(best_A)
        reps = calc_reps_from_set(A)
        curr_peak = get_peak(reps)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            peak_sums = np.where(reps[N_LOW:N_HIGH+1] == curr_peak)[0] + N_LOW
            
            # Try removing elements that contribute to peak sums
            # Or swapping them
            for s in peak_sums:
                # Find pairs (a,b) in A such that a+b=s
                # Try removing one of them
                # We need to ensure coverage is not broken.
                
                # To speed up, we can just try random swaps on A
                # But let's do a targeted search.
                break # Simplified: just do random swaps in a tight loop
            
            # Random swaps to escape local minima in peak
            for _ in range(1000):
                if time.time() > deadline - 2:
                    break
                u = random.choice(list(A))
                v = random.randint(0, 128)
                if v == u or v in A:
                    continue
                
                new_A = set(A)
                new_A.remove(u)
                new_A.add(v)
                
                new_reps = calc_reps_from_set(new_A)
                if not is_valid(new_reps):
                    continue
                
                new_peak = get_peak(new_reps)
                if new_peak < curr_peak:
                    A = new_A
                    reps = new_reps
                    curr_peak = new_peak
                    improved = True
                    best_peak = curr_peak
                    best_A = set(A)
                    break
                elif new_peak == curr_peak:
                    # Accept to move around
                    if random.random() < 0.5:
                        A = new_A
                        reps = new_reps

    if best_A is None:
        # Fallback
        return list(range(0, 129, 2))
        
    return list(best_A)
