import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_S = []
    best_size = 0
    
    # We will run multiple random restarts
    # The number of restarts depends on time.
    # Each restart: Greedy + Local Search
    
    for restart in range(50): # Cap restarts, but break on time
        if time.time() > deadline - 15.0:
            break
            
        # Phase 1: Greedy Initialization
        S = np.array([], dtype=int)
        S_set = set()
        counts = np.zeros(2001, dtype=int)
        
        # Helper to add element
        def try_add(x, S_arr, S_set, counts):
            if x in S_set:
                return False, S_arr, S_set, counts
            
            if len(S_arr) > 0:
                # Sums x + s for s in S_arr
                # We need to check if any of these sums already have 2 representations
                if np.any(counts[x + S_arr] >= 2):
                    return False, S_arr, S_set, counts
            
            if counts[2*x] >= 2:
                return False, S_arr, S_set, counts
            
            # Update counts
            if len(S_arr) > 0:
                counts[x + S_arr] += 1
            counts[2*x] += 1
            
            # Update S
            S_new = np.insert(S_arr, np.searchsorted(S_arr, x), x)
            S_set.add(x)
            
            return True, S_new, S_set, counts

        def try_remove(x, S_arr, S_set, counts):
            if x not in S_set:
                return False, S_arr, S_set, counts
            
            # Update counts
            # We need to decrement counts for sums formed by x and other elements in S_arr
            # S_arr contains x.
            # So we consider pairs {x, y} for y in S_arr, y != x, and {x, x}.
            
            # Get others (S_arr without x)
            # Since S_arr is sorted, we can find index and slice, or use mask.
            # Mask is O(N).
            mask = S_arr != x
            others = S_arr[mask]
            
            if len(others) > 0:
                counts[x + others] -= 1
            counts[2*x] -= 1
            
            # Update S
            idx = np.searchsorted(S_arr, x)
            # Verify it's actually there (should be, given S_set check)
            if idx < len(S_arr) and S_arr[idx] == x:
                S_new = np.delete(S_arr, idx)
            else:
                return False, S_arr, S_set, counts
            
            S_set.remove(x)
            
            return True, S_new, S_set, counts

        # Greedy: iterate 1..1000
        for x in range(1, 1001):
            ok, S, S_set, counts = try_add(x, S, S_set, counts)
        
        # Phase 2: Local Search
        # Try insertions and swaps
        
        # To speed up, we can pre-compute a list of all numbers.
        # all_nums = np.arange(1, 1001)
        
        # We will try a fixed number of random moves per restart phase
        # and break if no improvement is found after a certain number of tries
        # or if time runs out.
        
        no_improve_count = 0
        
        while time.time() < deadline - 5.0:
            improved = False
            
            # Try random insertions
            # Pick a random number not in S
            for _ in range(50): # Try 50 random insertions
                if time.time() > deadline - 5.0:
                    break
                x = random.randint(1, 1000)
                if x in S_set:
                    continue
                ok, S_new, S_set_new, counts_new = try_add(x, S, S_set, counts)
                if ok:
                    S = S_new
                    S_set = S_set_new
                    counts = counts_new
                    improved = True
                    no_improve_count = 0
                    break # Restart loop to re-evaluate state
            
            if improved:
                continue
            
            # Try random swaps
            # Pick u from S, v from not S
            if len(S) > 0:
                # Convert S to list for random choice, as random.choice on numpy array is slow?
                # Actually random.choice works on sequences. numpy array is a sequence.
                # But it might be slower than list. Let's just use list if size is small.
                S_list = S.tolist()
                for _ in range(50): # Try 50 random swaps
                    if time.time() > deadline - 5.0:
                        break
                    u = random.choice(S_list)
                    v = random.randint(1, 1000)
                    if v in S_set:
                        continue
                    if u == v:
                        continue
                    
                    # Try swap u -> v
                    # 1. Remove u
                    ok1, S_temp, S_set_temp, counts_temp = try_remove(u, S, S_set, counts)
                    if not ok1:
                        continue
                    
                    # 2. Add v
                    ok2, S_new, S_set_new, counts_new = try_add(v, S_temp, S_set_temp, counts_temp)
                    if ok2:
                        S = S_new
                        S_set = S_set_new
                        counts = counts_new
                        improved = True
                        no_improve_count = 0
                        break
                    # If not ok, we discard S_temp etc. Original S remains.
                
                if improved:
                    continue
            
            # If no improvement in this round
            no_improve_count += 1
            if no_improve_count > 10:
                # Stuck, break to next restart
                break
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.tolist()
            
    return best_S
