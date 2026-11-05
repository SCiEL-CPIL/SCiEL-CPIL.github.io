import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    target_min = 128
    target_max = 256
    target_sums = list(range(target_min, target_max + 1))
    num_targets = len(target_sums)
    
    # Precompute pair contributions for each element x
    # For each x in [0, 128], and for each other y in [0, 128] (y >= x for unordered),
    # we know the sum s = x + y.
    # If x is in A and y is in A, then s gets +1 (or +1 for x!=y, +1 for x==y? 
    # Wait, unordered pairs (a,b) with a<=b.
    # If a != b, the pair is {a,b}. If a == b, the pair is {a,a}.
    # The count N[s] is the number of such pairs.
    
    # Let's precompute for each x, the list of (y, s) pairs that would contribute to N[s]
    # if x is added to A, assuming y is already in A (and y >= x to avoid double counting? 
    # No, it's easier to think of "adding x to A".
    # The new pairs formed are {x, y} for all y in A where y >= x? 
    # Actually, if we maintain A as a set, and we add x, the new pairs are:
    # 1. {x, x} -> sum 2x
    # 2. {x, y} for all y in A such that y > x.
    # But we also need to consider if we REMOVE x.
    
    # Let's use a different representation for speed.
    # A_mask: bool array of size 129.
    # N: int array of size 257.
    
    # Precompute for each x in [0, 128]:
    # contrib[x] = list of (y, s) where y > x. 
    # When adding x:
    #   N[2x] += 1
    #   for each y in A where y > x: N[x+y] += 1
    #   for each y in A where y < x: N[x+y] += 1  (Wait, these were already counted when y was added? 
    #   No, if we add x, and y is already in A, the pair {x,y} is new.
    #   It doesn't matter if y < x or y > x. The pair is unique.
    #   So: for all y in A, y != x: N[x+y] += 1.
    #   And N[2x] += 1.
    
    # To make this fast, we can precompute for each x, the vector of sums it would add.
    # But the sums depend on which y are in A.
    
    # Let's stick to the O(K) update per move, where K is |A| ~ 30-40.
    # This is very fast.
    
    def calculate_counts(A_mask):
        N = np.zeros(257, dtype=np.int32)
        A_indices = np.where(A_mask)[0]
        for i in range(len(A_indices)):
            a = A_indices[i]
            # Self pair
            s = 2 * a
            if target_min <= s <= target_max:
                N[s] += 1
            # Cross pairs with larger elements to avoid double counting in this specific loop structure?
            # Or just loop j from i to end?
            for j in range(i, len(A_indices)):
                b = A_indices[j]
                if i == j: continue # Already handled
                s = a + b
                if target_min <= s <= target_max:
                    N[s] += 1
        return N

    def get_objective(N):
        target_N = N[target_min:target_max+1]
        missing = np.sum(target_N == 0)
        if missing > 0:
            return (missing, float('inf'), float('inf'))
        peak = np.max(target_N)
        sum_sq = np.sum(target_N**2)
        return (0, peak, sum_sq)

    # Simulated Annealing / Random Restart Local Search
    # State: A_mask, N
    # Move: Swap a in A with b not in A.
    
    def local_search(A_mask, N, time_limit):
        A_mask = A_mask.copy()
        N = N.copy()
        obj = get_objective(N)
        if obj[0] > 0:
            return A_mask, N, obj
            
        current_peak = obj[1]
        current_sum_sq = obj[2]
        
        # We will use a simple hill climbing with random restarts on moves.
        # Or just random swaps that improve peak, or peak and sum_sq.
        
        A_indices = np.where(A_mask)[0]
        not_in_A = np.where(~A_mask)[0]
        
        # To speed up swap evaluation, we can precompute the delta N for removing a and adding b?
        # Delta N for removing a:
        #   N[2a] -= 1
        #   for each u in A, u != a: N[a+u] -= 1
        # Delta N for adding b:
        #   N[2b] += 1
        #   for each u in A, u != a (since a is removed, u is in A\{a}): N[b+u] += 1
        #   Note: if b == a, no change. But b not in A, so b != a.
        #   Also, the pair {a,b} is removed and {b,b} and {b,u} added?
        #   Wait, when we swap a->b:
        #   Old pairs involving a: {a,a}, {a,u} for u in A\{a}.
        #   New pairs involving b: {b,b}, {b,u} for u in A\{a}.
        #   Pairs not involving a or b remain unchanged.
        
        while time.time() < time_limit:
            improved = False
            # Try a batch of random swaps
            for _ in range(200):
                if time.time() > time_limit:
                    break
                    
                if len(A_indices) == 0:
                    break
                    
                # Pick random a from A
                a_idx = random.randint(0, len(A_indices) - 1)
                a = A_indices[a_idx]
                
                # Pick random b from not_in_A
                if len(not_in_A) == 0:
                    break
                b_idx = random.randint(0, len(not_in_A) - 1)
                b = not_in_A[b_idx]
                
                # Compute new N
                new_N = N.copy()
                
                # Remove a
                s = 2 * a
                if target_min <= s <= target_max:
                    new_N[s] -= 1
                for u in A_indices:
                    if u == a:
                        continue
                    s = a + u
                    if target_min <= s <= target_max:
                        new_N[s] -= 1
                        
                # Add b
                s = 2 * b
                if target_min <= s <= target_max:
                    new_N[s] += 1
                for u in A_indices:
                    if u == a:
                        continue
                    s = b + u
                    if target_min <= s <= target_max:
                        new_N[s] += 1
                
                new_obj = get_objective(new_N)
                new_peak = new_obj[1]
                new_sum_sq = new_obj[2]
                
                if new_peak < current_peak:
                    # Accept
                    A_mask[a] = False
                    A_mask[b] = True
                    N = new_N
                    current_peak = new_peak
                    current_sum_sq = new_sum_sq
                    # Update indices
                    A_indices[a_idx] = b
                    not_in_A[b_idx] = a
                    # Sort? Not strictly needed for logic, but helps for determinism/debugging.
                    # For performance, we can skip sorting.
                    improved = True
                    # Don't break, keep looking for more peak reductions
                elif new_peak == current_peak:
                    if new_sum_sq < current_sum_sq:
                        # Accept if it balances
                        A_mask[a] = False
                        A_mask[b] = True
                        N = new_N
                        current_sum_sq = new_sum_sq
                        A_indices[a_idx] = b
                        not_in_A[b_idx] = a
                        improved = True
                        # Don't break, might find peak reduction later
                    
            if not improved:
                # Stuck. Try a larger perturbation?
                # Or just break and let the main loop restart.
                # Let's try a few random "jumps" (multiple swaps) to escape local optima.
                escaped = False
                for _ in range(10):
                    if time.time() > time_limit:
                        break
                    # Perform 3-5 random swaps regardless of quality
                    for _ in range(random.randint(3, 5)):
                        if len(A_indices) == 0 or len(not_in_A) == 0:
                            break
                        a_idx = random.randint(0, len(A_indices) - 1)
                        a = A_indices[a_idx]
                        b_idx = random.randint(0, len(not_in_A) - 1)
                        b = not_in_A[b_idx]
                        
                        # Apply swap directly
                        new_N = N.copy()
                        s = 2 * a
                        if target_min <= s <= target_max: new_N[s] -= 1
                        for u in A_indices:
                            if u != a:
                                s = a + u
                                if target_min <= s <= target_max: new_N[s] -= 1
                                
                        s = 2 * b
                        if target_min <= s <= target_max: new_N[s] += 1
                        for u in A_indices:
                            if u != a:
                                s = b + u
                                if target_min <= s <= target_max: new_N[s] += 1
                                
                        A_mask[a] = False
                        A_mask[b] = True
                        N = new_N
                        A_indices[a_idx] = b
                        not_in_A[b_idx] = a
                        
                    # Check if we improved
                    obj_check = get_objective(N)
                    if obj_check[0] == 0:
                        if obj_check[1] < current_peak:
                            current_peak = obj_check[1]
                            current_sum_sq = obj_check[2]
                            escaped = True
                            break
                        elif obj_check[1] == current_peak and obj_check[2] < current_sum_sq:
                            current_sum_sq = obj_check[2]
                            # Stay here, maybe try again?
                            # Let's just break out of the escape loop if we are still valid and not worse peak
                            # Actually, if we didn't improve peak, we might be stuck in a bad spot.
                            # Let's continue searching from this new spot.
                            escaped = True
                            break
                    else:
                        # Invalid, revert? This is getting complex.
                        # Let's just break and restart from the last valid state.
                        # We didn't store the last valid state easily.
                        # Better to just break the local search and let main loop restart.
                        break
                        
                if not escaped:
                    break
                    
        return A_mask, N, (0, current_peak, current_sum_sq)

    def greedy_init():
        # Start with a random set and repair
        A_mask = np.zeros(129, dtype=bool)
        N = np.zeros(257, dtype=np.int32)
        
        # Heuristic: We need to cover [128, 256].
        # Sums are roughly 2x. So elements around 64-128 are important for high sums.
        # Elements around 0-64 are important for low sums?
        # Min sum 128. Max sum 256.
        
        # Let's try a random initialization with ~30 elements.
        k = random.randint(28, 35)
        init_set = set(random.sample(range(129), k))
        for x in init_set:
            A_mask[x] = True
            
        N = calculate_counts(A_mask)
        
        # Repair missing sums
        # If a sum s is missing, we need to add x such that x + y = s for some y in A.
        # Or add x and z such that x+z=s.
        # Simple repair: for each missing sum, try to add an element that covers it.
        
        for _ in range(100): # Max repair steps
            missing_sums = [s for s in target_sums if N[s] == 0]
            if not missing_sums:
                break
                
            # Pick a random missing sum
            s = random.choice(missing_sums)
            
            # Find candidates x such that there exists y in A with x+y=s (x!=y) or 2x=s.
            # Or just try to add any x that increases coverage of missing sums.
            
            best_x = None
            best_cover = 0
            
            # Sample a few candidates
            candidates = [x for x in range(129) if not A_mask[x]]
            random.shuffle(candidates)
            for x in candidates[:20]:
                # Check how many missing sums it covers
                cover = 0
                # It covers s if there is y in A with x+y=s, or 2x=s.
                # It covers other missing sums s' if there is y in A with x+y'=s'.
                
                # Quick check for the specific s
                y = s - x
                if 0 <= y <= 128:
                    if (A_mask[y] and y != x) or (y == x):
                        cover += 1
                        
                # Check other missing sums
                for s2 in missing_sums:
                    if s2 == s: continue
                    y2 = s2 - x
                    if 0 <= y2 <= 128:
                        if (A_mask[y2] and y2 != x) or (y2 == x):
                            cover += 1
                            
                if cover > best_cover:
                    best_cover = cover
                    best_x = x
                    
            if best_x is not None and best_cover > 0:
                # Add best_x
                A_mask[best_x] = True
                # Update N
                A_indices = np.where(A_mask)[0]
                # Remove best_x from indices temporarily? No, just add contributions.
                # Wait, calculate_counts is O(K^2). K is small.
                # Let's just recompute N? Or update incrementally.
                # Incremental update:
                s = 2 * best_x
                if target_min <= s <= target_max:
                    N[s] += 1
                for u in np.where(A_mask)[0]:
                    if u == best_x: continue
                    s = best_x + u
                    if target_min <= s <= target_max:
                        N[s] += 1
            else:
                # Can't fix this missing sum easily?
                # Try adding a random element?
                break
                
        return A_mask, N

    best_peak = float('inf')
    best_A_mask = None
    best_sum_sq = float('inf')
    
    # Main Loop
    for i in range(100):
        if time.time() > deadline:
            break
            
        # Greedy Init
        A_mask, N = greedy_init()
        
        # Check if valid
        obj = get_objective(N)
        if obj[0] > 0:
            # Try to repair more aggressively?
            # For now, skip if not valid after greedy.
            # Or force validity by adding elements until valid.
            # Let's add elements until valid.
            while obj[0] > 0 and time.time() < deadline:
                # Add a random element
                candidates = [x for x in range(129) if not A_mask[x]]
                if not candidates: break
                x = random.choice(candidates)
                A_mask[x] = True
                # Update N
                s = 2 * x
                if target_min <= s <= target_max:
                    N[s] += 1
                for u in np.where(A_mask)[0]:
                    if u == x: continue
                    s = x + u
                    if target_min <= s <= target_max:
                        N[s] += 1
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
        if time_left > 2:
            A_opt, N_opt, obj_opt = local_search(A_mask, N, time_left - 2)
            if obj_opt[0] == 0:
                if obj_opt[1] < best_peak or (obj_opt[1] == best_peak and obj_opt[2] < best_sum_sq):
                    best_peak = obj_opt[1]
                    best_sum_sq = obj_opt[2]
                    best_A_mask = A_opt.copy()
                    
        if best_peak <= 4:
            # Good enough?
            # Keep searching for 3?
            pass

    if best_A_mask is None:
        # Fallback: should not happen
        return list(range(64, 129))
        
    A_list = [i for i in range(129) if best_A_mask[i]]
    return A_list
