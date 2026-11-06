import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s. We stop at 85s to be safe.
    deadline = start_time + 85.0
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1 # 129
    
    # Precompute contribution matrix for delta updates
    # contrib[x, s] = 1 if x + y = s for some y? No, we need counts.
    # Let's use a different vectorized approach for delta.
    # For a move (remove x, add y):
    # Delta counts for sum s:
    # - For each z in A (z != x): 
    #   if x+z in range: counts[x+z] -= 1
    #   if y+z in range: counts[y+z] += 1
    # - Self sums:
    #   if 2x in range: counts[2x] -= 1
    #   if 2y in range: counts[2y] += 1
    
    # To make this fast, we can precompute for each x, the list of sums it affects.
    # sums_affected_by[x] = list of s in [128, 256] such that s-x is in [0, 128].
    # Actually, for a fixed x, the sums are x + z for z in [0, 128].
    # So sums are in [x, x+128].
    # Intersection with [128, 256].
    
    # Let's initialize A with a random set of size ~40-50.
    # Why 40-50? 
    # Pairs ~ 1000-1200. Targets 129. Avg ~ 8-9.
    # Current best peak is 8.25.
    # Let's try size 45.
    
    best_A = None
    best_score = float('inf')
    
    # Run multiple restarts to escape local minima
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(42 + r)
        
        # Initialize A
        # Ensure 0 and 128 are included to cover extremes efficiently?
        # 0 covers 128..256 via 0+128..0+256? No, 0+128=128, 0+128 is max if A max is 128.
        # 0 + A covers [min(A), max(A)].
        # 128 + A covers [128+min(A), 128+max(A)].
        # To cover 128, need 0+128 or 64+64 or etc.
        # Let's just start random but force 0 and 128 in.
        
        init_size = 45
        A_set = {0, 128}
        while len(A_set) < init_size:
            A_set.add(random.randint(0, 128))
            
        A = np.zeros(129, dtype=bool)
        for x in A_set:
            A[x] = True
            
        # Calculate initial counts
        # We can compute counts by convolution?
        # A is a boolean vector of length 129.
        # Convolution of A with A gives counts of ordered pairs?
        # np.convolve(A, A) gives c[s] = sum_{i+j=s} A[i]A[j].
        # This counts ordered pairs (i,j).
        # We want unordered pairs {i,j} with i<=j.
        # c_unordered[s] = (c_ordered[s] + (1 if s is even and A[s/2] else 0)) / 2
        # Let's verify.
        # If i != j, (i,j) and (j,i) are both in ordered. Count is 2. Unordered is 1. 2/2=1.
        # If i == j, (i,i) is in ordered. Count is 1. Unordered is 1. (1+1)/2 = 1.
        # So yes.
        
        A_conv = np.convolve(A, A)
        # A_conv[s] is the count for sum s.
        # Note: A_conv has length 257 (indices 0..256).
        
        def get_objective(counts):
            # counts is array of length 257
            c_target = counts[target_lo:target_hi+1]
            violations = np.sum(c_target == 0)
            # Penalty for high peak
            # We want to minimize max(c_target).
            # Using sum of squares or max directly?
            # Max is non-differentiable but fine for SA.
            # Let's use a composite score: violations * 1000 + max(c_target)
            # Or better: violations * 1000 + sum((c_target - avg)^2) to smooth?
            # The task asks to minimize the LARGEST multiplicity.
            # So peak is the primary metric.
            # However, if we have violations, we must fix them first.
            # Score = violations * 10000 + max(c_target)
            if violations > 0:
                return violations * 10000 + np.max(c_target)
            return np.max(c_target)
        
        current_counts = A_conv
        current_score = get_objective(current_counts)
        
        # Simulated Annealing Parameters
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.9999
        
        # Precompute neighbors for speed?
        # Move: pick x in A, pick y not in A. Swap.
        # Delta counts calculation:
        # new_counts = old_counts
        # For each z in A (z != x):
        #   if x+z in [128,256]: new_counts[x+z] -= 1
        #   if y+z in [128,256]: new_counts[y+z] += 1
        # Self:
        #   if 2x in [128,256]: new_counts[2x] -= 1
        #   if 2y in [128,256]: new_counts[2y] += 1
        
        # To speed up, we can precompute for each x, the indices in counts array that change.
        # Let idx_x = [s for s in range(128, 257) if s-x in [0, 128]]
        # This is just range(max(128, x), min(256, x+128)+1)
        
        # Actually, we can just iterate over the changed sums.
        # Number of changed sums is at most 129.
        
        best_local_A = A.copy()
        best_local_score = current_score
        best_local_counts = current_counts.copy()
        
        steps_per_restart = 50000 # Max steps, but we check time
        
        while time.time() < deadline - 2.0:
            if T < T_min:
                break
                
            # Pick x from A, y from not A
            # Using numpy for selection might be slow if we do it every step.
            # Let's keep a list of elements in A for fast sampling.
            # But A changes. 
            # Alternative: pick x in [0,128], y in [0,128].
            # If x not in A and y in A, it's a swap.
            # If x in A and y not in A, it's a swap.
            # If both in or both out, skip.
            
            x = random.randint(0, 128)
            y = random.randint(0, 128)
            
            if x == y:
                continue
            
            if A[x] == A[y]:
                continue # No change in set composition
            
            # Swap x and y
            # x is in A, y is not (or vice versa)
            # Let's assume x in A, y not in A.
            # If A[y] is True and A[x] is False, swap roles.
            if A[y]:
                x, y = y, x
            # Now A[x] is True, A[y] is False.
            
            # Calculate delta counts
            # We need to update counts for sums involving x (remove) and y (add).
            
            # Sums affected by removing x:
            # For each z in A (z != x): sum x+z decreases by 1.
            # Self sum 2x decreases by 1.
            
            # Sums affected by adding y:
            # For each z in A (z != x, z != y): sum y+z increases by 1.
            # Note: z=x is removed, so we don't add y+x?
            # Wait.
            # Old state: x in A.
            # New state: y in A, x not in A.
            
            # Pairs in Old:
            # (x, x) -> 2x
            # (x, z) for z in A, z != x
            
            # Pairs in New:
            # (y, y) -> 2y
            # (y, z) for z in A \ {x}, z != y
            
            # Delta for sum s:
            # -1 if s == 2x
            # -1 if s == x+z for z in A\{x}
            # +1 if s == 2y
            # +1 if s == y+z for z in A\{x}
            
            # We can compute this by iterating z in A.
            # But A is a numpy bool array. Iterating is slow.
            # Let's maintain a list of elements in A.
            
            # Actually, 50000 steps * 129 checks * 5 restarts = 32M checks.
            # Python loop might be too slow.
            # Let's try to vectorize the delta calculation?
            
            # Alternative: Use the convolution property?
            # Updating convolution is hard.
            
            # Let's stick to the list of A elements for iteration, but optimize.
            # If we use a list, updating it is O(1) for swap.
            
            # Let's restart the inner loop with a list-based A representation.
            
            # --- Reset inner loop structure for speed ---
            A_list = [x for x in range(129) if A[x]]
            
            # We need to recalculate counts from scratch initially?
            # We already have current_counts from convolution.
            # Let's assume current_counts is correct.
            
            # To speed up delta:
            # We can precompute for each x, the list of sums it forms with ALL possible z.
            # But we only care about z in A.
            
            # Let's try a different move strategy:
            # Instead of random swap, try to fix a violation or reduce peak.
            # But SA needs random moves.
            
            # Optimization: 
            # The number of sums affected by x is at most 129.
            # We can create a temporary array for delta.
            
            # Let's try to just do the iteration in Python but keep it tight.
            # 50000 steps might be too many if each is 129 ops.
            # 50000 * 129 = 6.5M ops. In Python, ~1-2 seconds.
            # So 5 restarts * 2s = 10s. This is feasible.
            
            # Let's implement the delta update carefully.
            
            # To avoid modifying A_list while iterating, we just iterate over a copy or handle indices.
            # Since we swap x and y, and x is in A_list, y is not.
            # We can just iterate over A_list.
            
            # Precompute sum ranges?
            # s_min_x = max(128, x)
            # s_max_x = min(256, x+128)
            
            # For each z in A_list:
            #   if z == x: continue (handled by self sum logic? No, self sum is separate)
            #   Actually, pair (x,z) is distinct from (x,x).
            #   In the list A_list, x is present once.
            #   The pairs are unordered.
            #   If we iterate z in A_list:
            #     If z == x: this represents the pair (x,x). Sum 2x.
            #     If z != x: this represents the pair (x,z). Sum x+z.
            
            # So:
            # Delta for s:
            # -1 for 2x
            # -1 for x+z (z in A_list, z != x)
            # +1 for 2y
            # +1 for y+z (z in A_list, z != x) # Note: z=x is removed, so we don't add y+x?
            # Wait. In the new set, y is in, x is out.
            # Pairs with y: (y,y) and (y,z) for z in NewA\{y}.
            # NewA = OldA \ {x} U {y}.
            # So z in NewA\{y} is exactly OldA\{x}.
            
            # So yes, iterate z in A_list (which is OldA).
            # If z == x:
            #   Old had (x,x). New has no x.
            #   Old had (x,z) which is (x,x).
            #   So we remove 2x.
            #   Do we add anything for z=x? No, because x is not in NewA.
            #   So for z=x: delta[2x] -= 1.
            # If z != x:
            #   Old had (x,z). Remove x+z.
            #   New has (y,z). Add y+z.
            #   So delta[x+z] -= 1, delta[y+z] += 1.
            # Plus self sum for y: delta[2y] += 1.
            
            # Let's code this.
            
            delta = np.zeros(257, dtype=np.int32)
            
            # Self x
            s_2x = 2 * x
            if 128 <= s_2x <= 256:
                delta[s_2x] -= 1
                
            # Self y
            s_2y = 2 * y
            if 128 <= s_2y <= 256:
                delta[s_2y] += 1
            
            # Pairs
            for z in A_list:
                if z == x:
                    continue # Handled by self x? 
                    # Wait. (x,x) is one pair.
                    # My loop logic:
                    # If z==x, I skipped it in the "Pairs" section.
                    # But I already did delta[2x] -= 1 in "Self x".
                    # Is that correct?
                    # Yes, (x,x) is the only pair involving x and x.
                    
                s_old = x + z
                if 128 <= s_old <= 256:
                    delta[s_old] -= 1
                    
                s_new = y + z
                if 128 <= s_new <= 256:
                    delta[s_new] += 1
            
            new_counts = current_counts + delta
            new_score = get_objective(new_counts)
            
            # Acceptance
            if new_score < current_score:
                # Accept
                current_counts = new_counts
                current_score = new_score
                A[x] = False
                A[y] = True
                # Update A_list
                # Replace x with y in A_list
                idx = A_list.index(x)
                A_list[idx] = y
                A_list.sort() # Keep sorted for determinism? Not strictly needed but good.
                
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_A = A.copy()
                    best_local_counts = current_counts.copy()
            else:
                # Reject or accept with prob exp(-dE/T)
                dE = new_score - current_score
                if dE < 0:
                    pass # Impossible since we checked <
                # Accept with probability
                if random.random() < np.exp(-dE / T):
                    current_counts = new_counts
                    current_score = new_score
                    A[x] = False
                    A[y] = True
                    idx = A_list.index(x)
                    A_list[idx] = y
                    A_list.sort()
                    
            T *= cooling_rate
            
        # End of SA loop for this restart
        
        if best_local_score < best_score:
            best_score = best_local_score
            best_A = best_local_A.copy()
            
        # Early exit if we found a perfect solution?
        # Min peak is likely > 1.
        # If best_score <= 3, break.
        if best_score <= 3:
            break

    if best_A is None:
        # Fallback
        best_A = np.zeros(129, dtype=bool)
        best_A[0] = True
        best_A[128] = True
        # Just return a valid set, even if bad.
        # Fill randomly to ensure coverage?
        for i in range(129):
            best_A[i] = True
            
    # Convert numpy bool array to list of indices
    result = [i for i in range(129) if best_A[i]]
    
    return result

if __name__ == "__main__":
    A = construct()
    print(A)
