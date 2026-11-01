import time
import random
import numpy as np

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin for 120s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute base for index <-> word conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute balls for all indices
    # ball_indices[i] is a numpy array of indices within Hamming distance 'radius'
    ball_indices = [None] * total
    for i in range(total):
        temp = i
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q
        
        ball = [i]
        # Radius 1 logic: change one coordinate to any other value in 0..q-1
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                ball.append(i + (v - orig) * base[coord])
        ball_indices[i] = np.array(ball, dtype=np.int32)

    # Helper to convert index to list of digits
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        return w

    # Greedy initialization
    def greedy_init():
        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered_count = total
        
        # Precompute a list of all indices for random access
        all_indices = list(range(total))
        
        while uncovered_count > 0:
            best_c = -1
            max_new = -1
            
            # Optimization: If uncovered is small, we might find a codeword quickly.
            # But standard greedy checks all.
            # To speed up, we can sample if time is tight, but 2187 is small enough to scan.
            # However, scanning 2187 * 15 ops is ~30k ops. Doing this ~150 times is ~4.5M ops.
            # This is fast in Python if vectorized, but loops are slow.
            
            # Let's use numpy for the coverage check in greedy to be fast.
            # We want max( count of 0s in covered[ball_indices[c]] )
            
            # Since we do this repeatedly, let's just loop.
            # We can optimize by only checking candidates that cover at least one uncovered point?
            # Hard to know without checking.
            
            # Randomized greedy to break symmetry
            random.shuffle(all_indices)
            
            for c in all_indices:
                ball = ball_indices[c]
                # Count how many in this ball are currently uncovered
                new_cov = np.count_nonzero(covered[ball] == 0)
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            
            code.append(best_c)
            covered[ball_indices[best_c]] = 1
            uncovered_count = np.count_nonzero(covered == 0)
            
        return code

    # Simulated Annealing / Local Search
    def local_search(initial_code, time_budget):
        # State: Set of codewords (indices)
        code_set = set(initial_code)
        
        # Coverage count: count[i] = number of codewords covering index i
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_indices[c]] += 1
        
        uncovered = np.count_nonzero(cnt == 0)
        
        # We want to minimize size, subject to uncovered == 0.
        # Penalty for uncovered is high.
        # Cost = len(code_set) + 1000 * uncovered
        
        best_code = list(code_set)
        best_size = len(code_set)
        
        # Temperature
        T = 1.0
        cooling = 0.9999
        
        t_start = time.time()
        
        # Precompute list of all indices for random sampling
        all_indices = list(range(total))
        
        while time.time() - t_start < time_budget:
            # Random move type
            r = random.random()
            
            if r < 0.3:
                # Try to remove a codeword
                if len(code_set) == 0:
                    continue
                c_out = random.choice(list(code_set))
                
                # Calculate delta
                # Remove c_out
                ball = ball_indices[c_out]
                
                # Check if any point in ball becomes uncovered
                # new_cnt = cnt - 1 for points in ball
                # If any cnt[x] == 1, it becomes 0 -> uncovered increases
                # Number of points that become uncovered = count of x in ball where cnt[x] == 1
                delta_uncovered = np.count_nonzero(cnt[ball] == 1)
                
                delta_cost = -1 + 1000 * delta_uncovered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    code_set.remove(c_out)
                    cnt[ball] -= 1
                    uncovered += delta_uncovered
                    
                    if uncovered == 0:
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)
                
            elif r < 0.6:
                # Try to add a codeword (useful if we are in a bad state with uncovered points)
                # Only useful if we have uncovered points?
                # Or if it allows future removals?
                # For now, let's only add if it reduces uncovered.
                if uncovered == 0:
                    # Adding a codeword increases size, doesn't help size minimization directly
                    # unless it enables a swap later. But swaps are handled separately.
                    # Let's skip pure adds if covered, to keep size down.
                    continue
                
                c_in = random.choice(all_indices)
                if c_in in code_set:
                    continue
                
                ball = ball_indices[c_in]
                # Points in ball that are currently uncovered (cnt == 0)
                # Adding c_in will cover them.
                # delta_uncovered is negative (reduction)
                num_covered_new = np.count_nonzero(cnt[ball] == 0)
                delta_uncovered = -num_covered_new
                
                delta_cost = 1 + 1000 * delta_uncovered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    code_set.add(c_in)
                    cnt[ball] += 1
                    uncovered += delta_uncovered
                    
            else:
                # Swap: remove one, add one
                if len(code_set) == 0:
                    continue
                c_out = random.choice(list(code_set))
                c_in = random.choice(all_indices)
                
                if c_in in code_set or c_in == c_out:
                    continue
                
                ball_out = ball_indices[c_out]
                ball_in = ball_indices[c_in]
                
                # Calculate change in uncovered count
                # We need to be careful with points in intersection.
                # Easier: just simulate on a copy? No, too slow.
                # Let's calculate delta_uncovered manually.
                
                # Points affected are in union(ball_out, ball_in)
                # But we can just check:
                # 1. Points in ball_out that are NOT in ball_in: cnt decreases by 1
                # 2. Points in ball_in that are NOT in ball_out: cnt increases by 1
                # 3. Points in both: cnt unchanged
                
                # Actually, simpler logic:
                # Remove c_out:
                #   Points in ball_out with cnt == 1 become uncovered.
                #   Points in ball_out with cnt > 1 stay covered.
                # Add c_in:
                #   Points in ball_in with cnt == 0 become covered.
                #   Points in ball_in with cnt > 0 stay covered.
                
                # Interaction:
                # Let U be set of points becoming uncovered due to removal.
                # Let C be set of points becoming covered due to addition.
                # Net change in uncovered count = |U \ C| - |C \ U| ?
                # No.
                # Uncovered count U_total = count(cnt == 0).
                
                # Let's just compute the new uncovered count for the affected points.
                # Affected points = union(ball_out, ball_in)
                # We can create a boolean mask or just iterate.
                # Size of union <= 30.
                
                affected = np.union1d(ball_out, ball_in)
                
                # Current cnt for affected
                old_cnt = cnt[affected]
                
                # New cnt for affected
                # For x in affected:
                # new_cnt = old_cnt[x] - (1 if x in ball_out else 0) + (1 if x in ball_in else 0)
                
                in_out = np.isin(affected, ball_out)
                in_in = np.isin(affected, ball_in)
                
                new_cnt = old_cnt - in_out.astype(np.int32) + in_in.astype(np.int32)
                
                # Old uncovered in affected: count of old_cnt == 0
                old_uncov = np.count_nonzero(old_cnt == 0)
                # New uncovered in affected: count of new_cnt == 0
                new_uncov = np.count_nonzero(new_cnt == 0)
                
                delta_uncovered = new_uncov - old_uncov
                
                delta_cost = 0 + 1000 * delta_uncovered # Size doesn't change
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    code_set.remove(c_out)
                    code_set.add(c_in)
                    
                    # Update cnt
                    cnt[ball_out] -= 1
                    cnt[ball_in] += 1
                    uncovered += delta_uncovered

            # Cooling
            T *= cooling
            if T < 0.01:
                T = 1.0 # Reset temperature occasionally to escape local minima? 
                # Or keep cooling. Resetting is like random restarts.
                # Let's do a soft reset: T = 1.0 if no improvement in last X steps?
                # For simplicity, just cool. If stuck, the random moves will eventually find a way out or we restart.

        return best_code

    # Run multiple restarts
    best_overall = []
    best_size_overall = total

    # 1. Greedy + LS
    greedy_code = greedy_init()
    # Allocate most time to the first run
    ls_code = local_search(greedy_code, time_budget=60.0)
    best_size_overall = len(ls_code)
    best_overall = ls_code

    # 2. More restarts with less time each
    seed = 1
    while time.time() < deadline - 5.0:
        greedy_code = greedy_init() # Re-randomize greedy? 
        # The greedy_init uses random.shuffle, so it's different each time.
        
        # If we have time, run LS on this new greedy solution
        time_left = deadline - 5.0 - time.time()
        if time_left < 5.0:
            break
            
        # Shorter budget for subsequent runs
        tb = min(10.0, time_left / 2.0)
        
        ls_code = local_search(greedy_code, time_budget=tb)
        
        if len(ls_code) < best_size_overall:
            best_size_overall = len(ls_code)
            best_overall = ls_code
            
        seed += 1

    # Convert indices to words
    result = []
    for idx in best_overall:
        result.append(idx_to_word(idx))
        
    return result
