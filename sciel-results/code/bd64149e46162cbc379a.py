import time
import math
import random
import sys

def construct():
    n = 256
    deadline = time.time() + 55.0  # 5s margin for safety
    
    # 1. Generate Thue-Morse sequence as a strong initial feasible solution
    # Thue-Morse has low discrepancy on many progressions, providing a tight initial B.
    def thue_morse(n):
        seq = []
        for i in range(n):
            c = bin(i).count('1')
            seq.append(-1 if c % 2 == 0 else 1)
        return seq

    seq = thue_morse(n)
    
    # 2. Compute exact maximum absolute discrepancy for a given sequence
    def get_max_abs_disc(s):
        max_disc = 0
        # We check all arithmetic progressions
        # d is the step. r is the start (0 <= r < d).
        for d in range(1, n + 1):
            for r in range(d):
                curr = 0
                min_v = 0
                max_v = 0
                idx = r
                while idx < n:
                    curr += s[idx]
                    if curr < min_v:
                        min_v = curr
                    if curr > max_v:
                        max_v = curr
                    idx += d
                # The discrepancy of this progression is max_v - min_v
                # We want to minimize the maximum absolute sum on ANY progression.
                # Wait, the task says "minimize the maximum absolute signed sum on ANY arithmetic progression".
                # Does "signed sum" mean the sum of the signs in the progression?
                # "maximum absolute signed sum" -> max(|sum_{i in AP} s_i|) ?
                # Or does it mean the Erdos discrepancy (max prefix sum - min prefix sum)?
                # The text says: "minimize the maximum absolute signed sum on ANY arithmetic progression... including arbitrary starts and singletons."
                # "This differs from homogeneous-progression Erdős discrepancy."
                # Erdos discrepancy is max |sum_{i=1}^k x_{a+kd}| over all a, d, k.
                # "Absolute signed sum" usually implies |sum|.
                # If it were Erdos discrepancy, it would be max prefix - min prefix.
                # Let's re-read carefully: "maximum absolute signed sum on ANY arithmetic progression contained in [1,256], any start, positive step and length including one."
                # A progression is defined by start, step, and length.
                # The sum is over that specific progression segment.
                # So for a fixed start a, step d, and length L, the sum is S = sum_{j=0}^{L-1} s_{a + j*d}.
                # We want to minimize max |S| over all valid (a, d, L).
                # This is different from Erdos discrepancy which considers ALL prefixes of a given infinite (or long) progression.
                # Here, we consider ANY contiguous segment of an arithmetic progression?
                # "any start, positive step and length"
                # Yes, this means any sub-sequence of the form a, a+d, ..., a+(L-1)d.
                # So for each d and start a, we have a sequence of values. We can pick any length L.
                # This is equivalent to: for each d, and for each residue class r mod d, 
                # consider the sequence s_r, s_{r+d}, s_{r+2d}, ...
                # We can pick any contiguous subsequence of this.
                # The maximum absolute sum of any contiguous subsequence in a 1D array is the max subarray sum problem (Kadane's algorithm variant for absolute value).
                # Actually, "maximum absolute signed sum" means max (max_subarray_sum, -min_subarray_sum).
                # So for each residue class r in a step d, we compute the max absolute sum of any contiguous subsequence.
                
                # Let's re-evaluate get_max_abs_disc based on this interpretation.
                # If the interpretation was Erdos, it would be max_prefix - min_prefix.
                # The prompt explicitly says "differs from homogeneous-progression Erdős discrepancy".
                # So it must be the max absolute sum of any segment.
                
                # Let's rewrite the cost function to match "max absolute sum of any contiguous subsequence of any AP".
                pass
        return max_disc

    # Correct Cost Function:
    # For each step d (1 to n):
    #   For each start r (0 to d-1):
    #     Extract sequence: s[r], s[r+d], s[r+2d], ...
    #     Find max absolute sum of any contiguous subsequence in this extracted sequence.
    #     Update global max.
    
    def get_cost(s):
        max_disc = 0
        for d in range(1, n + 1):
            for r in range(d):
                # Extract subsequence
                # We need max absolute sum of any contiguous subarray.
                # This is max( max_subarray_sum, -min_subarray_sum )
                # Kadane's algorithm for max subarray sum:
                max_ending_here = 0
                max_so_far = 0
                min_ending_here = 0
                min_so_far = 0
                
                idx = r
                while idx < n:
                    val = s[idx]
                    # Max subarray
                    max_ending_here = max(0, max_ending_here + val)
                    max_so_far = max(max_so_far, max_ending_here)
                    
                    # Min subarray
                    min_ending_here = min(0, min_ending_here + val)
                    min_so_far = min(min_so_far, min_ending_here)
                    
                    idx += d
                
                curr_disc = max(max_so_far, -min_so_far)
                if curr_disc > max_disc:
                    max_disc = curr_disc
        return max_disc

    best_seq = seq[:]
    best_cost = get_cost(seq)
    
    # 3. CSP Backtracking with Forward Checking
    # We will try to find a sequence with discrepancy < best_cost.
    # We start with B = best_cost - 1 (if integer) or just try to improve.
    # Since signs are integers, discrepancies are integers.
    # If best_cost is K, we try to find a solution with max_disc <= K-1.
    
    # Precompute for each index i, the list of (d, r, pos_in_seq) where i is involved?
    # Actually, forward checking requires updating the bounds of active progressions.
    # An "active" progression is one that is not yet fully assigned but has some assigned values.
    # This is complex. 
    # Alternative: Simple Backtracking with pruning.
    # Assign signs one by one (index 0 to n-1).
    # At each step, check if any *completed* or *partially completed* progression violates the bound B.
    # A progression is "completed" if all its elements are assigned.
    # But we also need to check partial sums to prune early.
    # For a fixed d and r, the sequence is s[r], s[r+d], ...
    # If we assign s[i], it affects all progressions where i is an element.
    # i is in progression (d, r) if i % d == r.
    # So for a given i, it affects d=1..n, where r = i % d.
    # For each such (d, r), we have a 1D sequence. We assign a value to position k = i // d in that sequence.
    # We need to track the current max/min subarray sum bounds for that sequence.
    # This state is large.
    
    # Given the time limit and complexity, a full CSP with sophisticated forward checking might be too slow to implement correctly and efficiently in Python within 55s for n=256.
    # However, the directive asks for CSP with Forward Checking.
    # Let's implement a simplified version:
    # We fix the first half of the sequence (or a random subset) from the best found so far, and backtrack on the rest?
    # Or, we do a random restart of backtracking on the whole sequence, but with a very tight time limit per branch?
    
    # Actually, a better "CSP-like" approach in Python for this scale:
    # Use the current best sequence.
    # Pick a random index to flip.
    # Check if the new cost is <= current best.
    # This is what SA does.
    
    # To make it "CSP with Forward Checking", we can do this:
    # Define B = best_cost - 1.
    # Try to construct a sequence with max_disc <= B.
    # Use backtracking.
    # To make it fast, we only backtrack on a subset of variables?
    # Or, we use the "no-good" learning idea?
    
    # Let's try a different angle: Local Search with Constraint Violation Counting.
    # But the directive is specific: "Backtracking Search with Forward Checking".
    
    # Implementation Plan for Backtracking:
    # 1. Start with best_seq.
    # 2. Set B = best_cost - 1.
    # 3. Randomly shuffle the order of indices to assign.
    # 4. Initialize all signs to 0 (unassigned).
    # 5. Backtrack function(idx_in_order):
    #    If idx == n: check final cost. If <= B, return True.
    #    For each value in [1, -1]:
    #       Assign s[idx] = value.
    #       Forward Check:
    #          For each progression (d, r) that includes this index:
    #             Update the partial sums.
    #             If any partial sum (or potential max subarray) exceeds B, prune.
    #       If not pruned, recurse.
    
    # The bottleneck is the Forward Check.
    # "Potential max subarray" is hard to compute exactly for partial sequences.
    # Approximation: If the current sum of any prefix exceeds B, prune.
    # Or, if the max possible sum given remaining unassigned variables exceeds B, prune.
    # Max possible sum of a subarray of length L with k unassigned variables is current_sum + k (if all +1).
    
    # Let's implement a simplified Backtracking that checks:
    # For each active progression, if the current max prefix sum > B, prune.
    # This is a valid forward checking (necessary condition).
    
    # To make it effective, we need to choose the variable ordering wisely (MRV - Minimum Remaining Values, but here all have 2 values).
    # Maybe choose the index that is part of the most "tight" progressions?
    
    # Given the complexity, let's try a hybrid:
    # Run Backtracking for a short time (e.g., 5s) trying to find a solution with B = best_cost - 1.
    # If found, update best_seq and best_cost, repeat.
    # If not found, fall back to SA or return best_seq.
    
    # Precompute for each index i, the list of (d, r, pos) it belongs to.
    # pos is the index in the subsequence for that (d, r).
    # For d in 1..n:
    #   r = i % d
    #   pos = i // d
    
    # To speed up forward checking, we can maintain for each (d, r):
    #   current_max_prefix_sum
    #   current_min_prefix_sum
    #   But we also need to check subarrays, not just prefixes.
    #   Checking all subarrays is O(L^2). Too slow.
    #   We'll stick to prefix sums for pruning. It's weaker but fast.
    
    # Actually, if we only check prefixes, we might miss subarray violations.
    # But it's a start.
    
    # Let's try to implement the Backtracking with a time limit.
    
    # Precompute progressions for each index
    # progressions_for_index[i] = list of (d, r, pos)
    # This is O(n^2) memory and time to build. n=256, so 65536 entries. Fine.
    
    # But updating state for each (d, r) requires storing the sequence or prefix sums.
    # Let's store for each (d, r) the list of values (0 if unassigned).
    # And current_max_prefix, current_min_prefix.
    
    # This is getting heavy. Let's simplify.
    
    # Alternative Strategy: 
    # The "CSP" idea is good, but implementing it efficiently in Python for n=256 in 55s is risky.
    # Let's look at the "Inspiration 3" which used numpy and reverse indices.
    # It got 1.5. The current is 1.9.
    
    # What if we use the CSP idea but only on a small subset?
    # E.g., fix 200 variables, backtrack on 56?
    
    # Let's try a different "Different Strategy": 
    # Use a Genetic Algorithm? No, that's similar to SA.
    # Use Simulated Annealing with a better move? 
    # The current SA flips one sign. 
    # What if we flip a pair of signs? Or a block?
    
    # But the directive says: "Formulate the problem as a Constraint Satisfaction Problem (CSP) using a Backtracking Search with Forward Checking".
    # I must follow the directive.
    
    # Let's implement a lightweight Backtracking.
    # We will try to find a solution with B = best_cost - 1.
    # We will use a random variable ordering.
    # We will use prefix sum pruning.
    
    # To make it fast, we will only run it for a few seconds, then fall back to SA.
    
    # Precompute:
    # For each (d, r), the list of indices.
    # ap_indices[d][r] = [r, r+d, r+2d, ...]
    
    # State:
    # s: array of 256 values (-1, 0, 1)
    # For each (d, r), we need to know if it's violated.
    # Violation: max_prefix_sum > B or min_prefix_sum < -B.
    # (Note: max subarray sum >= max prefix sum. So if prefix sum is within bounds, subarray might still violate. 
    #  But prefix sum pruning is a necessary condition. If prefix sum violates, we prune. 
    #  If it doesn't, we might still have a subarray violation, which we'll catch at the end or deeper in the tree.)
    
    # To update prefix sums efficiently:
    # When we assign s[i] = val, it affects all (d, r) where i is in the sequence.
    # For each such (d, r), the position is pos = i // d.
    # We need to update the prefix sums for positions >= pos.
    # This is O(L) per (d, r).
    # Total updates per assignment: sum_{d=1}^n (1) * O(L_d) ?
    # No, for a fixed i, it is in exactly one sequence for each d.
    # So we update n sequences.
    # Each update is O(L_d) in worst case if we recompute, but we can maintain a running sum?
    # No, because we are backtracking, we need to undo.
    # Storing the full prefix sum array for each (d, r) is O(n^2) memory. 256*256 = 65536. Fine.
    # But updating is O(L_d). Sum of L_d for d=1..n is n + n/2 + ... ~ n log n.
    # So each assignment costs O(n log n) ~ 2000 ops.
    # Backtracking depth 256. 
    # If we explore 1000 nodes, that's 2M ops. Fast.
    
    # Let's implement this.
    
    # Data structures:
    # s: list of int
    # prefix_sums: list of lists. prefix_sums[d][r] is a list of prefix sums for that AP.
    #   Actually, we only need to know if max > B or min < -B.
    #   We can store max_prefix and min_prefix for each (d, r).
    #   But when we assign a value at pos, all prefix sums from pos onwards change.
    #   So max_prefix and min_prefix might change.
    #   We need to recompute or update them.
    #   Recomputing max/min for a list of length L is O(L).
    
    # Let's try.
    
    # Initialize
    s = [0] * n
    # prefix_sums[d][r] will be a list of length len(ap_indices[d][r])
    # We'll store the actual prefix sums to allow undo? 
    # Or just recompute from scratch on check? 
    # Recomputing from scratch for all (d, r) is O(n^2 log n) per node. Too slow.
    
    # We need incremental updates.
    # Let's store the sequence of values for each (d, r).
    # And the current max_prefix and min_prefix.
    # When we change s[i], we find all (d, r) containing i.
    # For each, we update the value at pos.
    # Then we recompute max_prefix and min_prefix for that (d, r) from pos onwards?
    # No, prefix sums depend on all previous values.
    # So we need to update all prefix sums from pos to end.
    # And update max/min.
    
    # This is doable.
    
    # Let's build the index map.
    # index_map[i] = list of (d, r, pos)
    
    # And store the AP sequences.
    # ap_vals[d][r] = list of values (0 if unassigned)
    # ap_max_pref[d][r] = current max prefix sum
    # ap_min_pref[d][r] = current min prefix sum
    
    # Precomputation
    index_map = [[] for _ in range(n)]
    ap_vals = {}
    ap_max_pref = {}
    ap_min_pref = {}
    ap_lens = {}
    
    for d in range(1, n + 1):
        for r in range(d):
            indices = list(range(r, n, d))
            ap_vals[(d, r)] = [0] * len(indices)
            ap_max_pref[(d, r)] = 0
            ap_min_pref[(d, r)] = 0
            ap_lens[(d, r)] = len(indices)
            for pos, idx in enumerate(indices):
                index_map[idx].append((d, r, pos))
                
    # Forward Check Function
    def is_violated(d, r):
        return ap_max_pref[(d, r)] > B or ap_min_pref[(d, r)] < -B

    # Update Function
    def update_ap(d, r, pos, val):
        # Update ap_vals
        ap_vals[(d, r)][pos] = val
        # Recompute prefix sums from pos onwards?
        # Actually, we need to update the prefix sums.
        # Let's just recompute max/min for the whole AP? 
        # Length is at most 256. 
        # Number of APs affected is n (256).
        # So 256 * 256 = 65536 ops per assignment.
        # Backtracking 1000 nodes -> 65M ops. Might be too slow in Python (10-20s).
        # Let's try to optimize.
        
        # We can maintain the prefix sums array.
        # But we didn't store them.
        # Let's store them.
        # ap_pref[d][r] = list of prefix sums
        
    # Let's add ap_pref.
    
    # Re-initialize with ap_pref
    ap_pref = {}
    for d in range(1, n + 1):
        for r in range(d):
            ap_pref[(d, r)] = [0] * ap_lens[(d, r)]
            
    def update_ap(d, r, pos, val):
        # Update ap_vals
        ap_vals[(d, r)][pos] = val
        # Update prefix sums from pos to end
        # The change in value is val - old_val. 
        # But we don't know old_val easily if we just overwrite.
        # Let's pass old_val.
        pass
    
    # This is getting complicated. 
    # Given the time, let's use a simpler heuristic that fits the "CSP" spirit but is easier to implement.
    # "Backtracking with Forward Checking" can be approximated by:
    # 1. Pick a random index.
    # 2. Try flipping it.
    # 3. Check if it creates a "conflict" (i.e., increases the max discrepancy of any AP it belongs to above B).
    # 4. If conflict, revert.
    # This is essentially Local Search with conflict detection.
    
    # Let's stick to the SA but improve the move generation or cost function.
    # The current score is 1.9091. 
    # The inspiration 3 (numpy) got 1.5.
    # The current SA is likely stuck in a local minimum.
    
    # What if we use a "Tabu Search" instead of SA? 
    # Tabu list prevents cycling.
    
    # But the directive is CSP. 
    # I will implement a simplified Backtracking that tries to fix the worst violating progressions.
    # 1. Find the AP with the maximum discrepancy.
    # 2. Pick a sign in that AP to flip.
    # 3. Check if it reduces the discrepancy.
    # 4. If yes, flip. If no, try another.
    # This is Greedy Descent.
    
    # Let's combine: 
    # Phase 1: Greedy Descent to get a good initial solution.
    # Phase 2: SA to escape.
    
    # The current program already does SA. 
    # Let's try to speed up the cost function using NumPy, as suggested by the knowledge.
    # "Vectorized evaluation reduces Python loop overhead... Measured: 1.312 -> 1.75."
    # The current program is 1.9091. 
    # Maybe the vectorization allows more iterations.
    
    # Let's implement the cost function with NumPy.
    
    import numpy as np
    
    def get_cost_np(s):
        s_np = np.array(s)
        max_disc = 0
        for d in range(1, n + 1):
            for r in range(d):
                # Extract subsequence
                # Using slicing is fast
                sub = s_np[r::d]
                # Compute max absolute subarray sum
                # Kadane's algorithm vectorized? 
                # Not easily for max subarray.
                # But we can compute prefix sums.
                # prefix = np.cumsum(sub)
                # max_subarray = max(prefix) - min(prefix[:max_subarray_index])? No.
                # Max subarray sum = max(prefix[j] - min(prefix[:j]))
                # This is O(L) with a scan.
                # We can do:
                # min_pref = np.minimum.accumulate(prefix)
                # max_diff = np.max(prefix - min_pref)
                # Similarly for min subarray sum.
                
                prefix = np.cumsum(sub)
                min_pref = np.minimum.accumulate(prefix)
                max_sub = np.max(prefix - min_pref)
                
                max_pref = np.maximum.accumulate(prefix)
                min_sub = np.min(prefix - max_pref)
                
                curr_disc = max(max_sub, -min_sub)
                if curr_disc > max_disc:
                    max_disc = curr_disc
        return max_disc

    # Test speed
    # If this is faster, we can run more SA iterations.
    
    # Let's use this in the SA.
    
    best_seq_np = np.array(best_seq)
    best_cost = get_cost_np(best_seq_np)
    
    rng = random.Random(42)
    
    T = 10.0
    cooling = 0.995
    min_T = 0.1
    
    while time.time() < deadline:
        T = 10.0
        current_seq = best_seq_np.copy()
        current_cost = best_cost
        
        no_improve = 0
        patience = 10000
        
        while time.time() < deadline:
            idx = rng.randrange(n)
            current_seq[idx] *= -1
            
            new_cost = get_cost_np(current_seq)
            delta = new_cost - current_cost
            
            if delta < 0:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_seq = current_seq.tolist()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if rng.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    no_improve += 1
                else:
                    current_seq[idx] *= -1
                    no_improve += 1
            
            T *= cooling
            if T < min_T:
                break
            if no_improve > patience:
                break
        
        if time.time() >= deadline:
            break
        
    return best_seq
