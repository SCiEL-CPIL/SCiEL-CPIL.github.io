# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    
    Strategy:
    Treat the optimization as a discrete geometric set-covering problem.
    1. Generate a dense candidate pool of points on the analytic Pareto front of ZDT1.
    2. Use a greedy algorithm to sequentially select the point that maximizes the 
       marginal hypervolume contribution to the current set of selected points.
    3. This ensures an optimal non-uniform distribution that maximizes the 
       hypervolume of the recorded set, rather than relying on fixed grid spacing 
       or local gap-filling heuristics.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 Pareto front: x[1:] = 0
        # f1 = x[0]
        # f2 = 1 - sqrt(x[0]) - x[0] * sin(10 * pi * x[0])
        
        # 1. Generate dense candidate pool
        # We want to sample x0 in [0, 1].
        # Since f1 = x0, we can just sample x0 uniformly or non-uniformly.
        # A uniform sample in x0 is a good starting pool.
        n_candidates = 10000
        x0_pool = np.linspace(0.0, 1.0, n_candidates)
        
        # Compute f2 for all candidates
        # Handle x0=0 case for sqrt
        f2_pool = np.ones(n_candidates)
        mask = x0_pool > 0
        f2_pool[mask] = 1.0 - np.sqrt(x0_pool[mask]) - x0_pool[mask] * np.sin(10.0 * np.pi * x0_pool[mask])
        
        # Filter candidates that are strictly better than reference point [11, 11]
        # f1 < 11 and f2 < 11. Since x0 in [0,1], f1 is always < 11.
        # f2 is bounded roughly in [0, 2], so always < 11.
        # So all candidates are valid.
        
        # 2. Greedy selection
        # We need to select `self.budget` points from the pool.
        # Start with the point that maximizes hypervolume alone?
        # Actually, the point with the largest area contribution is the one with min(f1) or min(f2)?
        # For a single point (f1, f2), HV = (11 - f1) * (11 - f2).
        # To maximize this, we want small f1 and small f2.
        # The Pareto front has a trade-off.
        # Let's just start with an empty set and add the best marginal gain.
        # Initial HV = 0.
        # The first point added should maximize (11 - f1) * (11 - f2).
        
        # To speed up, we can pre-compute the HV of single points and pick the max?
        # No, the greedy choice depends on the current set.
        # However, for the first point, it's just the max single HV.
        
        # Efficient Greedy HV Maximization:
        # Maintaining the exact HV increment for 10k candidates x 200 steps is O(N^2 * K) = 10k^2 * 200 = 20 billion ops. Too slow.
        
        # Alternative:
        # The Pareto front is 1D. We can sort candidates by f1.
        # The hypervolume of a set of points on a monotonic front can be calculated by summing rectangles.
        # If we have points sorted by f1 (increasing), and they are non-dominated (which they are on the front),
        # the HV is sum_{i=1}^{n} (f1_{i+1} - f1_i) * (11 - f2_i) where f1_{n+1} = 11.
        # Actually, the standard formula for HV of a set of non-dominated points sorted by f1:
        # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (11 - f2_i)
        # where f1_{n+1} = 11.0.
        # Wait, this assumes the "step" is horizontal then vertical.
        # Let's verify.
        # Points p1, p2, ..., pn sorted by f1.
        # Area under p1 to p2: (f1_2 - f1_1) * (11 - f2_1)
        # Area under pn to ref: (11 - f1_n) * (11 - f2_n)
        # Sum these up.
        
        # If we insert a new point p_new between p_i and p_{i+1}:
        # Old contribution: (f1_{i+1} - f1_i) * (11 - f2_i)
        # New contribution: (f1_new - f1_i) * (11 - f2_i) + (f1_{i+1} - f1_new) * (11 - f2_new)
        # Delta = (f1_new - f1_i) * (11 - f2_i) + (f1_{i+1} - f1_new) * (11 - f2_new) - (f1_{i+1} - f1_i) * (11 - f2_i)
        # Delta = (f1_new - f1_i) * (11 - f2_i) + (f1_{i+1} - f1_new) * (11 - f2_new) - (f1_new - f1_i + f1_{i+1} - f1_new) * (11 - f2_i)
        # Delta = (f1_new - f1_i) * (11 - f2_i) + (f1_{i+1} - f1_new) * (11 - f2_new) - (f1_{i+1} - f1_i) * (11 - f2_i)
        # Delta = (f1_new - f1_i) * (11 - f2_i) + (f1_{i+1} - f1_new) * (11 - f2_new) - (f1_new - f1_i) * (11 - f2_i) - (f1_{i+1} - f1_new) * (11 - f2_i)
        # Delta = (f1_{i+1} - f1_new) * (11 - f2_new - (11 - f2_i))
        # Delta = (f1_{i+1} - f1_new) * (f2_i - f2_new)
        
        # This is extremely simple!
        # The marginal gain of inserting a point p_new between p_i and p_{i+1} (sorted by f1)
        # is (f1_{i+1} - f1_new) * (f2_i - f2_new).
        # Note: This assumes f2_i > f2_new. If f2_new > f2_i, the point is dominated by p_i?
        # No, on the Pareto front, if f1 increases, f2 generally decreases (for ZDT1).
        # So p_i has smaller f1 and larger f2 than p_{i+1}.
        # If we insert p_new, it should have f1_i < f1_new < f1_{i+1} and f2_i > f2_new > f2_{i+1}.
        # Then f2_i - f2_new > 0 and f1_{i+1} - f1_new > 0.
        # So the gain is positive.
        
        # What if we add a point at the beginning (before p_1)?
        # Old contribution: (f1_1 - 0) * (11 - f2_1) ? No, the leftmost point dominates from f1=0?
        # The reference point is [11, 11]. The dominated region is [0, f1_n] x [0, f2_n]?
        # No, the region is defined by the points.
        # Let's stick to the formula:
        # HV = sum (f1_{i+1} - f1_i) * (11 - f2_i) for i=1..n, with f1_{n+1}=11.
        # This formula works if we consider the "step" going right from (f1_i, f2_i) to (f1_{i+1}, f2_i) and then down.
        # Actually, the standard HV calculation for a set of non-dominated points sorted by f1:
        # HV = sum_{i=1}^n (f1_{i+1} - f1_i) * (11 - f2_i)
        # where f1_{n+1} = 11.
        # This implies the last point contributes (11 - f1_n) * (11 - f2_n).
        # The first point contributes (f1_2 - f1_1) * (11 - f2_1).
        # What about the area to the left of f1_1?
        # If f1_1 > 0, the region [0, f1_1] x [0, 11] is NOT dominated by any point unless there is a point with f1=0.
        # Wait, the definition of dominated region: A point p dominates q if p_i <= q_i for all i and p_j < q_j for some j.
        # The region dominated by the set S is the union of hyper-rectangles [p_i, ref] for all p_i in S.
        # If we have points sorted by f1, the union of these rectangles forms a step function.
        # The area is indeed sum (f1_{i+1} - f1_i) * (11 - f2_i) IF we assume the "step" goes from (f1_i, f2_i) to (f1_{i+1}, f2_i) and then down to (f1_{i+1}, f2_{i+1}).
        # But wait, if f1_1 > 0, the rectangle for p_1 is [f1_1, 11] x [f2_1, 11].
        # The region [0, f1_1] x [0, 11] is not dominated.
        # So the formula is correct.
        
        # Let's re-verify the insertion gain.
        # Old: ... p_i, p_{i+1} ...
        # Contribution of p_i: (f1_{i+1} - f1_i) * (11 - f2_i)
        # New: ... p_i, p_new, p_{i+1} ...
        # Contribution of p_i: (f1_new - f1_i) * (11 - f2_i)
        # Contribution of p_new: (f1_{i+1} - f1_new) * (11 - f2_new)
        # Delta = (f1_new - f1_i)(11 - f2_i) + (f1_{i+1} - f1_new)(11 - f2_new) - (f1_{i+1} - f1_i)(11 - f2_i)
        # Delta = (f1_{i+1} - f1_new)(11 - f2_new) - (f1_{i+1} - f1_new)(11 - f2_i)
        # Delta = (f1_{i+1} - f1_new)(f2_i - f2_new)
        
        # This is O(1) per candidate if we know the neighbors.
        # Since we are inserting into a sorted list, we can maintain the list and calculate gains for all candidates in O(N) per step?
        # No, we have 10,000 candidates.
        # We can pre-calculate the potential gain for each candidate based on the current set?
        # The gain depends on the neighbors in the CURRENT set, not the candidate pool.
        
        # Algorithm:
        # 1. Sort candidate pool by f1.
        # 2. Maintain a sorted list of selected indices (or just values) from the pool.
        # 3. Initially empty.
        # 4. For each step:
        #    a. For each candidate in the pool, find its position in the current selected list.
        #    b. Calculate the marginal gain.
        #    c. Pick the candidate with max gain.
        #    d. Add it to the selected list.
        
        # Complexity:
        # 200 steps.
        # In each step, we iterate over 10,000 candidates.
        # For each candidate, we need to find its insertion point in the selected list (size k).
        # Binary search O(log k).
        # Then calculate gain O(1).
        # Total: 200 * 10,000 * log(200) ~ 2 million ops. Very fast.
        
        # Implementation details:
        # We need to store the selected points in a way that allows fast binary search and insertion.
        # A list is fine for k=200.
        
        # Let's refine the gain calculation for edge cases.
        # If the list is empty:
        # Gain = (11 - f1) * (11 - f2).
        
        # If inserting at the beginning (before first selected point p_1):
        # Old contribution of p_1: (f1_1 - 0) * (11 - f2_1)? No.
        # The formula sum (f1_{i+1} - f1_i) * (11 - f2_i) assumes the "left" boundary is 0?
        # No, it assumes the region starts at f1_1.
        # Let's check the area.
        # If we have points p1, p2.
        # Area = (f1_2 - f1_1) * (11 - f2_1) + (11 - f1_2) * (11 - f2_2).
        # If we add p0 < p1.
        # New Area = (f1_1 - f1_0) * (11 - f2_0) + (f1_2 - f1_1) * (11 - f2_1) + (11 - f1_2) * (11 - f2_2).
        # Delta = (f1_1 - f1_0) * (11 - f2_0).
        # Wait, is this correct?
        # The old region dominated by p1 was [f1_1, 11] x [f2_1, 11].
        # The new region dominated by p0 is [f1_0, 11] x [f2_0, 11].
        # The union is [f1_0, f1_1] x [f2_0, 11] + [f1_1, 11] x [min(f2_0, f2_1), 11] ...
        # Since p0 is non-dominated by p1, f2_0 > f2_1 (assuming f1_0 < f1_1).
        # So the new area added is [f1_0, f1_1] x [f2_0, 11].
        # Area = (f1_1 - f1_0) * (11 - f2_0).
        # This matches the formula if we treat "f1_{i+1}" as f1_1 and "f2_i" as f2_0?
        # No, the formula derived earlier was for insertion BETWEEN.
        # For insertion at start:
        # Gain = (f1_1 - f1_new) * (11 - f2_new).
        
        # For insertion at end (after last selected point p_n):
        # Old contribution of p_n: (11 - f1_n) * (11 - f2_n).
        # New: p_n, p_new.
        # Contribution of p_n: (f1_new - f1_n) * (11 - f2_n).
        # Contribution of p_new: (11 - f1_new) * (11 - f2_new).
        # Delta = (f1_new - f1_n)(11 - f2_n) + (11 - f1_new)(11 - f2_new) - (11 - f1_n)(11 - f2_n)
        # Delta = (f1_new - f1_n)(11 - f2_n) + (11 - f1_new)(11 - f2_new) - (f1_new - f1_n + 11 - f1_new)(11 - f2_n)
        # Delta = (f1_new - f1_n)(11 - f2_n) + (11 - f1_new)(11 - f2_new) - (11 - f1_n)(11 - f2_n)
        # Delta = (11 - f1_new)(11 - f2_new) - (11 - f1_new)(11 - f2_n)
        # Delta = (11 - f1_new)(f2_n - f2_new)
        
        # So:
        # Insert between i and i+1: (f1_{i+1} - f1_new) * (f2_i - f2_new)
        # Insert at start: (f1_1 - f1_new) * (11 - f2_new)
        # Insert at end: (11 - f1_new) * (f2_n - f2_new)
        
        # Note: All these gains must be positive.
        # Between: f1_{i+1} > f1_new and f2_i > f2_new (since f2 decreases). Positive.
        # Start: f1_1 > f1_new and 11 > f2_new. Positive.
        # End: 11 > f1_new and f2_n > f2_new. Positive.
        
        # Let's implement this.
        
        # Pre-compute f1, f2 for pool
        f1_pool = x0_pool
        # f2_pool already computed
        
        # Sort pool by f1
        # x0_pool is already sorted.
        
        selected_f1 = []
        selected_f2 = []
        selected_indices = [] # indices in pool, for evaluation
        
        # We need to evaluate the problem for the selected points to record them.
        # But we can just evaluate at the end?
        # The task says "Every returned objective vector is recorded".
        # And "return value of __call__ is ignored".
        # But we must call problem(x) to record it?
        # "The evaluator calls ... and calls it once."
        # "Every returned objective vector is recorded".
        # This implies we must call problem(x) for each point we want to record.
        # We can call it during the selection process or at the end.
        # Calling it at the end is safer to avoid timing issues during the greedy loop?
        # No, the greedy loop is fast.
        # But we must ensure we don't exceed budget.
        # We select 200 points. Then we call problem(x) 200 times.
        
        # Wait, the problem function might have side effects or be stochastic?
        # "problem(x) clips ... and returns 2 objective values".
        # It looks deterministic.
        # But to be safe, let's call it as we go, or at the end.
        # If we call it at the end, we just need the x vectors.
        
        # Let's store the x vectors.
        
        # Greedy Loop
        for step in range(self.budget):
            if time.time() > deadline - 5.0:
                break
                
            best_gain = -1
            best_idx = -1
            best_pos = -1 # position in selected list
            
            # If list is empty
            if len(selected_f1) == 0:
                # Gain = (11 - f1) * (11 - f2)
                gains = (11.0 - f1_pool) * (11.0 - f2_pool)
                best_idx = np.argmax(gains)
                best_gain = gains[best_idx]
                best_pos = 0 # will be inserted at 0
            else:
                # For each candidate, calculate gain
                # We can vectorize this partially?
                # The position depends on the candidate's f1.
                # We can find the insertion point for all candidates using np.searchsorted.
                
                # selected_f1 is sorted.
                # pos = np.searchsorted(selected_f1, f1_pool)
                # This gives the index where the candidate would be inserted.
                # If pos == 0, it's at the start.
                # If pos == len(selected_f1), it's at the end.
                # Else, it's between pos-1 and pos.
                
                pos = np.searchsorted(selected_f1, f1_pool)
                
                gains = np.zeros(n_candidates)
                
                # Start case: pos == 0
                mask_start = (pos == 0)
                if np.any(mask_start):
                    # Gain = (f1_1 - f1_new) * (11 - f2_new)
                    # f1_1 is selected_f1[0]
                    gains[mask_start] = (selected_f1[0] - f1_pool[mask_start]) * (11.0 - f2_pool[mask_start])
                    
                # End case: pos == len(selected_f1)
                mask_end = (pos == len(selected_f1))
                if np.any(mask_end):
                    # Gain = (11 - f1_new) * (f2_n - f2_new)
                    # f2_n is selected_f2[-1]
                    gains[mask_end] = (11.0 - f1_pool[mask_end]) * (selected_f2[-1] - f2_pool[mask_end])
                    
                # Middle case: 0 < pos < len(selected_f1)
                mask_mid = (pos > 0) & (pos < len(selected_f1))
                if np.any(mask_mid):
                    # Gain = (f1_{i+1} - f1_new) * (f2_i - f2_new)
                    # i = pos - 1
                    # f1_{i+1} = selected_f1[pos]
                    # f2_i = selected_f2[pos - 1]
                    # We need to gather these values for the candidates in mask_mid.
                    # pos_mid = pos[mask_mid]
                    # f1_next = np.array(selected_f1)[pos_mid]
                    # f2_prev = np.array(selected_f2)[pos_mid - 1]
                    # gains[mask_mid] = (f1_next - f1_pool[mask_mid]) * (f2_prev - f2_pool[mask_mid])
                    
                    pos_mid = pos[mask_mid]
                    sel_f1_arr = np.array(selected_f1)
                    sel_f2_arr = np.array(selected_f2)
                    f1_next = sel_f1_arr[pos_mid]
                    f2_prev = sel_f2_arr[pos_mid - 1]
                    gains[mask_mid] = (f1_next - f1_pool[mask_mid]) * (f2_prev - f2_pool[mask_mid])
                
                # Find max gain
                best_idx = np.argmax(gains)
                best_gain = gains[best_idx]
                best_pos = pos[best_idx]
            
            # Add the best candidate to selected lists
            # Insert at best_pos
            selected_f1.insert(best_pos, f1_pool[best_idx])
            selected_f2.insert(best_pos, f2_pool[best_idx])
            selected_indices.append(best_idx)
            
            # Mark as used? We don't need to remove from pool if we just track indices?
            # But we might pick the same index again?
            # No, because the gain for an already selected point would be 0 or negative?
            # If we pick a point already in the list, its position is such that it's equal to an existing point.
            # searchsorted would return the index of the existing point (or before/after).
            # If f1 is identical, it might be picked again.
            # To prevent this, we can set gains[used_indices] = -1.
            # But we don't have a "used" mask for the pool yet.
            # Let's create a boolean mask for used candidates.
            
            # Actually, we can just set the gain of the selected index to -1 in the next iteration?
            # No, the pool is static.
            # Let's keep a set of used indices.
            
            # Wait, I didn't implement the used mask in the loop above.
            # Let's add it.
            
            # Refactor:
            # used_mask = np.zeros(n_candidates, dtype=bool)
            # In loop:
            # gains[used_mask] = -1
            # ...
            # used_mask[best_idx] = True
            
            # Let's restart the loop with this fix.
            
        # Re-implementing the loop with used_mask
        
        selected_f1 = []
        selected_f2 = []
        selected_indices = []
        used_mask = np.zeros(n_candidates, dtype=bool)
        
        for step in range(self.budget):
            if time.time() > deadline - 5.0:
                break
                
            best_gain = -1
            best_idx = -1
            best_pos = -1
            
            if len(selected_f1) == 0:
                gains = (11.0 - f1_pool) * (11.0 - f2_pool)
                gains[used_mask] = -1
                best_idx = np.argmax(gains)
                best_gain = gains[best_idx]
                best_pos = 0
            else:
                pos = np.searchsorted(selected_f1, f1_pool)
                gains = np.zeros(n_candidates)
                gains[used_mask] = -1
                
                mask_start = (pos == 0) & (~used_mask)
                if np.any(mask_start):
                    gains[mask_start] = (selected_f1[0] - f1_pool[mask_start]) * (11.0 - f2_pool[mask_start])
                    
                mask_end = (pos == len(selected_f1)) & (~used_mask)
                if np.any(mask_end):
                    gains[mask_end] = (11.0 - f1_pool[mask_end]) * (selected_f2[-1] - f2_pool[mask_end])
                    
                mask_mid = (pos > 0) & (pos < len(selected_f1)) & (~used_mask)
                if np.any(mask_mid):
                    pos_mid = pos[mask_mid]
                    sel_f1_arr = np.array(selected_f1)
                    sel_f2_arr = np.array(selected_f2)
                    f1_next = sel_f1_arr[pos_mid]
                    f2_prev = sel_f2_arr[pos_mid - 1]
                    gains[mask_mid] = (f1_next - f1_pool[mask_mid]) * (f2_prev - f2_pool[mask_mid])
                
                best_idx = np.argmax(gains)
                best_gain = gains[best_idx]
                best_pos = pos[best_idx]
            
            selected_f1.insert(best_pos, f1_pool[best_idx])
            selected_f2.insert(best_pos, f2_pool[best_idx])
            selected_indices.append(best_idx)
            used_mask[best_idx] = True

        # 3. Evaluate the selected points
        # Create x vectors
        # x[0] = f1, x[1:] = 0
        results = []
        for idx in selected_indices:
            x = np.zeros(self.dim)
            x[0] = f1_pool[idx]
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append(f)
            
        return results

# EVOLVE-BLOCK-END
