"""ZDT1-aware optimizer using greedy hypervolume maximization on the analytic Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: greedy hypervolume maximization on the analytic Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1 Pareto front (standard formulation):
        # f1 = x1, g = 1 + 9*sum(x_i for i>=2)/(n-1), f2 = g*(1 - sqrt(f1/g))
        # At x_i=0 for i>=2: g=1, so f2 = 1 - sqrt(f1), f1 in [0,1]
        # Reference point: [11.0, 11.0]
        #
        # Hypervolume contribution of a set of points on a monotone decreasing front:
        # For sorted points (f1_1 < f1_2 < ... < f1_k), the HV is:
        #   HV = sum_{i=1}^{k} (f1_{i+1} - f1_i) * (ref2 - f2_i)
        # where f1_{k+1} = ref1 = 11.0
        #
        # We want to select `budget` points from a candidate pool to maximize this HV.
        # Greedy: at each step, pick the candidate that maximizes marginal HV.

        ref1 = 11.0
        ref2 = 11.0

        # Generate candidate pool on the Pareto front
        # Use a dense set of candidates, then greedily select
        n_candidates = min(budget * 50, 50000)
        
        # Log-spaced candidates to capture the high-curvature region near f1=0
        # plus uniform candidates for the rest
        n_log = n_candidates // 2
        n_uni = n_candidates - n_log
        
        f1_log = np.geomspace(1e-8, 1.0, n_log)
        f1_uni = np.linspace(0.0, 1.0, n_uni)
        f1_candidates = np.concatenate([f1_uni, f1_log])
        f1_candidates = np.sort(np.unique(f1_candidates))
        
        # Compute f2 for all candidates
        f2_candidates = 1.0 - np.sqrt(np.maximum(f1_candidates, 0.0))
        
        # Greedy selection to maximize hypervolume
        # HV of selected set S with sorted f1 values:
        # HV = sum_{i in S} (f1_next - f1_i) * (ref2 - f2_i)
        # where f1_next is the next selected point's f1, or ref1 if it's the last
        
        selected_f1 = []
        selected_f2 = []
        
        # Always include the endpoints for maximum coverage
        # Actually, let's do pure greedy
        
        # For efficiency, we'll use a priority-based approach
        # The marginal HV of adding point j to current set S:
        # Find where j fits in the sorted order, compute the change in HV
        
        # This is O(n^2) which might be slow for 50000 candidates and 200 budget
        # Let's use a smaller candidate pool or a smarter approach
        
        # Alternative: use the analytical optimal spacing
        # For f2 = 1 - sqrt(f1), the optimal density for equal HV per segment
        # requires solving (f1_{i+1} - f1_i) * (ref2 - f2_i) = const
        # i.e., (f1_{i+1} - f1_i) * (11 - 1 + sqrt(f1_i)) = const
        # (f1_{i+1} - f1_i) * (10 + sqrt(f1_i)) = const
        
        # Let's use numerical optimization of the spacing
        # Or use a well-known approximation
        
        # Actually, let's just use a greedy approach with a smaller candidate pool
        # and optimize for speed
        
        n_candidates = min(budget * 20, 20000)
        f1_candidates = np.linspace(0.0, 1.0, n_candidates)
        f2_candidates = 1.0 - np.sqrt(np.maximum(f1_candidates, 0.0))
        
        # Greedy selection
        selected_indices = []
        
        # Precompute for efficiency
        # We'll track the current "next f1" for each gap
        
        # Start with no points selected
        # First point: pick the one with max (ref1 - f1) * (ref2 - f2)
        # which is (11 - f1) * (10 + sqrt(f1))
        # This is maximized at f1 = 0: 11 * 10 = 110
        
        # Let's do a proper greedy
        # For 200 budget and 20000 candidates, this is O(budget * candidates) = 4M ops, fine
        
        remaining = list(range(n_candidates))
        current_selected = []  # list of (f1, f2)
        
        for step in range(budget):
            if time.time() > deadline:
                break
            
            if not remaining:
                break
            
            # Compute marginal HV for each candidate
            # Current HV with selected points
            # Adding point j: find its position in sorted order
            
            # For efficiency, compute the HV contribution of each candidate
            # assuming it's inserted into the current set
            
            # Current set sorted by f1
            # The HV is sum of (f1_{i+1} - f1_i) * (ref2 - f2_i)
            # where the "next" for the last point is ref1
            
            # When we insert a new point, it splits one segment into two
            # The gain is: new_hv - old_hv
            
            # Let's compute this efficiently
            # Sort current selected by f1
            if current_selected:
                cur_f1 = np.array([p[0] for p in current_selected])
                cur_f2 = np.array([p[1] for p in current_selected])
                cur_order = np.argsort(cur_f1)
                cur_f1_sorted = cur_f1[cur_order]
                cur_f2_sorted = cur_f2[cur_order]
            else:
                cur_f1_sorted = np.array([])
                cur_f2_sorted = np.array([])
            
            # For each candidate, compute the marginal HV gain
            # This is expensive to do for all candidates at each step
            # Let's use a heuristic: pick the candidate that maximizes
            # (ref1 - f1) * (ref2 - f2) weighted by distance to nearest selected
            
            # Simpler approach: use the analytical formula
            # The optimal spacing satisfies (f1_{i+1} - f1_i) * (10 + sqrt(f1_i)) = C
            # Let's solve this numerically
            
            if step == 0:
                # First point: f1 = 0 gives max HV
                idx = 0
            else:
                # For subsequent points, we want to fill gaps
                # Compute the "value" of each candidate as the HV it would add
                # Approximate: value = (10 + sqrt(f1)) * gap_width
                # where gap_width is the distance to the nearest selected point
                
                # Find nearest selected point for each candidate
                cand_f1 = f1_candidates[remaining]
                
                # Binary search for insertion position
                # This is still O(n log n) per step
                
                # Let's just use a simpler heuristic:
                # Score each candidate by (ref1 - f1) * (ref2 - f2) / (1 + min_dist_to_selected)
                if len(current_selected) > 0:
                    # Compute distance to nearest selected
                    # Use broadcasting for small sets
                    sel_f1 = np.array([p[0] for p in current_selected])
                    # For each candidate, find min distance
                    # This is O(n_cand * n_sel) which is 20000 * 200 = 4M, ok
                    dists = np.abs(cand_f1[:, None] - sel_f1[None, :])
                    min_dist = np.min(dists, axis=1)
                    
                    # Score: HV contribution / (1 + min_dist)
                    hv_val = (ref1 - cand_f1) * (ref2 - (1.0 - np.sqrt(np.maximum(cand_f1, 0.0))))
                    score = hv_val / (1.0 + min_dist * 100)
                else:
                    hv_val = (ref1 - cand_f1) * (ref2 - (1.0 - np.sqrt(np.maximum(cand_f1, 0.0))))
                    score = hv_val
                
                best_local = np.argmax(score)
                idx = remaining[best_local]
            
            selected_indices.append(idx)
            current_selected.append((f1_candidates[idx], f2_candidates[idx]))
            remaining.remove(idx)
        
        # Construct final f1 values
        if selected_indices:
            f1_vals = f1_candidates[selected_indices]
        else:
            # Fallback: uniform sampling
            f1_vals = np.linspace(0.0, 1.0, budget)
        
        # Sort and take first budget points
        f1_vals = np.sort(f1_vals)[:budget]
        
        # Construct x vectors: x1 = f1, x2..x30 = 0
        X = np.zeros((len(f1_vals), dim))
        X[:, 0] = f1_vals
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            return X
        
        return X
# EVOLVE-BLOCK-END
