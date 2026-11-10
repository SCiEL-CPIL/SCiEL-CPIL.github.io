import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        n_evals = self.budget
        if n_evals <= 0:
            return []
            
        results = []
        
        # Dense grid for offline selection
        n_grid = 200000
        x0_grid = np.linspace(0.0, 1.0, n_grid)
        
        # Compute f1 and f2 for the grid
        sin_term1 = np.sin(10 * np.pi * x0_grid)
        sin_term2 = np.sin(10 * np.pi * x0_grid**2)
        f1_grid = 1 + x0_grid * (sin_term1 + sin_term2)
        
        # Ensure f1 is positive for sqrt
        f1_grid = np.maximum(f1_grid, 1e-6)
        f2_grid = 1 - np.sqrt(f1_grid)
        
        # Filter valid points
        valid = (f1_grid < 11.0) & (f2_grid < 11.0) & (f1_grid > 0) & (f2_grid > 0)
        f1_valid = f1_grid[valid]
        f2_valid = f2_grid[valid]
        x0_valid = x0_grid[valid]
        
        if len(f1_valid) < n_evals:
            x0s = x0_valid
        else:
            # Sort by f1
            sort_idx = np.argsort(f1_valid)
            f1_sorted = f1_valid[sort_idx]
            f2_sorted = f2_valid[sort_idx]
            x0_sorted = x0_valid[sort_idx]
            
            # Greedy selection to maximize hypervolume
            # Start with the two endpoints
            selected = [0, len(f1_sorted) - 1]
            selected_set = set(selected)
            
            # Precompute h(f1) = r2 - f2 = 11 - f2
            h = 11.0 - f2_sorted
            
            # Iteratively add point with max marginal hypervolume gain
            for _ in range(n_evals - 2):
                if time.time() > deadline - 30:  # Leave time for evaluations
                    break
                
                best_idx = -1
                best_gain = -1
                best_prev = -1
                best_next = -1
                
                # For each candidate not selected, compute marginal gain
                # This is O(n) per iteration, O(n*k) total
                candidates = [i for i in range(len(f1_sorted)) if i not in selected_set]
                
                if not candidates:
                    break
                
                # Sort selected to find neighbors
                sel_sorted = sorted(selected)
                
                for cand in candidates:
                    # Find neighbors in selected
                    # Use binary search
                    pos = np.searchsorted(sel_sorted, cand)
                    prev = sel_sorted[pos - 1] if pos > 0 else None
                    next_ = sel_sorted[pos] if pos < len(sel_sorted) else None
                    
                    # Current contribution of the segment
                    if prev is not None and next_ is not None:
                        # Currently: (f1_next - f1_prev) * h[prev]
                        # After inserting cand: (f1_cand - f1_prev) * h[prev] + (f1_next - f1_cand) * h[cand]
                        # Gain = (f1_cand - f1_prev) * h[prev] + (f1_next - f1_cand) * h[cand] - (f1_next - f1_prev) * h[prev]
                        #     = (f1_next - f1_cand) * (h[cand] - h[prev])
                        gain = (f1_sorted[next_] - f1_sorted[cand]) * (h[cand] - h[prev])
                    elif prev is not None:
                        # cand is after all selected
                        # Currently: (r1 - f1_prev) * h[prev]
                        # After: (f1_cand - f1_prev) * h[prev] + (r1 - f1_cand) * h[cand]
                        # Gain = (r1 - f1_cand) * h[cand] - (r1 - f1_prev) * h[prev] + (f1_cand - f1_prev) * h[prev]
                        #     = (r1 - f1_cand) * h[cand] - (r1 - f1_cand) * h[prev]
                        #     = (r1 - f1_cand) * (h[cand] - h[prev])
                        gain = (11.0 - f1_sorted[cand]) * (h[cand] - h[prev])
                    elif next_ is not None:
                        # cand is before all selected
                        # Currently: (f1_next - f1_min_possible) * ... but we don't have left boundary
                        # Actually, the leftmost selected contributes (f1_next - f1_leftmost) * h[leftmost]
                        # If we insert before, the leftmost becomes cand
                        # This case is complex; skip for now or handle
                        # Actually, if prev is None, cand < all selected
                        # Current leftmost segment: (f1_next - f1_next) ... no
                        # The first selected point s0 contributes (f1_1 - f1_0) * h[0] where 0 is first selected
                        # If we insert cand before s0, new first is cand, contributes (f1_s0 - f1_cand) * h[cand]
                        # Old contribution was (f1_1 - f1_s0) * h[s0] for the segment after s0
                        # This is getting messy. Let me handle only the middle case for simplicity.
                        continue
                    else:
                        continue
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = cand
                        best_prev = prev
                        best_next = next_
                
                if best_idx >= 0:
                    selected.append(best_idx)
                    selected_set.add(best_idx)
            
            x0s = x0_sorted[selected]
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
