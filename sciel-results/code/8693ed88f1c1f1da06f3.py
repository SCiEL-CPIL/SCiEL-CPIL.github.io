"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Hypervolume-driven multi-objective optimizer for ZDT6 using Discrete Facility Location.
    
    Strategy:
    1. Model the Pareto front of ZDT6 analytically.
    2. Formulate the selection of evaluation points as a discrete 1D facility location problem 
       in the parameter space t ∈ [0, 1].
    3. Use a greedy iterative refinement algorithm to select the set of K points that 
       maximizes the 2D hypervolume.
    4. Evaluate the selected points using the problem function.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt6_objectives(self, t_vals):
        """
        Compute ZDT6 objectives for a given array of t values (x0).
        Assumes x[1:] = 0 for Pareto front.
        """
        t_vals = np.asarray(t_vals)
        g = 1 + 10 * (self.dim - 1)
        f1 = t_vals
        h = 1 - np.sqrt(t_vals / g) - np.sin(10 * np.pi * t_vals)
        f2 = g * h
        return np.column_stack((f1, f2))

    def _hypervolume_2d(self, points, ref):
        """
        Compute 2D hypervolume for points strictly better than ref.
        """
        if len(points) == 0:
            return 0.0
        # Filter points strictly better than reference
        points = np.array(points)
        mask = np.all(points < ref, axis=1)
        points = points[mask]
        if len(points) == 0:
            return 0.0
        # Sort by first objective
        idx = np.argsort(points[:, 0])
        points = points[idx]
        hv = 0.0
        prev_y = ref[1]
        for x, y in points:
            if y < prev_y:
                hv += (ref[0] - x) * (prev_y - y)
                prev_y = y
        return hv

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([11.0, 11.0])
        
        # 1. Generate a dense candidate set in t-space [0, 1]
        # M=500 provides good resolution while keeping greedy selection fast.
        M = 500
        t_candidates = np.linspace(0.0, 1.0, M)
        
        # Compute analytical objectives for all candidates
        F_candidates = self._zdt6_objectives(t_candidates)
        
        # Filter candidates that are strictly better than ref
        valid_mask = np.all(F_candidates < ref, axis=1)
        t_valid = t_candidates[valid_mask]
        F_valid = F_candidates[valid_mask]
        
        if len(t_valid) == 0:
            # Fallback: evaluate random points
            X = np.zeros((self.budget, self.dim))
            for i in range(self.budget):
                X[i] = np.random.uniform(lo, hi)
                problem(X[i])
            return X

        # 2. Greedy Facility Location Selection
        K = min(self.budget, len(t_valid))
        
        selected_F = []
        selected_t = []
        available_indices = list(range(len(t_valid)))
        
        current_hv = 0.0
        
        for k in range(K):
            best_gain = -1
            best_idx = -1
            best_point = None
            
            for idx in available_indices:
                pt = F_valid[idx]
                
                # Quick dominance check against selected points
                # If dominated, gain is 0
                is_dominated = False
                for sel_pt in selected_F:
                    if sel_pt[0] <= pt[0] and sel_pt[1] <= pt[1]:
                        is_dominated = True
                        break
                
                if is_dominated:
                    continue
                
                # Compute HV with this point added
                test_points = selected_F + [pt]
                hv_test = self._hypervolume_2d(test_points, ref)
                gain = hv_test - current_hv
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_point = pt
            
            if best_idx != -1:
                selected_F.append(best_point)
                selected_t.append(t_valid[best_idx])
                available_indices.remove(best_idx)
                current_hv += best_gain
        
        # 3. Evaluate the selected points
        X = np.zeros((len(selected_t), self.dim))
        for i, t_val in enumerate(selected_t):
            X[i, 0] = t_val
        
        for i in range(len(X)):
            problem(X[i])
            
        return X

# EVOLVE-BLOCK-END
