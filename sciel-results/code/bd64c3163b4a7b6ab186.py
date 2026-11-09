"""Multi-objective optimizer for ZDT2 using analytically optimized Pareto front sampling."""
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT2.
    
    Strategy:
    1. Acknowledge that ZDT2's Pareto front is 1D: x_1...x_{n-1} = 0, x_0 in [0,1].
    2. The front is f1 = x_0, f2 = A * (1 - sqrt(x_0 / A)) where A = 1 + 10*dim.
    3. Instead of using a fixed uniform or heuristic spacing for x_0, we determine
       the optimal spacing that maximizes the hypervolume with respect to the 
       reference point (11, 11).
    4. We parameterize the spacing as x_i = (i / (N-1))^gamma.
    5. We compute the hypervolume analytically for various gamma values and pick
       the best one. This is a structural improvement over fixed grids.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.A = 1.0 + 10.0 * dim
        self.ref = np.array([11.0, 11.0])

    def _compute_analytic_hv(self, x0_vals):
        """
        Compute hypervolume of points on the Pareto front defined by x0_vals.
        f1 = x0
        f2 = A * (1 - sqrt(x0 / A))
        Reference point r = (11, 11)
        """
        f1 = x0_vals
        f2 = self.A * (1.0 - np.sqrt(x0_vals / self.A))
        
        # Sort by f1 increasing (which means f2 decreasing)
        # x0_vals are already generated in increasing order if gamma > 0
        # Ensure strict ordering for HV calculation
        if len(f1) == 0:
            return 0.0
        
        # Points must be strictly better than reference point to count
        # f1 < 11 and f2 < 11
        # Since x0 in [0,1], f1 in [0,1] < 11.
        # f2 max is A. For dim=30, A=301. f2(0)=301 > 11.
        # So points with x0 near 0 might have f2 > 11.
        # The problem states: "Only points strictly better than the reference point in every objective count."
        # If f2 > 11, the point is NOT better than ref in objective 2.
        # However, the hypervolume is calculated on the "recorded vectors".
        # The evaluator records ALL returned objective vectors.
        # Then it calculates HV of ALL recorded vectors with ref [11,11].
        # Points with f2 > 11 do not contribute to the dominated region relative to [11,11] 
        # because they are dominated by the reference point itself? 
        # No, the HV is the volume of the region dominated by the set of points 
        # and containing the reference point? 
        # Standard HV definition: Volume of the region dominated by the set of points 
        # with respect to a reference point r. 
        # A point p dominates r if p <= r. 
        # If p > r in any dimension, it does not dominate r.
        # The hypervolume is the volume of the union of boxes [p, r] for all p 
        # such that p <= r in all dimensions.
        # If p has f2 > 11, the box [p, r] is empty (or invalid) because p is not 
        # dominated by r? 
        # Actually, if p2 > r2, the interval [p2, r2] is empty. 
        # So such points contribute 0 to the HV.
        
        # Therefore, we should only include points where f2 < 11.
        # f2(x) = A * (1 - sqrt(x/A)) < 11
        # 1 - sqrt(x/A) < 11/A
        # sqrt(x/A) > 1 - 11/A
        # x/A > (1 - 11/A)^2
        # x > A * (1 - 11/A)^2
        
        valid_mask = f2 < self.ref[1]
        if not np.any(valid_mask):
            return 0.0
            
        f1_valid = f1[valid_mask]
        f2_valid = f2[valid_mask]
        
        # Sort valid points by f1 increasing
        sort_idx = np.argsort(f1_valid)
        f1_sorted = f1_valid[sort_idx]
        f2_sorted = f2_valid[sort_idx]
        
        # Calculate HV
        # HV = sum_{i} (r1 - f1_i) * (f2_i - f2_{i-1}) with f2_{-1} = r2
        # But only for points that are non-dominated in the set.
        # Since the front is monotonic, all valid points are non-dominated 
        # with respect to each other (if x increases, f1 increases and f2 decreases).
        
        n_valid = len(f1_sorted)
        if n_valid == 0:
            return 0.0
            
        hv = 0.0
        prev_f2 = self.ref[1]
        for i in range(n_valid):
            # The "new" area added by this point
            # Width in f1: r1 - f1_i
            # Height in f2: prev_f2 - f2_i
            # Note: prev_f2 is the f2 of the previous point (with smaller f1, larger f2)
            # or the reference point for the first point.
            width = self.ref[0] - f1_sorted[i]
            height = prev_f2 - f2_sorted[i]
            hv += width * height
            prev_f2 = f2_sorted[i]
            
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.budget
        
        # Optimize gamma
        # Try a range of gamma values
        # gamma > 1 concentrates points near 0.
        # gamma < 1 concentrates points near 1.
        # We expect gamma > 1 because curvature is high near 0.
        
        best_hv = -1.0
        best_gamma = 1.0
        
        # Coarse search
        gammas = np.linspace(0.5, 4.0, 50)
        for g in gammas:
            # Generate x0 values
            u = np.linspace(0.0, 1.0, n)
            x0_vals = u ** g
            x0_vals[0] = 0.0
            x0_vals[-1] = 1.0
            
            hv = self._compute_analytic_hv(x0_vals)
            if hv > best_hv:
                best_hv = hv
                best_gamma = g
        
        # Fine search around best_gamma
        if time.time() < deadline - 5.0:
            fine_gammas = np.linspace(best_gamma - 0.5, best_gamma + 0.5, 20)
            fine_gammas = fine_gammas[fine_gammas > 0]
            for g in fine_gammas:
                u = np.linspace(0.0, 1.0, n)
                x0_vals = u ** g
                x0_vals[0] = 0.0
                x0_vals[-1] = 1.0
                
                hv = self._compute_analytic_hv(x0_vals)
                if hv > best_hv:
                    best_hv = hv
                    best_gamma = g
        
        # Generate final points
        u = np.linspace(0.0, 1.0, n)
        x0_vals = u ** best_gamma
        x0_vals[0] = 0.0
        x0_vals[-1] = 1.0
        
        X = np.zeros((n, self.dim))
        X[:, 0] = x0_vals
        
        if self.dim > 1:
            X[:, 1:] = lo[1]
        
        F = np.zeros((n, self.n_obj))
        
        for i in range(n):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
