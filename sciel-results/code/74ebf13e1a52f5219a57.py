import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        # We need to reserve evaluations for the returned points.
        # Let's aim to return 50 points.
        # So internal budget is budget - 50.
        max_return = 50
        internal_budget = max(10, budget - max_return)
        
        eval_count = 0
        
        def eval_point(x):
            nonlocal eval_count
            if eval_count >= internal_budget:
                # Return a dummy value or raise? 
                # We should stop calling if over budget.
                # But minimize might call it. 
                # We need a safe wrapper.
                pass
            # Clip to bounds
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            eval_count += 1
            return f

        # 1. Sobol Initialization
        n_sobol = min(60, internal_budget)
        if n_sobol > 0:
            sobol = qmc.Sobol(d=dim, scramble=True)
            sobol_points = sobol.random(n_sobol)
            # Scale to bounds
            sobol_points = lo + sobol_points * (hi - lo)
            
            X_sobol = sobol_points
            F_sobol = np.array([eval_point(x) for x in X_sobol])
            
            # Check if we ran out of budget
            if eval_count >= internal_budget:
                # Just return the best points from Sobol
                return self._select_best(X_sobol, F_sobol, max_return)

        else:
            X_sobol = np.empty((0, dim))
            F_sobol = np.empty((0, n_obj))

        # 2. Local Refinement
        # Find non-dominated points in Sobol set
        nd_indices = self._non_dominated_indices(F_sobol)
        
        if len(nd_indices) == 0:
            # Fallback: return random points
            X_rand = lo + (hi - lo) * np.random.random((min(max_return, internal_budget), dim))
            # Evaluate them? No, we are out of internal budget?
            # If n_sobol == 0, we have full budget.
            # But we assumed n_sobol > 0.
            # If no non-dominated (impossible unless all same), return random.
            return X_rand[:max_return]

        # Select start points for Nelder-Mead
        # We want to explore different parts of the front.
        # Use weighted sums to select diverse points.
        weights_list = [
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.4, 0.4, 0.2],
            [0.3, 0.3, 0.4]
        ]
        
        # Adjust weights if n_obj != 3
        if n_obj != 3:
            weights_list = []
            for i in range(min(5, n_obj)):
                w = np.ones(n_obj) * 0.1
                w[i] = 0.7
                weights_list.append(w)
        
        start_points = []
        for w in weights_list:
            # Find point in nd_indices that minimizes weighted sum
            if len(nd_indices) == 0:
                break
            w = np.array(w)
            if len(w) != n_obj:
                w = np.ones(n_obj) / n_obj
            weighted_sums = F_sobol[nd_indices] @ w
            best_idx_in_nd = nd_indices[np.argmin(weighted_sums)]
            start_points.append(X_sobol[best_idx])
            
        # Deduplicate start points
        unique_starts = []
        for sp in start_points:
            is_dup = False
            for usp in unique_starts:
                if np.allclose(sp, usp):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(sp)
        
        X_local = []
        F_local = []
        
        n_local = len(unique_starts)
        if n_local > 0:
            remaining_budget = internal_budget - eval_count
            if remaining_budget > 0:
                budget_per_local = remaining_budget // n_local
                if budget_per_local < 5:
                    budget_per_local = 5 # Minimum for NM to do anything
                    # If even this is too much, reduce n_local
                    n_local = max(1, remaining_budget // 5)
                    unique_starts = unique_starts[:n_local]
                    budget_per_local = remaining_budget // n_local

                for sp in unique_starts:
                    if eval_count >= internal_budget:
                        break
                    
                    # Define objective for Nelder-Mead
                    # Use a fixed weight vector for each local search to explore different fronts
                    # Or just minimize sum of objectives?
                    # Let's use a weight vector based on the index or just equal weights
                    # To ensure diversity, let's use the weights from the selection step
                    # But we lost the mapping. Let's just use equal weights for simplicity,
                    # or rotate weights.
                    
                    # Better: Use the weight that selected this point.
                    # We need to map sp back to weight.
                    # For now, use equal weights.
                    w_nm = np.ones(n_obj) / n_obj
                    
                    def objective(x):
                        if eval_count >= internal_budget:
                            return 1e10
                        f = eval_point(x)
                        return np.dot(w_nm, f)
                    
                    try:
                        res = minimize(objective, sp, method='Nelder-Mead', 
                                       options={'maxiter': int(budget_per_local / 2), 'xatol': 1e-4, 'fatol': 1e-4})
                        if res.success or res.fun < 1e9:
                            x_opt = res.x
                            f_opt = eval_point(x_opt) # One more eval to record exact F
                            X_local.append(x_opt)
                            F_local.append(f_opt)
                    except Exception:
                        pass

        # Combine Sobol and Local results
        if len(X_local) > 0:
            X_all = np.vstack([X_sobol, np.array(X_local)])
            F_all = np.vstack([F_sobol, np.array(F_local)])
        else:
            X_all = X_sobol
            F_all = F_sobol

        # Select best points to return
        best_X, best_F = self._select_best(X_all, F_all, max_return)
        
        return best_X

    def _non_dominated_indices(self, F):
        n = F.shape[0]
        if n == 0:
            return []
        nd = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd.append(i)
        return nd

    def _select_best(self, X, F, k):
        """
        Select k non-dominated points from X, F.
        If fewer than k non-dominated, return all non-dominated and fill with random?
        Or just return the non-dominated ones.
        """
        if len(X) == 0:
            return X
        
        nd_indices = self._non_dominated_indices(F)
        
        if len(nd_indices) <= k:
            return X[nd_indices]
        
        # If more than k non-dominated, select k most diverse
        # Use crowding distance or simple hypervolume contribution heuristic
        # For simplicity, select k points that are evenly spaced in objective space
        
        F_nd = F[nd_indices]
        
        # Normalize objectives
        f_min = F_nd.min(axis=0)
        f_max = F_nd.max(axis=0)
        f_range = f_max - f_min
        f_range[f_range == 0] = 1.0
        F_norm = (F_nd - f_min) / f_range
        
        # Select points based on distance to centroid or random sampling with replacement?
        # Let's use a greedy hypervolume maximization (approximate)
        # Or just pick k random non-dominated points?
        # Random might be bad.
        # Let's pick k points that maximize the minimum pairwise distance in normalized space.
        
        # Simple greedy:
        selected = []
        # Start with the point closest to ideal (0,0,0)
        dists = np.sum(F_norm**2, axis=1)
        first_idx = np.argmin(dists)
        selected.append(first_idx)
        
        while len(selected) < k:
            # Find point in nd_indices not in selected that maximizes min distance to selected
            best_idx = -1
            best_min_dist = -1
            for i in range(len(nd_indices)):
                if i in selected:
                    continue
                min_dist = np.inf
                for s in selected:
                    d = np.linalg.norm(F_norm[i] - F_norm[s])
                    if d < min_dist:
                        min_dist = d
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            
        return X[nd_indices[selected]]
