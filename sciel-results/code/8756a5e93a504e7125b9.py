# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective Optimizer using Adaptive Sampling with Multi-Weight Nelder-Mead refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 280

    def _check_time(self):
        return time.time() > self.deadline

    def _evaluate(self, x, problem):
        if self.eval_count >= self.budget:
            return None
        self.eval_count += 1
        try:
            return problem(x)
        except Exception:
            return None

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        X_list = []
        F_list = []
        
        def add_point(x, f):
            if f is not None and len(f) == self.n_obj and np.all(np.isfinite(f)):
                x_clipped = np.clip(x, lo, hi)
                for i in range(len(X_list)):
                    if np.allclose(X_list[i], x_clipped, atol=1e-8):
                        return
                X_list.append(x_clipped)
                F_list.append(f)
        
        # Phase 1: Initial Sampling
        # Use a focused set of structured points + random to cover the space.
        # Budget is 200. Use ~40 points for initial exploration to leave more for refinement.
        n_initial = 40
        
        rng = np.random.default_rng(0)
        
        # Corners (8 points for dim=4)
        corners = []
        for i in range(2**self.dim):
            c = []
            for d in range(self.dim):
                c.append(hi[d] if (i >> d) & 1 else lo[d])
            corners.append(c)
        
        # Center
        center = [(lo[d] + hi[d])/2 for d in range(self.dim)]
        
        # Face centers (4 faces for dim=4, but we'll use axis-aligned midpoints)
        face_centers = []
        for d in range(self.dim):
            for val in [lo[d], hi[d]]:
                fc = [(lo[i] + hi[i])/2 if i != d else val for i in range(self.dim)]
                face_centers.append(fc)
        
        # Initial points list: corners + center + face centers
        init_points = corners + [center] + face_centers
        
        # Add random points to fill up to n_initial
        while len(init_points) < n_initial:
            p = rng.uniform(lo, hi)
            init_points.append(p)
        
        # Evaluate initial points
        for x in init_points:
            if self.eval_count >= self.budget or self._check_time():
                break
            f = self._evaluate(x, problem)
            if f is not None:
                add_point(x, f)
        
        if not F_list:
            x0 = (lo + hi) / 2
            f0 = self._evaluate(x0, problem)
            if f0 is not None: add_point(x0, f0)
            return X_list if X_list else [x0]
        
        F = np.array(F_list)
        X = np.array(X_list)
        
        # Phase 2: Iterative Local Refinement
        # We will perform several rounds of refinement.
        # In each round, we select a set of promising points and run Nelder-Mead with different weight vectors.
        
        # Weight vectors to cover the Pareto front more uniformly
        # For 3 objectives, use a denser set of weight vectors on the simplex
        
        weight_vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1.0/3, 1.0/3, 1.0/3],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.7, 0.1, 0.2],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
            [0.6, 0.3, 0.1],
            [0.3, 0.6, 0.1],
            [0.1, 0.3, 0.6],
        ]
        
        # We have ~160 evals left after initial 40.
        # Nelder-Mead in 4D: use maxfev=30 to allow better convergence.
        
        n_refine_rounds = 6  # Number of refinement rounds
        evals_per_run = 30   # Max evaluations per Nelder-Mead run
        
        for r in range(n_refine_rounds):
            if self.eval_count >= self.budget or self._check_time():
                break
            
            # Select starting points for this round.
            # We prioritize non-dominated points and those with low weighted sums.
            
            # Recalculate non-dominated set
            n = len(F)
            if n == 0: break
            
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if is_dominated[i]: continue
                for j in range(i+1, n):
                    if is_dominated[j]: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        is_dominated[j] = True
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break
            
            non_dom_indices = np.where(~is_dominated)[0]
            
            # If no non-dominated, use best by sum of normalized objectives
            if len(non_dom_indices) == 0:
                f_min = np.min(F, axis=0)
                f_max = np.max(F, axis=0)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1.0
                F_norm = (F - f_min) / f_range
                sums = np.sum(F_norm, axis=1)
                non_dom_indices = np.argsort(sums)[:5]
            
            # Select a few start points from non-dominated set
            # Shuffle to get different starts in different rounds
            rng.shuffle(non_dom_indices)
            
            n_starts = min(5, len(non_dom_indices))
            start_indices = non_dom_indices[:n_starts]
            
            for s_idx, idx in enumerate(start_indices):
                if self.eval_count >= self.budget or self._check_time():
                    break
                
                # Select a weight vector. Cycle through them.
                w_idx = (r * n_starts + s_idx) % len(weight_vectors)
                w = np.array(weight_vectors[w_idx])
                
                x_start = X[idx].copy()
                
                best_f = None
                best_x = x_start.copy()
                
                def objective(x, w=w):
                    nonlocal best_f, best_x
                    f = self._evaluate(x, problem)
                    if f is None:
                        return 1e6
                    val = np.dot(w, f)
                    if best_f is None or val < np.dot(w, best_f):
                        best_f = f
                        best_x = x.copy()
                    return val
                
                try:
                    res = minimize(
                        objective, 
                        x_start, 
                        method='Nelder-Mead', 
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 60, 'maxfev': evals_per_run, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                    
                    # Add the best point found during NM (already evaluated)
                    if best_f is not None:
                        add_point(best_x, best_f)
                            
                except Exception:
                    pass
            
            # Update F and X
            F = np.array(F_list)
            X = np.array(X_list)
        
        # Phase 3: Final polish pass - use remaining budget to refine the best points
        if self.eval_count < self.budget and not self._check_time():
            # Get non-dominated points
            n = len(F)
            if n > 0:
                is_dominated = np.zeros(n, dtype=bool)
                for i in range(n):
                    if is_dominated[i]: continue
                    for j in range(i+1, n):
                        if is_dominated[j]: continue
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            is_dominated[j] = True
                        elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            is_dominated[i] = True
                            break
                
                non_dom_indices = np.where(~is_dominated)[0]
                
                if len(non_dom_indices) > 0:
                    # Polish each non-dominated point with a tight NM
                    evals_left = self.budget - self.eval_count
                    n_polish = min(len(non_dom_indices), max(1, evals_left // 15))
                    
                    for idx in non_dom_indices[:n_polish]:
                        if self.eval_count >= self.budget or self._check_time():
                            break
                        
                        x_start = X[idx].copy()
                        best_f = None
                        best_x = x_start.copy()
                        
                        def polish_obj(x):
                            nonlocal best_f, best_x
                            f = self._evaluate(x, problem)
                            if f is None:
                                return 1e6
                            val = np.sum(f)  # Use sum as a generic objective for polishing
                            if best_f is None or val < np.sum(best_f):
                                best_f = f
                                best_x = x.copy()
                            return val
                        
                        try:
                            minimize(
                                polish_obj,
                                x_start,
                                method='Nelder-Mead',
                                bounds=list(zip(lo, hi)),
                                options={'maxiter': 30, 'maxfev': 15, 'xatol': 1e-7, 'fatol': 1e-7}
                            )
                            if best_f is not None:
                                add_point(best_x, best_f)
                        except Exception:
                            pass
        
        return X_list
# EVOLVE-BLOCK-END
