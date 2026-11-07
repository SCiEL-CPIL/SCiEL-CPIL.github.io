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
        
        # Face centers (4 faces x 2 = 8 points for dim=4, but we'll pick a few)
        face_centers = []
        for d in range(self.dim):
            for val in [lo[d], hi[d]]:
                fc = [(lo[i] + hi[i])/2 for i in range(self.dim)]
                fc[d] = val
                face_centers.append(fc)
        
        # Initial points list: corners + center + some face centers + random
        init_points = corners + [center] + face_centers[:4]
        
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
        
        # Weight vectors to cover the Pareto front
        # For 3 objectives, we use a diverse set of weight vectors.
        
        # Generate weight vectors:
        # 1. Vertices of the simplex: (1,0,0), (0,1,0), (0,0,1)
        # 2. Edge midpoints: (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
        # 3. Center: (1/3, 1/3, 1/3)
        # 4. Additional points for better coverage along edges
        
        weight_vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1.0/3, 1.0/3, 1.0/3],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.3, 0.6, 0.1],
            [0.1, 0.3, 0.6],
        ]
        
        # Add some random weight vectors for diversity
        for _ in range(6):
            w = rng.dirichlet(np.ones(self.n_obj))
            weight_vectors.append(w)
        
        # We have ~120 evals left.
        # Nelder-Mead in 4D typically uses ~50-100 evals per run if not constrained.
        # We'll constrain maxfev to ~20-30 per run to allow multiple runs.
        
        n_refine_rounds = 6  # Number of refinement rounds
        evals_per_run = 22   # Max evaluations per Nelder-Mead run
        
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
            
            n_starts = min(6, len(non_dom_indices))
            start_indices = non_dom_indices[:n_starts]
            
            for s_idx, idx in enumerate(start_indices):
                if self.eval_count >= self.budget or self._check_time():
                    break
                
                # Select a weight vector. Cycle through them.
                w_idx = (r * n_starts + s_idx) % len(weight_vectors)
                w = np.array(weight_vectors[w_idx])
                
                x_start = X[idx].copy()
                
                def objective(x, w=w):
                    f = self._evaluate(x, problem)
                    if f is None:
                        return 1e6
                    return np.dot(w, f)
                
                try:
                    res = minimize(
                        objective, 
                        x_start, 
                        method='Nelder-Mead', 
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 40, 'maxfev': evals_per_run, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                    
                    # Evaluate the final result to get the vector
                    if res.x is not None:
                        f_final = self._evaluate(res.x, problem)
                        if f_final is not None:
                            add_point(res.x, f_final)
                            
                except Exception:
                    pass
            
            # Update F and X
            F = np.array(F_list)
            X = np.array(X_list)
        
        # Phase 3: High-precision polish on the best non-dominated points
        if self.eval_count < self.budget and not self._check_time():
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
                    # Polish the top 3-5 non-dominated points with tighter tolerance
                    rng.shuffle(non_dom_indices)
                    n_polish = min(5, len(non_dom_indices))
                    
                    for idx in non_dom_indices[:n_polish]:
                        if self.eval_count >= self.budget or self._check_time():
                            break
                        
                        x_start = X[idx].copy()
                        
                        # Use a balanced weight for polishing
                        w = np.ones(self.n_obj) / self.n_obj
                        
                        def polish_obj(x, w=w):
                            f = self._evaluate(x, problem)
                            if f is None:
                                return 1e6
                            return np.dot(w, f)
                        
                        try:
                            res = minimize(
                                polish_obj,
                                x_start,
                                method='Nelder-Mead',
                                bounds=list(zip(lo, hi)),
                                options={'maxiter': 80, 'maxfev': 30, 'xatol': 1e-8, 'fatol': 1e-8}
                            )
                            
                            if res.x is not None:
                                f_final = self._evaluate(res.x, problem)
                                if f_final is not None:
                                    add_point(res.x, f_final)
                        except Exception:
                            pass
        
        return X_list
# EVOLVE-BLOCK-END
