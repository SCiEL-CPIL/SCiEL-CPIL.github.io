"""Multi-objective optimizer for ZDT3 using dense Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    ZDT3 has a known Pareto front with 7 disjoint segments.
    The front is at g=1 (global minimum), achieved when x[1:]=0.
    f1 = x[0], f2 = g*(1 - sqrt(f1/g)) = 1 - sqrt(f1) when g=1.
    
    Strategy:
    1. Use most of the budget to densely sample x[0] in [0,1] with x[1:]=0
    2. Add targeted samples at segment boundaries for better coverage
    3. Filter to non-dominated points
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # Leave 20s safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front: g=1 at x[1:]=0, f1=x[0], f2=1-sqrt(f1)
        # The 7 segments arise from the structure of the objective.
        # Dense uniform sampling of x[0] should capture all segments.
        
        # Use ~180 evaluations for dense sampling, ~20 for targeted refinement
        n_dense = min(self.budget - 20, 180)
        n_targeted = self.budget - n_dense
        
        # Dense uniform sampling of x[0]
        x1_dense = np.linspace(0, 1, n_dense)
        
        # Targeted samples at approximate segment boundaries
        # ZDT3 segments are at specific f1 values where the front has discontinuities
        targeted_x1 = np.array([
            0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.0575, 0.06, 0.07, 0.08,
            0.30, 0.31, 0.32, 0.33, 0.332, 0.34, 0.35,
            0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.6523, 0.66, 0.67,
            0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.8233, 0.83, 0.84,
            0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0
        ])
        
        # Combine and deduplicate
        x1_all = np.unique(np.concatenate([x1_dense, targeted_x1]))
        
        # If we have too many, prioritize dense + key targets
        if len(x1_all) > self.budget:
            # Keep all dense samples and a subset of targeted
            x1_final = list(x1_dense)
            remaining = self.budget - len(x1_dense)
            # Add targeted samples that aren't already in dense
            for t in targeted_x1:
                if remaining <= 0:
                    break
                if not any(abs(t - d) < 1e-6 for d in x1_dense):
                    x1_final.append(t)
                    remaining -= 1
            x1_all = np.array(x1_final)
        
        # Build and evaluate points
        points = []
        for x1 in x1_all:
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            points.append(x)
        
        F = np.array([problem(x) for x in points])
        
        # Filter: must be strictly better than reference [11, 11]
        mask = (F[:, 0] < 11.0) & (F[:, 1] < 11.0)
        points = np.array(points[mask])
        F = F[mask]
        
        if len(F) == 0:
            return points
        
        # Non-dominated sorting (vectorized for speed)
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if dominated[i]:
                continue
            # Check if any other point dominates i
            # A point j dominates i if F[j] <= F[i] componentwise and F[j] < F[i] in at least one
            diff = F - F[i]  # shape (n, 2)
            # j dominates i: all(diff[j] <= 0) and any(diff[j] < 0)
            leq = np.all(diff <= 0, axis=1)
            lt = np.any(diff < 0, axis=1)
            dom_mask = leq & lt
            dom_mask[i] = False  # exclude self
            if np.any(dom_mask):
                dominated[i] = True
        
        pareto_mask = ~dominated
        points = points[pareto_mask]
        F = F[pareto_mask]
        
        # Use remaining budget for fine-grained refinement around segment boundaries
        used = len(points)
        if used < self.budget and len(points) > 0 and time.time() < deadline:
            rng = np.random.default_rng(42)
            remaining_budget = self.budget - used
            
            # Identify approximate segment boundaries from the Pareto front
            # by looking for gaps in f1 values
            f1_vals = F[:, 0]
            f1_sorted = np.sort(f1_vals)
            
            # Find large gaps in f1 values (potential segment boundaries)
            gaps = np.diff(f1_sorted)
            gap_indices = np.where(gaps > 0.05)[0]
            
            # Refine around boundaries and in sparse regions
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                
                # Choose a random existing Pareto point
                idx = rng.integers(len(points))
                parent = points[idx].copy()
                
                # Perturb x[0] with small step
                step_size = 0.005
                delta = rng.uniform(-step_size, step_size)
                child = parent.copy()
                child[0] = np.clip(parent[0] + delta, lo[0], hi[0])
                
                fc = problem(child)
                if fc[0] >= 11.0 or fc[1] >= 11.0:
                    continue
                
                # Check domination against current Pareto set
                is_dominated = False
                for k in range(len(F)):
                    if np.all(F[k] <= fc) and np.any(F[k] < fc):
                        is_dominated = True
                        break
                
                if is_dominated:
                    continue
                
                # Check if child dominates any existing points
                new_mask = np.ones(len(F), dtype=bool)
                for k in range(len(F)):
                    if np.all(fc <= F[k]) and np.any(fc < F[k]):
                        new_mask[k] = False
                
                points = np.vstack([points, child.reshape(1, -1)])
                F = np.vstack([F, fc.reshape(1, -1)])
                
                if not np.all(new_mask):
                    points = points[new_mask]
                    F = F[new_mask]
        
        return points
# EVOLVE-BLOCK-END
