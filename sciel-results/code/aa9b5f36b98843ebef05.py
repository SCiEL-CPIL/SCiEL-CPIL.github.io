# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer for RE37 (4 variables, 3 objectives, bounds [0,1]).
    Strategy:
      1. Phase 1: Evolutionary Strategy (ES) with Pareto selection for global exploration.
         - Uses Latin Hypercube Sampling for initialization.
         - Mutates non-dominated parents.
         - Maintains an archive of non-dominated points.
      2. Phase 2: Local Polish using Weighted-Sum Scalarization + Nelder-Mead.
         - Identifies gaps in the current Pareto front.
         - Runs local searches from the best points to refine the front.
      3. Deadline guard to stay within 300s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Fixed seed for reproducibility within the call, though evaluator seeds np.random
        # We use a local RNG for internal consistency if needed, but problem() might use global state.
        # The task says: "evaluator calls np.random.seed(seed)...". 
        # We should rely on the global np.random state being set, or create our own.
        # Creating our own is safer for the internal logic if we want deterministic behavior independent of external calls,
        # but the problem function itself might use np.random. 
        # However, the task says "problem(x) clips... and returns". It doesn't say problem uses random.
        # So we can use our own RNG for search steps.
        rng = np.random.RandomState(42)

        # Archive of (x, f)
        archive_x = []
        archive_f = []
        eval_count = 0

        def is_dominated(f, archive_f_list):
            """Check if f is dominated by any point in archive_f_list."""
            for af in archive_f_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def add_to_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points."""
            nonlocal archive_x, archive_f
            # Check if new point is dominated
            if is_dominated(f, archive_f):
                return
            
            # Remove points from archive that are dominated by new point
            new_archive_x = []
            new_archive_f = []
            for ax, af in zip(archive_x, archive_f):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_x.append(ax)
                    new_archive_f.append(af)
            
            new_archive_x.append(x)
            new_archive_f.append(f)
            
            archive_x = new_archive_x
            archive_f = new_archive_f

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            if time.time() > deadline:
                return None
            fx = np.array(problem(x), dtype=float)
            eval_count += 1
            return fx

        # --- Phase 1: Evolutionary Strategy ---
        # Use 70% of budget for exploration
        es_budget = int(budget * 0.7)
        
        # Initialize with Latin Hypercube Sampling
        pop_size = 20
        n_init = min(pop_size, es_budget)
        
        # LHS generation
        lhs_points = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            lhs_points[:, d] = (perm + rng.random(n_init)) / n_init
        lhs_points = lo + lhs_points * (hi - lo)
        
        for i in range(n_init):
            x = lhs_points[i]
            f = evaluate(x)
            if f is None:
                break
            add_to_archive(x, f)
            
        # Evolutionary Loop
        # We maintain a list of non-dominated points in the archive.
        # We pick a parent from the archive (or a random point if archive is small/empty)
        # and mutate it.
        
        # To avoid stalling, we can also keep a "current population" of recent non-dominated points
        # for faster parent selection.
        
        while eval_count < es_budget and time.time() < deadline:
            if not archive_x:
                break
                
            # Select parent: Uniform from archive
            idx = rng.randint(len(archive_x))
            parent_x = archive_x[idx]
            
            # Mutation: Gaussian noise
            # Scale step size: start larger, decrease? Or fixed?
            # Fixed step of 0.1 * range seems reasonable for [0,1] bounds.
            step = 0.1
            child_x = parent_x + step * rng.standard_normal(dim)
            child_x = np.clip(child_x, lo, hi)
            
            f_child = evaluate(child_x)
            if f_child is None:
                break
            
            add_to_archive(child_x, f_child)

        # --- Phase 2: Local Polish ---
        # Use remaining budget to refine the front.
        # Generate weight vectors to cover the front.
        
        # Get current front
        if not archive_f:
            # Fallback if something went wrong
            return lo

        front_f = np.array(archive_f)
        front_x = np.array(archive_x)
        
        # Identify "gaps" or just use a set of standard weight vectors
        # Since n_obj=3, we can use a grid of weights.
        # But we only have limited budget.
        # Let's pick 5 weight vectors that are diverse.
        
        # Simple strategy: Use the vertices of the simplex and midpoints.
        # For 3 objectives: (1,0,0), (0,1,0), (0,0,1), (0.5, 0.25, 0.25), etc.
        # However, the objectives might not be convex.
        # Better: Use weights derived from the current front points' normals?
        # Or just a fixed set of 10 weights.
        
        weights = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.3, 0.4],
            [0.2, 0.3, 0.5]
        ]
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # For each weight, find the best starting point in the archive
        # and run Nelder-Mead.
        
        n_polish = min(10, budget - eval_count)
        
        if n_polish > 0:
            # Allocate budget per polish run
            evals_per_polish = max(5, n_polish // len(weights))
            
            for w in weights:
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Find best start: minimize w . f(x) over archive
                if len(front_f) == 0:
                    break
                scores = np.dot(front_f, w)
                best_idx = np.argmin(scores)
                x0 = front_x[best_idx].copy()
                
                def objective(x):
                    nonlocal eval_count
                    if time.time() > deadline:
                        return 1e10
                    if eval_count >= budget:
                        return 1e10
                    fx = np.array(problem(x), dtype=float)
                    eval_count += 1
                    return np.dot(w, fx)
                
                try:
                    res = minimize(objective, x0, method='Nelder-Mead',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 30, 'xatol': 1e-6, 'fatol': 1e-8})
                    
                    if res.success or res.fun < 1e9:
                        x_best = np.clip(res.x, lo, hi)
                        # Re-evaluate to get exact f
                        f_best = evaluate(x_best)
                        if f_best is not None:
                            add_to_archive(x_best, f_best)
                except Exception:
                    pass

        # Final check: ensure we return something valid
        if not archive_x:
            return lo
            
        return archive_x[0]

# EVOLVE-BLOCK-END
