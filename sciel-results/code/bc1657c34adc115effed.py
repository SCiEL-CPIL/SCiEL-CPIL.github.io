import numpy as np
import time

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
        deadline = start_time + 290  # Safety margin for 330s limit

        # Track all evaluated points and their objective values
        X_all = []
        F_all = []
        
        def evaluate(x):
            """Evaluate a point and record it."""
            f = np.array(problem(x), dtype=np.float64)
            X_all.append(x.copy())
            F_all.append(f.copy())
            return f

        # Phase 1: Initial Sampling with Latin Hypercube (LHS)
        # LHS provides good coverage with fewer points than random.
        n_init = 100
        if n_init > budget:
            n_init = budget
        
        # Generate LHS points
        # Simple LHS: divide each dimension into n_init intervals, pick one random point in each.
        # To ensure coverage, we can use scipy.stats.qmc.LatinHypercube or manual.
        # Let's use a simple manual LHS for speed and control.
        rng = np.random.RandomState(42) # Fixed seed for reproducibility within the run, though global seed is set by evaluator
        # Actually, the evaluator sets np.random.seed. We should use that or our own.
        # Using np.random directly is fine.
        
        # Generate indices 0..n_init-1, shuffle them for each dimension
        indices = np.arange(n_init)
        X_init = np.zeros((n_init, dim))
        for d in range(dim):
            perm = np.random.permutation(n_init)
            # Uniformly distributed in [0, 1)
            # u = (perm + np.random.rand(n_init)) / n_init
            # But simpler: just use random points in each stratum
            X_init[:, d] = (perm + np.random.rand(n_init)) / n_init
        
        # Scale to bounds
        X_init = lo + X_init * (hi - lo)
        
        # Evaluate initial population
        for x in X_init:
            if time.time() > deadline:
                break
            evaluate(x)
        
        if time.time() > deadline:
            return None # Return ignored, but stop

        # Phase 2: Local Refinement with Weighted Sums and Nelder-Mead
        # We will use the remaining budget to refine promising regions.
        remaining_budget = budget - len(X_all)
        
        if remaining_budget <= 0:
            return None

        # Generate weight vectors for the 3 objectives
        # Use a grid on the simplex for uniform coverage of the front
        # For 3 objectives, weights (w1, w2, w3) such that w1+w2+w3=1, wi>=0
        # We can sample from a Dirichlet distribution or use a fixed set.
        # A fixed set of well-distributed weights is better for hypervolume.
        
        # Let's create a set of weight vectors.
        # For 3D, we can use points on the 2D simplex.
        # Number of weights to use: let's say 6 or 8, depending on budget.
        # Each Nelder-Mead run costs approx dim + maxiter evaluations.
        # dim=12. If maxiter=10, cost ~22.
        # If remaining_budget is 100, we can do ~4-5 runs.
        
        n_weights = 6
        # Generate weights using a grid or random
        # Let's use random Dirichlet for diversity, but seeded for reproducibility?
        # The evaluator sets np.random.seed, so np.random is deterministic per seed.
        
        weights = []
        # Fixed weights to ensure coverage:
        # (1,0,0), (0,1,0), (0,0,1) -> corners
        # (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5) -> edges
        # (1/3, 1/3, 1/3) -> center
        # This gives 7 points.
        fixed_weights = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1.0/3, 1.0/3, 1.0/3]
        ]
        
        # Select how many weights we can afford
        # Cost per NM run: roughly 15-20 evals.
        # Let's estimate max_evals_per_run = 15
        max_evals_per_run = 15
        affordable_runs = max(1, remaining_budget // max_evals_per_run)
        
        # Use up to affordable_runs weight vectors from the fixed list
        # Shuffle the weights to vary which ones are picked? Or just take the first N.
        # Taking the first N ensures the corners are always explored.
        weights_to_use = fixed_weights[:min(affordable_runs, len(fixed_weights))]
        
        # For each weight, find the best starting point from the initial population
        # and run Nelder-Mead.
        
        for w in weights_to_use:
            if time.time() > deadline:
                break
            
            # Calculate weighted sum for all initial points
            # F_init is the first n_init entries of F_all
            F_init = np.array(F_all[:n_init])
            scores = F_init @ np.array(w)
            best_idx = np.argmin(scores)
            x_start = X_all[best_idx].copy()
            
            # Define scalar objective function
            def scalar_obj(x, w=w):
                f = problem(x)
                return np.dot(w, f)
            
            # Run Nelder-Mead
            # We need to track evaluations to ensure we don't exceed budget.
            # scipy.optimize.minimize doesn't easily let us count evals without wrapping.
            # We can wrap the problem function to count.
            
            eval_count = [0]
            def counting_scalar_obj(x, w=w):
                if eval_count[0] >= max_evals_per_run:
                    # Return a large value to stop? Or just let it run and check later.
                    # Better: check budget before calling minimize.
                    pass
                f = problem(x)
                eval_count[0] += 1
                X_all.append(x.copy())
                F_all.append(f.copy())
                return np.dot(w, f)
            
            # Check if we have enough budget
            if len(X_all) + max_evals_per_run > budget:
                continue
                
            try:
                from scipy.optimize import minimize
                res = minimize(counting_scalar_obj, x_start, method='Nelder-Mead',
                               options={'maxiter': 10, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                pass

        return None
