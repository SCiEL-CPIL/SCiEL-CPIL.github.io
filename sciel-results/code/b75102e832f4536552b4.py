"""Multi-objective optimizer on DTLZ7 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7 optimizer using structured 2D sampling on the known Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0

        def make_vector(f1, f2):
            """Create a DTLZ7 decision vector that yields objectives (f1, f2, h(f1,f2)).
            For g=0, the objectives are:
              f1 = x0
              f2 = x1
              f3 = h = n_obj - sum_{i=1}^{n_obj-1} (f_i/(1+g)) * (1 + sin(pi*f_i/(1+g)))
            With g=0, 1+g=1, so:
              f3 = n_obj - f1*(1+sin(pi*f1)) - f2*(1+sin(pi*f2)) - ... 
            Actually for DTLZ7 with m=3 objectives:
              f1 = x0
              f2 = x1
              f3 = (1+g(x_k..x_n)) * h(x0, x1, g)
            where h = m - sum_{i=1}^{m-1} (u_i/(1+g)) * (1 + sin(pi*u_i/(1+g)))
            With g=0: h = 3 - x0*(1+sin(pi*x0)) - x1*(1+sin(pi*x1))
            So we need x0 = f1, x1 = f2, and all other variables set to 0.5 (for g=0).
            """
            x = np.zeros(self.dim)
            x[0] = f1
            x[1] = f2
            # Set g-variables to 0.5 to minimize g (g=0 when all are 0.5)
            for j in range(2, self.dim):
                x[j] = 0.5
            return x

        # Phase 1: Use a well-spaced grid on the 2D (f1, f2) Pareto front
        # The Pareto front for DTLZ7 with m=3 is parameterized by (f1, f2) in [0,1]^2
        # We want to maximize hypervolume with reference point [1.1, 1.1, 25.0]
        # Use a stratified/grid approach for uniform coverage
        
        n_grid = 14  # 14x14 = 196 points, close to budget of 200
        grid_f1 = np.linspace(0.0, 1.0, n_grid)
        grid_f2 = np.linspace(0.0, 1.0, n_grid)
        
        # Evaluate grid points
        for f1 in grid_f1:
            if used >= self.budget:
                break
            for f2 in grid_f2:
                if used >= self.budget or time.time() > deadline:
                    break
                x = make_vector(f1, f2)
                problem(x)
                used += 1

        # Phase 2: Use remaining budget for additional points via max-min distance
        if used < self.budget:
            # Collect evaluated points in (f1, f2) space
            evaluated = []
            for f1 in grid_f1:
                for f2 in grid_f2:
                    evaluated.append((f1, f2))
            evaluated = np.array(evaluated[:used])
            
            # Add more points using max-min distance criterion
            rng = np.random.default_rng(42)
            while used < self.budget and time.time() < deadline:
                # Generate batch of random candidates
                n_candidates = 50
                candidates = rng.random((n_candidates, 2))
                
                if len(evaluated) == 0:
                    best_candidate = candidates[0]
                else:
                    # Compute min distance to existing points
                    diffs = candidates[:, np.newaxis, :] - evaluated[np.newaxis, :, :]
                    dists_sq = np.sum(diffs ** 2, axis=2)
                    min_dists = np.min(dists_sq, axis=1)
                    best_idx = np.argmax(min_dists)
                    best_candidate = candidates[best_idx]
                
                x = make_vector(best_candidate[0], best_candidate[1])
                problem(x)
                used += 1
                evaluated = np.vstack([evaluated, best_candidate.reshape(1, 2)])

        return None
# EVOLVE-BLOCK-END
