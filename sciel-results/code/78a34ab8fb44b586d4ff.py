"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for WFG4 using structured decomposition.
    
    WFG4 has 'deciding' variables (first 2) that determine the Pareto front shape
    and 'positioning' variables (rest) that determine distance from the ideal point.
    
    Strategy:
    1. Sample a diverse set of deciding variable configurations (x0, x1).
    2. For each, use local search on positioning variables to minimize a scalar
       aggregation (Tchebycheff) while keeping deciding variables fixed.
    3. Use remaining budget for random restarts and refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Track all evaluated points and their objective values
        X_archive = []
        F_archive = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            fx = np.array(problem(x))
            X_archive.append(x.copy())
            F_archive.append(fx)
            used += 1
            return fx

        # Phase 1: Structured sampling of deciding variables
        # For WFG4, x0 and x1 are the deciding variables
        # Sample a grid of (x0, x1) values
        n_deciding_samples = 15
        x0_samples = np.linspace(lo[0], hi[0], n_deciding_samples)
        x1_samples = np.linspace(lo[1], hi[1], n_deciding_samples)

        # For each (x0, x1) pair, do local search on remaining variables
        # Use Tchebycheff aggregation with different weight vectors
        n_weights = 3
        weights = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        # Add some mixed weights
        weights = np.vstack([weights, [0.33, 0.33, 0.34]])

        # Initial random evaluations
        n_init = min(10, self.budget // 4)
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(self.dim)
            evaluate(x)

        # Phase 2: For each deciding variable configuration, do local search
        for x0 in x0_samples:
            for x1 in x1_samples:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Initialize positioning variables randomly
                x_pos = lo[2:] + (hi[2:] - lo[2:]) * rng.random(self.dim - 2)
                x = np.concatenate([[x0, x1], x_pos])
                fx = evaluate(x)
                if fx is None:
                    break

                # Local search on positioning variables using Tchebycheff
                best_x = x.copy()
                best_f = fx.copy()

                # Simple local search with decreasing step size
                step_sizes = [0.3, 0.1, 0.03, 0.01]
                for step in step_sizes:
                    if used >= self.budget or time.time() > deadline:
                        break
                    improved = True
                    while improved and used < self.budget:
                        improved = False
                        for _ in range(5):
                            if used >= self.budget or time.time() > deadline:
                                break
                            # Perturb positioning variables
                            x_new = best_x.copy()
                            x_new[2:] += step * (hi[2:] - lo[2:]) * rng.standard_normal(self.dim - 2)
                            x_new = np.clip(x_new, lo, hi)
                            f_new = evaluate(x_new)
                            if f_new is None:
                                break
                            # Check if better in Tchebycheff sense for any weight
                            for w in weights:
                                if np.max(w * (f_new - best_f)) < 0:
                                    best_x = x_new.copy()
                                    best_f = f_new.copy()
                                    improved = True
                                    break

        # Phase 3: Use remaining budget for random restarts with local refinement
        while used < self.budget and time.time() < deadline:
            x = lo + (hi - lo) * rng.random(self.dim)
            fx = evaluate(x)
            if fx is None:
                break

            # Quick local search
            best_x = x.copy()
            best_f = fx.copy()
            for step in [0.1, 0.03]:
                for _ in range(3):
                    if used >= self.budget or time.time() > deadline:
                        break
                    x_new = best_x.copy()
                    x_new += step * (hi - lo) * rng.standard_normal(self.dim)
                    x_new = np.clip(x_new, lo, hi)
                    f_new = evaluate(x_new)
                    if f_new is None:
                        break
                    for w in weights:
                        if np.max(w * (f_new - best_f)) < 0:
                            best_x = x_new.copy()
                            best_f = f_new.copy()
                            break

        return X_archive
# EVOLVE-BLOCK-END
