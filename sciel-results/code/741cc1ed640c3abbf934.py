"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol-based global exploration + (1+1)-ES local refinement for 5D Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return func(x)

        def check_deadline():
            return calls[0] >= budget or time.time() > deadline

        # Phase 1: Sobol-sequence-based space-filling exploration snapped to half-integers
        # This leverages the periodicity of Rastrigin (period 2 in each dimension)
        # by sampling a quasi-random sequence that covers the space uniformly.
        
        # Generate Sobol sequence points
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        
        # Number of Sobol points: use ~250 for good coverage
        n_sobol = 250
        sobol_points = sampler.random(n_sobol)  # [0, 1]^dim
        
        # Map to [-5, 5]^dim
        sobol_points = sobol_points * 10.0 - 5.0
        
        # Snap to half-integers to align with Rastrigin's periodic structure
        # The minima of Rastrigin are at integer coordinates, so half-integers
        # give good coverage between the integer basins
        snapped_points = np.round(sobol_points * 2.0) / 2.0
        snapped_points = np.clip(snapped_points, -5.0, 5.0)
        
        # Also add some random points for diversity
        n_random = 30
        random_points = rng.uniform(-5, 5, (n_random, dim))
        
        # Combine and evaluate
        best_x = None
        best_f = np.inf
        
        # Evaluate Sobol points first
        for i in range(n_sobol):
            if check_deadline():
                break
            x = snapped_points[i]
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Evaluate random points
        for i in range(n_random):
            if check_deadline():
                break
            x = random_points[i]
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            return np.zeros(dim), np.inf
        
        # Phase 2: Local refinement with (1+1)-ES from the best point found
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        no_improve = 0
        max_no_improve = 30
        
        while not check_deadline():
            # Generate trial
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = call_func(y)
            
            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.1
                no_improve = 0
            else:
                sigma *= 0.95
                no_improve += 1
            
            # Restart logic: if stuck, jump to a new Sobol-like point
            if no_improve >= max_no_improve:
                no_improve = 0
                sigma = 0.8
                # Generate a new exploration point near the global best
                # Use a Sobol-like jump with moderate perturbation
                offset = 0.5 * rng.standard_normal(dim)
                x = np.clip(best_x + offset, -5, 5)
                fx = call_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                # Try a few local steps
                for _ in range(5):
                    if check_deadline():
                        break
                    step = sigma * rng.standard_normal(dim)
                    y = np.clip(x + step, -5, 5)
                    fy = call_func(y)
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                sigma = 0.5
        
        # Phase 3: Fine-tuning near the best point with very small steps
        if not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            max_fine_steps = budget - calls[0]
            for i in range(max_fine_steps):
                if check_deadline():
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = call_func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                if sigma < 0.001:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
