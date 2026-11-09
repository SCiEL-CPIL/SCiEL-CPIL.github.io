"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with Rastrigin-aware initialization for f15."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        total_calls = 0
        budget = self.budget

        def elapsed():
            return time.time() - start_time

        def time_left():
            return deadline - time.time()

        best_x = np.zeros(dim)
        best_f = float('inf')

        # Rastrigin f15: sum of 10*(x_i^2 - cos(2*pi*x_i)) + 10*dim
        # Local minima at integers. The global minimum is at the shifted/rotated origin.
        # Strategy: try multiple starting points near integer lattice points,
        # then run CMA-ES from the best few.

        # Phase 1: Sample candidate starting points
        # For Rastrigin, local minima are at integer coordinates.
        # With rotation, the global min is at R @ shift, but local minima are still
        # at rotated integer lattice points. We sample a diverse set of points.
        
        # Generate candidate starts: origin, random points, and integer-like points
        candidates = []
        
        # Origin
        candidates.append(np.zeros(dim))
        
        # Random uniform points
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, dim))
        
        # Points near integer lattice (rounded random points)
        for _ in range(10):
            p = rng.uniform(-4, 4, dim)
            candidates.append(np.round(p))
        
        # Evaluate candidates
        cand_scores = []
        for c in candidates:
            if total_calls >= budget or time_left() < 1:
                break
            c_clipped = np.clip(c, -5, 5)
            fc = func(c_clipped)
            total_calls += 1
            cand_scores.append((fc, c_clipped))
            if fc < best_f:
                best_f = fc
                best_x = c_clipped.copy()

        if not cand_scores:
            return best_x, best_f

        # Sort candidates by fitness
        cand_scores.sort(key=lambda x: x[0])

        # Phase 2: Run CMA-ES from the top 3 candidates
        # CMA-ES implementation
        def cma_es(func, x0, budget_remaining, dim, rng, best_f_ref, best_x_ref, total_calls_ref, deadline):
            """Simple CMA-ES implementation."""
            nonlocal total_calls, best_f, best_x
            
            # CMA-ES parameters
            lambda_ = 4 + int(3 * np.log(dim))  # population size
            mu = lambda_ // 2  # recombination weight count
            sigma = 2.0  # initial step size
            c_sigma = (mu_eff + 2) / ((dim + 2) ** 2 + mu_eff)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            mu_eff = sum((1 / i for i in range(1, mu + 1))) ** -1  # effective mu
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Mean and covariance
            mean = x0.copy().astype(float)
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_sqrt_inv = np.eye(dim)

            # Path variables
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Track best within this CMA-ES run
            local_best_x = x0.copy()
            local_best_f = func(x0)
            total_calls += 1
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            # Precompute
            def update_cma_matrices():
                nonlocal C, C_inv, C_sqrt, C_sqrt_inv
                # Update C using rank-mu update
                # For simplicity, use a simplified CMA-ES
                pass

            # Simplified CMA-ES: use (1+1)-ES with CMA-like adaptation
            # Actually, let's implement a proper but simple CMA-ES

            # For efficiency, let's use a simplified approach:
            # (1+lambda)-ES with adaptive sigma and restarts

            x = x0.copy()
            fx = local_best_f
            sigma = 1.5
            no_improve = 0
            stagnation_limit = 25

            while total_calls < budget and time_left() > 1:
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.25, 4.0)
                    no_improve = 0
                    if fx < local_best_f:
                        local_best_f = fx
                        local_best_x = x.copy()
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                else:
                    sigma = max(sigma * 0.8, 0.03)
                    no_improve += 1

                # Stagnation: restart locally
                if no_improve >= stagnation_limit and sigma < 0.3:
                    # Try a few random restarts around current best
                    for _ in range(3):
                        if total_calls >= budget or time_left() < 1:
                            break
                        z = local_best_x + 1.0 * rng.standard_normal(dim)
                        z = np.clip(z, -5, 5)
                        fz = func(z)
                        total_calls += 1
                        if fz < local_best_f:
                            local_best_f = fz
                            local_best_x = z.copy()
                            if fz < best_f:
                                best_f = fz
                                best_x = z.copy()
                    x = local_best_x.copy()
                    fx = local_best_f
                    sigma = 1.0
                    no_improve = 0

            return local_best_x, local_best_f

        # Run CMA-ES from top candidates
        for i, (fc, cx) in enumerate(cand_scores[:3]):
            if total_calls >= budget or time_left() < 5:
                break
            # Allocate remaining budget to this run
            cma_es(func, cx, budget, dim, rng, best_f, best_x, total_calls, deadline)

        return best_x, best_f
# EVOLVE-BLOCK-END
