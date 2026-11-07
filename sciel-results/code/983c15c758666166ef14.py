"""
Lattice Search + CMA-ES optimizer for BBOB f3 (Rastrigin) on [-5, 5]^5.
Exploits Rastrigin's periodicity: local minima at integer coordinates.
Uses a 0.1-step lattice scan to find the basin, then CMA-ES for refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin

        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def cmaes(func, x0, sigma0, max_iter, max_calls):
            """
            CMA-ES implementation for local refinement.
            Returns best point found.
            """
            nonlocal best_x, best_f, calls_made

            dim = n
            # CMA-ES parameters
            lambda_pop = max(4 + int(3 * np.log(dim)), 8)
            mu = lambda_pop // 2
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # State
            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

            for it in range(max_iter):
                if check_deadline() or calls_made >= max_calls:
                    break

                # Generate population
                B = np.linalg.cholesky(C)
                for i in range(lambda_pop):
                    z = rng.standard_normal(dim)
                    x = x_mean + sigma * (B @ z)
                    x = np.clip(x, -5, 5)
                    f = func(x)
                    update_best(x, f)
                    if i == 0:
                        xs = x
                        fs = f
                    else:
                        xs = np.vstack([xs, x])
                        fs = np.append(fs, f)

                if calls_made >= max_calls or check_deadline():
                    break

                # Sort by fitness
                idx = np.argsort(fs)
                xs = xs[idx]
                fs = fs[idx]

                # Update mean
                x_old = x_mean.copy()
                x_mean = weights[:mu] @ xs[:mu]
                x_mean = np.clip(x_mean, -5, 5)

                # Update covariance
                y = (x_mean - x_old) / sigma
                pc = (1 - c_c) * pc + c_c * y
                ps = (1 - c_sigma) * ps + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_mean - x_old)
                delta_sig = (np.linalg.norm(ps) / chi_n - 1) * c_sigma * d_sigma
                sigma *= np.exp(delta_sig)

                # Ensure sigma doesn't grow too large
                sigma = min(sigma, 5.0)

                # Update C
                v1 = np.outer(pc, pc) - C
                v2 = (weights[:mu] * (np.outer(xs[:mu] - x_old, xs[:mu] - x_old).sum(axis=0) / mu_eff - C))
                C = (1 - c_1 - c_mu) * C + c_1 * v1 + c_mu * v2

                # Symmetrize and ensure positive definite
                C = (C + C.T) / 2
                # Add small regularization if needed
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.min() < 1e-12:
                    C += (1e-12 - eigvals.min()) * np.eye(dim)

                # Convergence check
                if np.linalg.norm(x_mean - x_old) < 1e-10 * (1 + np.abs(x_mean).max()):
                    break

            return best_x, best_f

        # Phase 1: Lattice search with 0.1 step size
        # Rastrigin has local minima at integer coordinates with period 1.0
        # A 0.1-step lattice ensures we sample within 0.05 of any integer point
        # Sample from [-4, 4] in each dimension (9 points per dim = 9^5 = 59049 total)
        # But we only have 500 calls, so we need to be strategic
        
        # First, sample integer points (9^5 = 59049, but we'll use a subset)
        # Then refine around the best integer point with 0.1-step lattice
        
        # Step 1a: Sample integer points in [-4, 4]^5
        int_range = list(range(-4, 5))
        int_points = list(np.array(np.meshgrid(*[int_range] * n), dtype=float).T.reshape(-1, n))
        
        # Shuffle and evaluate up to 50 integer points
        rng.shuffle(int_points)
        max_int_calls = min(50, budget // 5)
        for pt in int_points[:max_int_calls]:
            if check_deadline():
                break
            x = np.clip(pt, -5, 5)
            f = func(x)
            update_best(x, f)

        if best_x is None:
            x = np.zeros(n)
            f = func(x)
            update_best(x, f)
            return best_x, best_f

        # Step 1b: Lattice search around best integer point with 0.1 step
        # Search in a box of radius 0.5 around best_x (11 points per dim = 11^5 = 161051)
        # But we'll use a smaller budget for this phase
        remaining = budget - calls_made
        if remaining > 20 and not check_deadline():
            # Create a 0.1-step lattice in [-0.5, 0.5] around best_x
            offsets = np.arange(-0.5, 0.51, 0.1)
            lattice_points = list(np.array(np.meshgrid(*[offsets] * n), dtype=float).T.reshape(-1, n))
            
            # Shuffle and evaluate up to remaining//2 points
            rng.shuffle(lattice_points)
            max_lattice_calls = min(len(lattice_points), remaining // 2)
            for off in lattice_points[:max_lattice_calls]:
                if check_deadline():
                    break
                x = best_x + off
                x = np.clip(x, -5, 5)
                f = func(x)
                update_best(x, f)

        # Phase 2: CMA-ES local refinement
        remaining = budget - calls_made
        if remaining > 30 and not check_deadline():
            # Use smaller sigma for local refinement
            sigma0 = 0.3
            max_iter = min(50, remaining // 10)
            cmaes(func, best_x, sigma0, max_iter, remaining)

        # Phase 3: Final fine-tuning with smaller sigma if budget remains
        remaining = budget - calls_made
        if remaining > 20 and not check_deadline():
            sigma0 = 0.1
            max_iter = min(30, remaining // 10)
            cmaes(func, best_x, sigma0, max_iter, remaining)

        return best_x, best_f

# EVOLVE-BLOCK-END
