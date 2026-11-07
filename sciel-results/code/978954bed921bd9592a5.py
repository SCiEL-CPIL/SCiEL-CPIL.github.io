"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with CMA-like adaptation, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # We'll use a population-based ES with adaptive step size
        # Key insight from knowledge: more initial exploration helps on bi-Rastrigin
        # Strategy: 
        #   Phase 1: Broad exploration with large population to find good basins
        #   Phase 2: Adaptive (mu+lambda)-ES with sigma adaptation for refinement
        #   Phase 3: Local polish with Nelder-Mead or fine ES steps

        def time_ok():
            return time.time() < deadline

        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_point(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            fx = func(np.asarray(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, copy=True)
            return fx

        # Phase 1: Broad random sampling to find promising regions
        # Use stratified sampling for better coverage
        n_initial = min(50, budget // 10)
        if n_initial < 10:
            n_initial = 10

        # Stratified sampling: divide box into cells
        # For dim=5, use a grid of 2^5=32 cells, sample 1-2 per cell
        grid_size = 2
        n_cells = grid_size ** dim
        points_per_cell = max(1, n_initial // n_cells)

        initial_points = []
        for i in range(n_cells):
            # Convert index to grid coordinates
            coords = []
            idx = i
            for d in range(dim):
                coords.append(idx % grid_size)
                idx //= grid_size
            # Random point within this cell
            for _ in range(points_per_cell):
                pt = np.zeros(dim)
                for d in range(dim):
                    lo = -5.0 + coords[d] * (10.0 / grid_size)
                    hi = lo + 10.0 / grid_size
                    pt[d] = rng.uniform(lo, hi)
                initial_points.append(pt)

        # Shuffle
        rng.shuffle(initial_points)

        for pt in initial_points:
            if not time_ok() or call_count >= budget:
                break
            eval_point(pt)

        if best_x is None:
            # Fallback: single random point
            x0 = rng.uniform(-5, 5, dim)
            eval_point(x0)

        if not time_ok() or call_count >= budget:
            return best_x, best_f

        # Phase 2: (mu+lambda)-ES with adaptive sigma
        # Start from best point
        x_best = best_x.copy()
        f_best = best_f

        # Initialize sigma based on problem scale
        sigma = 1.5
        mu = 5   # number of parents
        lam = 15  # number of offspring per generation

        # CMA-like covariance adaptation (simplified)
        # We'll use a diagonal covariance that adapts per-dimension
        c_sigma = 1.0 / (dim + 3)
        c_c = (4 - dim / (dim + 4)) / ((dim + 2) * (dim + 2) + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu) / ((dim + 2) ** 2 + mu_eff))
        mu_eff = mu / 2.0

        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Diagonal covariance (start as identity scaled)
        diag = np.ones(dim) * sigma ** 2

        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        nu = 1.0 + 2 * (mu / 2.0 - 1) + mu / 2.0 * (1 - mu / 2.0) / mu

        # Adaptation rates
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        gen = 0
        no_improve = 0
        patience = 40  # generations without improvement before potential restart

        while time_ok() and call_count < budget:
            # Generate offspring
            offspring = []
            offspring_f = []

            for i in range(lam):
                if not time_ok() or call_count >= budget:
                    break
                # Sample from multivariate normal with diagonal covariance
                z = rng.standard_normal(dim)
                y = x_best + sigma * z * np.sqrt(diag)
                y = np.clip(y, -5, 5)
                fy = eval_point(y)
                if fy is not None:
                    offspring.append(y)
                    offspring_f.append(fy)

            if len(offspring) < 2:
                break

            # Sort offspring by fitness
            idx = np.argsort(offspring_f)
            sorted_x = [offspring[i] for i in idx]
            sorted_f = [offspring_f[i] for i in idx]

            # Update best
            if sorted_f[0] < f_best:
                f_best = sorted_f[0]
                x_best = sorted_x[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Update mean (recombination)
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * sorted_x[i]
            
            # Update sigma (1/5 rule adapted)
            # Count successes
            n_success = sum(1 for i in range(min(mu, len(sorted_x))) if sorted_f[i] < f_best + 1e-10)
            # Actually, let's use a simpler sigma adaptation
            # Compare new mean to old mean
            diff = new_mean - x_best
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-10:
                # Expected norm of diff under current sigma
                expected_norm = sigma * chi_n
                sigma *= np.exp(c_sigma * (norm_diff / expected_norm - 1))
                sigma = np.clip(sigma, 0.01, 5.0)
            else:
                sigma *= 0.95

            # Update diagonal covariance
            for d in range(dim):
                # Simple adaptation: reduce variance if that dimension improved a lot
                var_d = np.var([p[d] for p in sorted_x[:mu]])
                diag[d] = 0.9 * diag[d] + 0.1 * max(var_d, 1e-6)

            x_best = new_mean

            # Check for stagnation
            if no_improve >= patience:
                # Perturb to escape local minimum
                if time_ok() and call_count < budget:
                    # Try a random jump
                    jump_size = 2.0
                    x_try = x_best + jump_size * rng.standard_normal(dim)
                    x_try = np.clip(x_try, -5, 5)
                    f_try = eval_point(x_try)
                    if f_try is not None and f_try < f_best:
                        x_best = x_try.copy()
                        f_best = f_try
                        no_improve = 0
                        sigma = 1.0
                    else:
                        # Reset sigma and try again with smaller perturbation
                        sigma = max(sigma * 0.5, 0.1)
                        no_improve = 0

            gen += 1

        # Phase 3: Fine polish with small steps
        if time_ok() and call_count < budget:
            sigma_fine = 0.05
            for _ in range(min(50, budget - call_count)):
                if not time_ok() or call_count >= budget:
                    break
                y = x_best + sigma_fine * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = eval_point(y)
                if fy is not None and fy < f_best:
                    x_best = y.copy()
                    f_best = fy
                    sigma_fine *= 1.2
                else:
                    sigma_fine *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
