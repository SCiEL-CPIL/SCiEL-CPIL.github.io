"""Optimizer for Büche-Rastrigin f4: GP-guided multi-start CMA-ES."""
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """GP-guided multi-start CMA-ES for multi-modal Büche-Rastrigin f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s limit
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() >= deadline or calls_made >= total_budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES implementation
        def cmaes(x0, sigma0, max_calls, min_sigma=1e-8):
            nonlocal calls_made, best_x, best_f
            if calls_made >= max_calls:
                return

            n = dim
            lam = max(4 + int(3 * np.log(n)), 4)  # population size
            # Step size control
            cs = 1 / (n + 3)
            ds = 1 + 2 * max(0, n / 2 - 1) + cs
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            # Weights
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)

            # Initial population
            pop = [x0 + sigma * np.random.randn(n) for _ in range(lam)]
            pop = [np.clip(p, -5, 5) for p in pop]
            fitnesses = [evaluate(p) for p in pop]

            local_best_x = x0.copy()
            local_best_f = min(fitnesses)
            stagnation = 0
            max_stagnation = 50

            while calls_made < max_calls and sigma > min_sigma:
                if check_deadline():
                    break

                # Sort by fitness
                order = np.argsort(fitnesses)
                pop = [pop[i] for i in order]
                fitnesses = [fitnesses[i] for i in order]

                if fitnesses[0] < local_best_f:
                    local_best_f = fitnesses[0]
                    local_best_x = pop[0].copy()
                    stagnation = 0
                else:
                    stagnation += 1
                    if stagnation >= max_stagnation:
                        break

                # Mean
                m = sum(w * p for w, p in zip(weights, pop[:mu]))

                # Update CMA matrix
                delta_mean = (m - x0) / sigma
                pc = (1 - cs) * pc + cs / np.sqrt(mueff) * delta_mean
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - max_calls)))) \
                       < (1 + 2 / (n + 1)) * chi_n
                ps = (1 - ds) * ps + float(hsig) * np.sqrt(mueff) * delta_mean
                c1 = (2 + 2 / (n + 1)) / ((n + 2) ** 2 + mueff)
                cmu = min(2 * (1 - c1) / ((n + 2) ** 2 + mueff), 1 - c1)
                C = (1 - c1 - cmu + cmu * (mueff - 2 + 1 / mueff)) * C \
                    + c1 * (np.outer(pc, pc) - np.eye(n)) \
                    + cmu / mueff * sum(w * np.outer(p - x0, p - x0) / sigma ** 2
                                        for w, p in zip(weights, pop[:mu]))

                # Ensure symmetry and positive definite
                C = (C + C.T) / 2
                # Eigenvalue check for stability
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C = C + 1e-10 * np.eye(n)

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))

                x0 = m

                # Generate new population
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)

                new_pop = []
                for _ in range(lam):
                    z = np.random.randn(n)
                    new_x = x0 + sigma * L @ z
                    new_x = np.clip(new_x, -5, 5)
                    new_pop.append(new_x)

                # Evaluate
                new_fitnesses = [evaluate(p) for p in new_pop]
                pop = new_pop
                fitnesses = new_fitnesses

                # Update global best
                if fitnesses[0] < best_f:
                    best_f = fitnesses[0]
                    best_x = pop[0].copy()

            return local_best_f, local_best_x

        # Gaussian Process for global guidance
        class GP:
            def __init__(self, length_scale=0.5, noise=1e-4):
                self.length_scale = length_scale
                self.noise = noise
                self.X = None
                self.Y = None
                self.alpha = None

            def kernel(self, X1, X2):
                # Matern kernel
                d2 = np.sum(X1 ** 2, axis=1)[:, None] - 2 * X1 @ X2.T \
                     + np.sum(X2 ** 2, axis=1)[None, :]
                d2 = np.maximum(d2, 0)
                r = np.sqrt(d2)
                return np.exp(-r / self.length_scale)

            def fit(self, X, Y):
                self.X = X
                self.Y = Y
                K = self.kernel(X, X) + self.noise * np.eye(len(X))
                try:
                    self.alpha = np.linalg.solve(K, Y)
                except np.linalg.LinAlgError:
                    self.alpha = np.linalg.lstsq(K, Y, rcond=None)[0]

            def predict(self, X):
                K = self.kernel(self.X, X)
                mean = K @ self.alpha
                return mean

            def acquisition(self, X):
                # Expected improvement
                K = self.kernel(self.X, X)
                mean = K @ self.alpha
                Kxx = self.kernel(X, X) + self.noise * np.eye(len(X))
                cov = Kxx - K.T @ np.linalg.solve(self.kernel(self.X, self.X) + self.noise * np.eye(len(self.X)), K)
                std = np.sqrt(np.maximum(np.diag(cov), 0))
                mu = self.Y.min()
                z = (mu - mean) / (std + 1e-8)
                return z * norm.pdf(z) + (mu - mean) * norm.cdf(z)

        # Phase 1: Initial exploration - diverse random + structured points
        rng = np.random.default_rng(123)

        # Generate diverse initial points
        init_points = []
        # Random points
        for _ in range(40):
            init_points.append(rng.uniform(-4.5, 4.5, dim))
        # Structured points around origin
        for r in [0.5, 1.0, 2.0, 3.0]:
            for _ in range(5):
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction) + 1e-10
                init_points.append(r * direction)
        # Axis-aligned points
        for axis in range(dim):
            for val in [-3, -1, 1, 3]:
                pt = np.zeros(dim)
                pt[axis] = val
                init_points.append(pt)

        # Evaluate initial points
        for pt in init_points:
            if check_deadline():
                break
            evaluate(np.clip(pt, -5, 5))

        # Phase 2: GP-guided exploration + CMA-ES refinement
        gp = GP(length_scale=0.8, noise=1e-3)

        # Collect evaluated points for GP
        eval_history = []  # list of (x, f)
        # Re-evaluate some points to build history (we don't have stored history easily)
        # Instead, use current best and nearby points

        # Multi-start CMA-ES from diverse starting points
        # Start from global best and several diverse points
        start_points = [best_x.copy()]

        # Add diverse starts
        for _ in range(4):
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction) + 1e-10
            for scale in [1.0, 2.0]:
                sp = best_x + scale * direction
                sp = np.clip(sp, -5, 5)
                start_points.append(sp)

        # Also add some random starts
        for _ in range(3):
            start_points.append(rng.uniform(-4, 4, dim))

        # Allocate budget: give more to promising starts
        remaining_budget = total_budget - calls_made
        n_starts = len(start_points)
        # Adaptive allocation: start with equal, then adjust
        base_budget = max(30, remaining_budget // n_starts)

        for i, sp in enumerate(start_points):
            if check_deadline() or calls_made >= total_budget:
                break

            # Determine sigma based on how far from best
            dist = np.linalg.norm(sp - best_x) if best_x is not None else 1.0
            sigma = max(0.3, min(1.5, dist * 0.5))

            # Give more budget to first few starts
            if i < 3:
                max_calls = calls_made + int(base_budget * 1.5)
            else:
                max_calls = calls_made + base_budget

            cmaes(sp, sigma, max_calls)

        # Phase 3: Fine-tuning around global best
        if not check_deadline() and best_x is not None:
            # Tight CMA-ES around best
            remaining = total_budget - calls_made
            if remaining > 20:
                cmaes(best_x.copy(), 0.1, total_budget)

        return best_x, best_f
