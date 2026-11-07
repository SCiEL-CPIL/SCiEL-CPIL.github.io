"""Black-box continuous optimizer on BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for BBOB f11 (Discus) using:
    - Phase 1: Sobol sampling for initial exploration.
    - Phase 2: CMA-ES for global search exploiting valley structure.
    - Phase 3: Multi-start L-BFGS-B local refinement from top candidates.
    - Deadline guard to ensure termination within time limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_budget():
            nonlocal calls_used
            if calls_used >= self.budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")

        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            check_budget()
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol sampling to build initial diverse set
        n_initial = min(32, self.budget // 8)
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_initial)
        X_initial = sobol_points * 10.0 - 5.0

        evaluations = []
        for x in X_initial:
            if calls_used >= self.budget - 50:
                break
            fx = safe_call(x)
            evaluations.append((fx, x))

        if not evaluations:
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            evaluations.append((fx, x))

        evaluations.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES for global search
        # CMA-ES is well-suited for the Discus function's elongated valley structure
        n_cma_steps = min(40, self.budget - calls_used - 30)
        
        # Initialize CMA-ES state
        pop_size = max(12, int(4 + 3 * np.log(self.dim)))
        x_mean = np.array([x for _, x in evaluations[:1]][0], dtype=float)
        sigma = 2.0  # Initial step size
        C = np.eye(self.dim)  # Covariance matrix
        ps = np.zeros(self.dim)  # Evolution path for sigma
        pc = np.zeros(self.dim)  # Evolution path for C
        cs = (pop_size + 2) / (self.dim + pop_size + 5)
        ds = 1 + 2 * max(0, (self.dim / (pop_size + 2) - 1)) + cs
        cmu = 2 / ((self.dim + 1.3) ** 2 + pop_size)
        c1 = (2 - cmu + 2 * self.dim / (pop_size + self.dim) ** 3) / ((self.dim + 2) ** 2 + pop_size)
        c_sigma = cs * (ds + 2 - cs)

        for step in range(n_cma_steps):
            if calls_used >= self.budget - 30:
                break
            if time.time() > deadline - 10:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                # Sample from N(x_mean, sigma^2 * C)
                B = np.linalg.cholesky(C)
                z = rng.standard_normal(self.dim)
                x_new = x_mean + sigma * (B @ z)
                x_new = np.clip(x_new, -5, 5)
                fx = safe_call(x_new)
                offspring.append((fx, x_new))
            
            if len(offspring) < pop_size:
                break

            offspring.sort(key=lambda t: t[0])
            lambda_sel = pop_size // 2

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Update mean
            weights = np.zeros(lambda_sel)
            w = 2.0 / (self.dim + pop_size - 2)
            weights[0] = min(w, 2.0 / pop_size)
            for i in range(1, lambda_sel):
                weights[i] = w * (1 - weights[0]) / (lambda_sel - 1)
            mu_eff = 1.0 / np.sum(weights ** 2)

            x_new_mean = np.zeros(self.dim)
            for i in range(lambda_sel):
                x_new_mean += weights[i] * offspring[i][1]
            
            # Update evolution paths
            sqrt_c_sigma = np.sqrt(cs * (ds + 2 - cs) * mu_eff)
            ps = (1 - cs) * ps + sqrt_c_sigma * np.linalg.inv(np.linalg.cholesky(C)) @ (x_new_mean - x_mean) / sigma
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (step + 1)))) < (1.4 + 2 / (self.dim + 1)) * np.sqrt((self.dim + 1) / (self.dim - 1))
            
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mu_eff) * (x_new_mean - x_mean) / sigma

            # Update covariance
            delta_hsig = (1 - hsig) * c1 * (2 - c1)
            C = (1 + c1 * cmu * mu_eff - c1) * C + c1 * cmu * np.outer(pc, pc) * delta_hsig
            for i in range(lambda_sel):
                y_i = (offspring[i][1] - x_mean) / sigma
                C += c1 * cmu * weights[i] * np.outer(y_i, y_i)
            
            # Ensure C is positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x_mean = x_new_mean

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 0.01, 10.0)

            # Add to evaluations
            for fx, x in offspring:
                evaluations.append((fx, x))
            evaluations.sort(key=lambda t: t[0])

        # Phase 3: Local refinement from the best few points
        remaining_budget = self.budget - calls_used

        # Select top candidates for local refinement
        n_restarts = min(8, len(evaluations))
        bounds = [(-5, 5)] * self.dim

        for i in range(n_restarts):
            if time.time() > deadline - 10:
                break
            if calls_used >= self.budget - 5:
                break

            fx0, x0 = evaluations[i]

            def lbfgs_func(x):
                return safe_call(x)

            try:
                res = minimize(
                    lbfgs_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'gtol': 1e-12,
                        'ftol': 1e-14,
                    }
                )
            except Exception:
                pass

        # Phase 4: If budget remains, try more local searches from diverse points
        if calls_used < self.budget - 20 and time.time() < deadline - 30:
            possible_restarts = (self.budget - calls_used) // 40
            for _ in range(possible_restarts):
                if time.time() > deadline - 10:
                    break
                if calls_used >= self.budget - 10:
                    break

                # Sample near the current best but with some perturbation
                if best_x is not None:
                    x0 = best_x + rng.normal(0, 0.5, self.dim)
                    x0 = np.clip(x0, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, self.dim)
                fx0 = safe_call(x0)

                if calls_used < self.budget - 5:
                    def lbfgs_func2(x):
                        return safe_call(x)
                    try:
                        res = minimize(
                            lbfgs_func2,
                            x0,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={
                                'maxiter': 30,
                                'gtol': 1e-10,
                            }
                        )
                    except Exception:
                        pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            if calls_used < self.budget:
                best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
