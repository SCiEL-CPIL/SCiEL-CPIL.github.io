"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-phase optimizer: CMA-ES global + L-BFGS-B local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s kill

        # Track best solution
        best_x = None
        best_f = float('inf')
        call_count = 0

        def counted_func(x):
            nonlocal best_x, best_f, call_count
            call_count += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES for global exploration
        # Use ~60% of budget for global phase
        global_budget = int(self.budget * 0.60)

        rng = np.random.default_rng(42)
        x0 = np.zeros(self.dim)
        sigma0 = 3.0  # Large initial step for Katsuura's multimodality

        # Simple CMA-ES implementation
        pop_size = self.dim + int(4 * np.log(self.dim))
        pop_size = max(pop_size, 4)

        # CMA-ES state
        mean = x0.copy()
        sigma = sigma0
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        cs = (pop_size + 2) / (self.dim + 2)
        cmu = 2 / ((self.dim + 1.3) ** 2 + pop_size)
        cc = (4 + self.dim / pop_size) / (self.dim + 4 + 2 * pop_size / self.dim)
        c1 = cmu
        cmu = 2 / ((self.dim + 1.3) ** 2 + pop_size)
        c1 = (2 + cmu) / ((self.dim + 1.3) ** 2 + pop_size)
        c1 = (2 + cmu) / ((self.dim + 1.3) ** 2 + pop_size)
        # Proper CMA-ES rates
        c1 = (2 + cmu) / ((self.dim + 1.3) ** 2 + pop_size)
        cmu = 2 / ((self.dim + 1.3) ** 2 + pop_size)
        c1 = (2 + cmu) / ((self.dim + 1.3) ** 2 + pop_size)
        cc = (4 + self.dim / pop_size) / (self.dim + 4 + 2 * pop_size / self.dim)
        cs = (pop_size + 2) / (self.dim + 2)
        damps = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (self.dim + 1) - 1) - 1) + cs
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

        # Generate initial population
        B = np.linalg.cholesky(C)
        D = np.sqrt(np.diag(C))

        for gen in range(global_budget // pop_size + 1):
            if time.time() > deadline - 60:  # Stop if running out of time
                break
            if call_count >= global_budget:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(self.dim)
                y = mean + sigma * (B @ z) / D
                y = np.clip(y, -5, 5)
                fy = counted_func(y)
                offspring.append((fy, y))

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update mean
            w = np.log(np.arange(1, pop_size // 2 + 1)) - np.log(pop_size // 2)
            w = w / np.sum(w)
            mu_eff = 1 / np.sum(w ** 2)

            new_mean = np.zeros(self.dim)
            for i in range(min(pop_size // 2, len(offspring))):
                new_mean += w[i] * offspring[i][1]
            mean = new_mean

            # Update covariance
            diff = np.array([offspring[i][1] for i in range(pop_size // 2)]) - mean
            C = (1 - c1 - cmu) * C
            for i in range(pop_size // 2):
                C += cmu * np.outer(diff[i], diff[i])

            # Update step size
            y_sigma = (mean - (1 - cc) * mean) / (cc * sigma * chi_n)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / chi_n * y_sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) <
                    (1.4 + 2 / (self.dim + 1)) * chi_n)
            if hsig:
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * y_sigma
            else:
                pc = (1 - cs) * pc

            # Update C with rank-one correction
            C = (1 + mu_eff * c1 * (hsig - 1) + cmu) * C
            C += cmu * (1 / mu_eff) * np.outer(pc, pc)

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

            # Recompute B and D
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(C))
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)
                B = np.eye(self.dim)
                D = np.ones(self.dim)

        # Phase 2: L-BFGS-B local polish from best CMA-ES solution
        if best_x is not None and time.time() < deadline - 30:
            try:
                res = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Phase 3: Fine polish with very small steps
        if best_x is not None and time.time() < deadline - 10:
            # Try several small perturbations
            polish_steps = 0
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                if call_count >= self.budget - 5:
                    break
                for scale in [0.01, 0.005, 0.001]:
                    if call_count >= self.budget - 5:
                        break
                    for d in range(self.dim):
                        if call_count >= self.budget - 5:
                            break
                        for direction in [-1, 1]:
                            if call_count >= self.budget - 5:
                                break
                            trial = best_x.copy()
                            trial[d] += direction * scale
                            trial = np.clip(trial, -5, 5)
                            f_trial = counted_func(trial)
                            if f_trial < best_f:
                                best_f = f_trial
                                best_x = trial.copy()
                                polish_steps += 1
                        if polish_steps > 10:
                            break
                    if polish_steps > 10:
                        break
                if polish_steps > 10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
