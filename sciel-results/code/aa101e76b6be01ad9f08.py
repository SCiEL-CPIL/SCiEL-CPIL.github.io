"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Population-based CMA-ES style optimizer with:
    - Restart on stagnation
    - Hessian estimation via finite differences for Newton-like refinement
    - Deadline guard
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.deadline = 320.0  # safety margin before 330s kill

    def _check_deadline(self):
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.deadline

    def _count_call(self):
        self.calls += 1

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Phase 1: CMA-ES with restarts
        remaining = self.budget
        max_restarts = 3
        restart_budget = 150

        for restart in range(max_restarts):
            if remaining <= 0 or self._check_deadline():
                break

            # Initialize population
            pop_size = min(4 + self.dim, remaining)
            if pop_size < 4:
                pop_size = max(2, remaining)

            # Starting point
            if best_x is None:
                x_mean = rng.uniform(-4, 4, self.dim)
            else:
                x_mean = best_x.copy()

            sigma = 2.0
            C = np.eye(self.dim)
            step_size = 1.0

            # Population
            pop_x = []
            pop_f = []

            for i in range(pop_size):
                if self._check_deadline() or self.calls >= self.budget:
                    break
                x = np.clip(x_mean + sigma * rng.multivariate_normal(
                    np.zeros(self.dim), C, size=1)[0], -5, 5)
                f = func(x)
                self._count_call()
                pop_x.append(x)
                pop_f.append(f)

                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            if not pop_x:
                break

            # Sort by fitness
            order = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in order]
            pop_f = [pop_f[i] for i in order]

            # CMA-ES update
            lambda_ = len(pop_x)
            mu = lambda_ // 2
            weights = np.zeros(lambda_)
            cum_weight = 0.0
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / self.dim) / ((self.dim + 2) ** 2 + mu_eff))

            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)

            # Track progress for restart
            prev_best = pop_f[0]
            stall_count = 0
            stall_threshold = 30

            while not self._check_deadline() and self.calls < self.budget:
                # Generate new population
                new_pop_x = []
                new_pop_f = []

                for i in range(lambda_):
                    if self._check_deadline() or self.calls >= self.budget:
                        break
                    z = rng.standard_normal(self.dim)
                    x = np.clip(x_mean + sigma * (z @ C), -5, 5)
                    f = func(x)
                    self._count_call()
                    new_pop_x.append(x)
                    new_pop_f.append(f)

                    if f < best_f:
                        best_f = f
                        best_x = x.copy()

                if not new_pop_x:
                    break

                order = np.argsort(new_pop_f)
                new_pop_x = [new_pop_x[i] for i in order]
                new_pop_f = [new_pop_f[i] for i in order]

                # Check for improvement
                if new_pop_f[0] < prev_best - 1e-10:
                    stall_count = 0
                else:
                    stall_count += 1

                if stall_count >= stall_threshold:
                    break

                prev_best = new_pop_f[0]

                # Update mean
                x_w = np.zeros(self.dim)
                for i in range(mu):
                    x_w += weights[i] * new_pop_x[i]
                x_mean = x_w

                # Update sigma
                d_sigma = np.sqrt((1 + 2 * max(0, c_sigma - 1 / self.dim)) *
                                   (1 - c_sigma))
                p_sigma = (1 - c_sigma) * p_sigma + d_sigma * np.sqrt(mu_eff) * (
                    x_w - x_mean) / (sigma + 1e-15)
                sigma *= np.exp(c_sigma / d_sigma * (
                    np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update C
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (
                    x_w - x_mean) / (sigma + 1e-15)
                C = (1 + c_1 * (mu_eff - 1) - c_mu) * C + \
                    c_1 * np.outer(p_c, p_c) + \
                    c_mu / sigma ** 2 * np.sum(
                        [weights[i] * np.outer(new_pop_x[i] - x_mean,
                                                new_pop_x[i] - x_mean)
                         for i in range(mu)], axis=0)

                # Ensure symmetry and positive definite
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except:
                    C = np.eye(self.dim)

                pop_x = new_pop_x
                pop_f = new_pop_f

            # After CMA-ES phase, do local refinement
            if not self._check_deadline() and self.calls < self.budget and best_x is not None:
                best_x, best_f = self._newton_refine(func, best_x, best_f, rng)

        # Final local refinement if time permits
        if not self._check_deadline() and self.calls < self.budget and best_x is not None:
            best_x, best_f = self._newton_refine(func, best_x, best_f, rng)

        return best_x, best_f

    def _newton_refine(self, func, x, fx, rng):
        """Newton-like refinement using finite-difference Hessian."""
        n = self.dim
        h = 1e-4
        calls_used = 0
        max_newton_iters = 20

        for it in range(max_newton_iters):
            if self._check_deadline() or self.calls >= self.budget:
                break

            # Estimate gradient
            grad = np.zeros(n)
            for i in range(n):
                if self._check_deadline() or self.calls >= self.budget:
                    break
                xp = x.copy()
                xp[i] += h
                xp = np.clip(xp, -5, 5)
                fp = func(xp)
                self._count_call()
                calls_used += 1

                xm = x.copy()
                xm[i] -= h
                xm = np.clip(xm, -5, 5)
                fm = func(xm)
                self._count_call()
                calls_used += 1

                grad[i] = (fp - fm) / (2 * h)

            if np.linalg.norm(grad) < 1e-8:
                break

            # Estimate Hessian (only if dimension is small enough)
            if n <= 6:
                H = np.zeros((n, n))
                for i in range(n):
                    if self._check_deadline() or self.calls >= self.budget:
                        break
                    for j in range(i, n):
                        if self._check_deadline() or self.calls >= self.budget:
                            break
                        xpp = x.copy()
                        xpp[i] += h
                        xpp[j] += h
                        xpp = np.clip(xpp, -5, 5)
                        fpp = func(xpp)
                        self._count_call()

                        xpm = x.copy()
                        xpm[i] += h
                        xpm[j] -= h
                        xpm = np.clip(xpm, -5, 5)
                        fpm = func(xpm)
                        self._count_call()

                        xmp = x.copy()
                        xmp[i] -= h
                        xmp[j] += h
                        xmp = np.clip(xmp, -5, 5)
                        fmp = func(xmp)
                        self._count_call()

                        xmm = x.copy()
                        xmm[i] -= h
                        xmm[j] -= h
                        xmm = np.clip(xmm, -5, 5)
                        fmm = func(xmm)
                        self._count_call()

                        Hij = (fpp - fpm - fmp + fmm) / (4 * h * h)
                        H[i, j] = Hij
                        H[j, i] = Hij

                # Regularize Hessian
                H = (H + H.T) / 2
                eigvals, eigvecs = np.linalg.eigh(H)
                eigvals = np.clip(eigvals, 1e-6, None)
                H_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T

                try:
                    step = np.linalg.solve(H_reg, -grad)
                except:
                    step = -grad / (np.linalg.norm(grad) + 1e-15)

                # Damped Newton step
                alpha = 1.0
                for _ in range(10):
                    x_new = np.clip(x + alpha * step, -5, 5)
                    if self._check_deadline() or self.calls >= self.budget:
                        return x, fx
                    f_new = func(x_new)
                    self._count_call()
                    if f_new < fx:
                        x, fx = x_new, f_new
                        break
                    alpha *= 0.5
            else:
                # Fallback: gradient descent
                lr = 0.01
                x_new = np.clip(x - lr * grad, -5, 5)
                if self._check_deadline() or self.calls >= self.budget:
                    break
                f_new = func(x_new)
                self._count_call()
                if f_new < fx:
                    x, fx = x_new, f_new
                else:
                    break

        return x, fx
# EVOLVE-BLOCK-END
