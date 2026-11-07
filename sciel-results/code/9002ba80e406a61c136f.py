import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer for BBOB f13 (Sharp Ridge).
    Uses multi-start initialization followed by CMA-ES with adaptive
    covariance matrix to handle the anisotropic ridge structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = clip(x)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start initialization to locate the ridge basin
        n_starts = 15
        for _ in range(n_starts):
            if time.time() > deadline or evaluated >= budget:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_point(x)

        if best_x is None:
            return None, np.inf

        # Phase 2: CMA-ES refinement
        x_mean = best_x.copy()
        f_mean = best_f

        # CMA-ES parameters
        lambda_ = 4  # Population size (small for budget efficiency)

        # Initial sigma
        sigma = 1.0

        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)

        # Evolution path parameters
        c_sigma = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 2) / (dim + 1) - 1) + c_sigma)
        cs = (dim + lambda_ + 2) / (dim + 5 + 3 * lambda_)
        c1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        cmu = min(1 - c1, 2 * ((lambda_ / 2 - 1) + 1) / ((dim + 2) ** 2 + lambda_))

        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Weights for recombination
        weights = np.array([np.log(lambda_ / 2 + 0.5) - np.log(i + 1) for i in range(lambda_)])
        weights /= np.sum(weights)
        mu = lambda_ // 2  # Number of parents

        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)

        # Adjust c1 and cmu for mu_eff
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff - 2) + 1) / ((dim + 2) ** 2 + mu_eff))

        def get_transformation(C):
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            return L

        T = get_transformation(C)

        # Stagnation tracking
        no_improve_count = 0
        max_no_improve = 25
        sigma_restart = 1.5

        while evaluated < budget and time.time() < deadline:
            # Generate population
            pop_x = []
            pop_f = []
            pop_z = []

            for i in range(lambda_):
                if evaluated >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (T @ z)
                x_i = clip(y)
                f_i = eval_point(x_i)
                pop_x.append(x_i)
                pop_f.append(f_i)
                pop_z.append(z)

            # Check if we have a full population
            if len(pop_x) < lambda_:
                break

            # Sort by fitness
            indices = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]
            pop_z = [pop_z[i] for i in indices]

            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop_x[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Save old mean for covariance update
            x_mean_old = x_mean.copy()

            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * pop_x[i]
            x_mean = x_new

            # Compute z_mean in the transformed space
            z_mean = np.zeros(dim)
            for i in range(mu):
                z_mean += weights[i] * pop_z[i]

            # Update evolution path ps
            # ps is in the original space
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                T @ z_mean
            )
            ps_norm = np.linalg.norm(ps)
            h_sigma = 1 if ps_norm / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * np.sqrt(
                (dim + 1) / dim - 1
            ) else 0

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / np.sqrt(dim) - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Update evolution path pc
            pc = (1 - cs) * pc + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * (
                T @ z_mean
            )

            # Update covariance matrix
            # Rank-1 update
            delta_mean = x_mean - x_mean_old
            C = (1 + c1 * (h_sigma * (2 - cs) - 1) + cmu) * C - c1 * np.outer(delta_mean, delta_mean)

            # Rank-mu update
            for i in range(mu):
                diff_i = pop_x[i] - x_mean_old
                C += cmu * weights[i] * np.outer(diff_i, diff_i)

            # Ensure C is positive definite
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                # Regularize
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_inv = np.linalg.inv(C)

            # Recompute transformation
            T = get_transformation(C)

            # Check for stagnation and restart if needed
            if no_improve_count >= max_no_improve and sigma < 0.3:
                # Restart from a point near the best
                x_mean = best_x + 0.5 * rng.standard_normal(dim)
                x_mean = clip(x_mean)
                f_mean = eval_point(x_mean)
                sigma = sigma_restart
                C = np.eye(dim)
                C_inv = np.eye(dim)
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                T = get_transformation(C)
                no_improve_count = 0

        return best_x, best_f
