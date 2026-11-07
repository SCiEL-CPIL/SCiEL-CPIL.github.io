import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        dim = self.dim
        budget = self.budget

        n_calls = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        # Multi-restart CMA-ES with aggressive restart policy
        # Each restart uses a small budget; restart when stalled
        n_max_restarts = 8
        budget_per_restart = max(50, budget // n_max_restarts)

        # Pre-generate diverse starting points
        rng = np.random.RandomState(42)
        start_points = []
        # First: center
        start_points.append(np.zeros(dim))
        # Then: quasi-random spread across [-4, 4]^dim
        for i in range(1, n_max_restarts):
            p = rng.uniform(-4, 4, dim)
            start_points.append(p)

        # CMA-ES parameters (small population for fast early convergence)
        lam = 6
        mu = lam // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)

        cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        dsigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c1
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Stagnation detection: no improvement in last K evaluations
        stagnation_window = 30
        stagnation_threshold = 1e-10

        for restart_idx in range(n_max_restarts):
            if n_calls >= budget:
                break
            if time.time() > deadline - 5:
                break

            # Initialize CMA-ES state
            x = start_points[restart_idx].copy()
            sigma = 2.0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            DC = np.linalg.cholesky(C)

            # Evaluate initial point
            fx = func(x)
            n_calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # Track stagnation
            last_improvement = fx
            last_improvement_call = n_calls

            # Run CMA-ES until budget or stagnation
            while n_calls < budget:
                if time.time() > deadline - 5:
                    break

                # Check stagnation: if no improvement in stagnation_window calls
                if n_calls - last_improvement_call >= stagnation_window:
                    break

                # Generate population
                Z = np.random.randn(lam, dim)
                Y = Z @ DC.T

                # Evaluate population
                F = np.empty(lam)
                for i in range(lam):
                    if n_calls >= budget:
                        break
                    xi = x + sigma * Y[i]
                    F[i] = func(xi)
                    n_calls += 1
                    if F[i] < best_fx:
                        best_fx = F[i]
                        best_x = xi.copy()
                        last_improvement = F[i]
                        last_improvement_call = n_calls

                if n_calls >= budget:
                    break

                # Sort offspring by fitness
                indices = np.argsort(F)
                Z_sorted = Z[indices]
                Y_sorted = Y[indices]

                # Update mean
                z_bar = np.dot(weights, Z_sorted[:mu])
                mean_step = DC @ z_bar
                x_new = x + sigma * mean_step

                # Update evolution path ps
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_bar

                # Update sigma
                sigma = sigma * np.exp(cs / dsigma * (np.linalg.norm(ps) / chi_N - 1))

                # Update pc
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * mean_step

                # Update C
                h_sig = 1.0 if np.linalg.norm(ps) / chi_N < (1.4 + 2 / (dim + 1)) else 0.0
                Y_sel = Y_sorted[:mu]
                M = np.zeros((dim, dim))
                for i in range(mu):
                    M += weights[i] * np.outer(Y_sel[i], Y_sel[i])

                C = (1 - c1 - cmu * h_sig) * C + c1 * np.outer(pc, pc) + cmu * M
                C = (C + C.T) / 2

                try:
                    DC = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 0.1 * np.eye(dim)
                    try:
                        DC = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        DC = np.eye(dim)

                x = x_new

        # Nelder-Mead polish with remaining budget
        remaining = budget - n_calls
        if remaining > 20 and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                max_nm_calls = min(remaining - 1, 150)
                result = minimize(
                    func, best_x.copy(), method='Nelder-Mead',
                    options={'maxfun': max_nm_calls, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_fx
