import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        calls = 0

        def time_ok():
            return time.time() < deadline

        # Multi-start: evaluate diverse initial points
        n_starts = 6
        starts = []
        # Deterministic diverse starts
        if calls < budget and time_ok():
            x0 = np.zeros(n)
            fx = func(x0)
            calls += 1
            starts.append((x0, fx))
        
        for i in range(1, n_starts):
            if calls >= budget or not time_ok():
                break
            # Mix of different scales and directions
            scale = 2.0 * (i / n_starts) + 0.5
            x0 = rng.uniform(-scale, scale, n)
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            calls += 1
            starts.append((x0, fx))

        if not starts:
            return np.zeros(n), 0.0

        # Pick the best start
        best_idx = min(range(len(starts)), key=lambda i: starts[i][1])
        x = starts[best_idx][0].copy()
        fbest = starts[best_idx][1]

        # CMA-ES parameters - slightly larger population for better exploration
        lam = 8  # 4 + int(3 * log(5)) = 8
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        sigma = 1.5  # Slightly larger initial sigma for better exploration
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        C_sqrt = np.eye(n)
        inv_C_sqrt = np.eye(n)

        stagnation_count = 0
        restart_threshold = 25
        best_f_so_far = fbest
        no_improve_gens = 0
        prev_fbest = fbest

        remaining = budget - calls
        gen = 0
        cma_calls = 0
        
        # Phase 1: CMA-ES for global optimization
        while calls < budget and time_ok():
            # Generate lambda candidates
            offspring = []
            for i in range(lam):
                if calls >= budget or not time_ok():
                    break
                z = rng.standard_normal(n)
                y = C_sqrt @ z
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                cma_calls += 1
                offspring.append((fi, xi, z))

            if len(offspring) < 2:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < fbest:
                fbest = offspring[0][0]
                stagnation_count = 0
                no_improve_gens = 0
            else:
                stagnation_count += 1
                no_improve_gens += 1

            # Check for restart - more nuanced
            if no_improve_gens > restart_threshold:
                # If sigma is small, we're likely converged or stuck
                if sigma < 0.3:
                    # Reset CMA-ES state but keep x
                    sigma = 1.0
                    pc = np.zeros(n)
                    ps = np.zeros(n)
                    C = np.eye(n)
                    C_sqrt = np.eye(n)
                    inv_C_sqrt = np.eye(n)
                    stagnation_count = 0
                    no_improve_gens = 0
                    # Also try a small perturbation
                    x = x + 0.5 * rng.standard_normal(n)
                    x = np.clip(x, -5, 5)
                    continue
                else:
                    # Just reset sigma
                    sigma = max(sigma * 0.5, 0.5)
                    no_improve_gens = 0

            # Update mean
            x_old = x.copy()
            x = np.zeros(n)
            for i in range(mu):
                x += weights[i] * offspring[i][1]

            # Compute y_mean
            if sigma > 1e-12:
                y_mean = (x - x_old) / sigma
            else:
                y_mean = np.zeros(n)

            # Update step size (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_C_sqrt @ y_mean)
            ps_norm = np.linalg.norm(ps)
            h_sigma = 1.0 if ps_norm / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
            if h_sigma:
                sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))

            # Update covariance matrix
            pc = (1 - c_c) * pc + c_c * y_mean
            delta = (1 + 2 * max(0, mu_eff / n - 1)) * c_c * (1 - c_c)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(offspring[i][2], offspring[i][2])
            C = (1 + c_1 * delta - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                inv_C_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                C_sqrt = np.eye(n)
                inv_C_sqrt = np.eye(n)

            # Ensure sigma is reasonable
            sigma = np.clip(sigma, 1e-4, 10.0)
            gen += 1

            # Stop CMA-ES if we've used most of the budget or time is running out
            if calls > budget * 0.7 or (deadline - time.time()) < 30:
                break

        # Phase 2: Nelder-Mead for local refinement
        time_left = deadline - time.time()
        if time_left > 10.0 and calls < budget:
            try:
                def wrapped_func(x):
                    nonlocal calls
                    if calls >= budget or not time_ok():
                        return fbest
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    return fx

                # Use tighter tolerances for better local convergence
                result = scipy.optimize.minimize(
                    wrapped_func, x, method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if result.fun < fbest:
                    fbest = result.fun
                    x = result.x
            except Exception:
                pass

        # Phase 3: Final fine-tuning with very small steps if budget remains
        if calls < budget and time_ok():
            for _ in range(min(20, budget - calls)):
                if not time_ok() or calls >= budget:
                    break
                # Try small perturbations around the best point
                for scale in [0.01, 0.005, 0.001]:
                    if calls >= budget or not time_ok():
                        break
                    perturbation = rng.standard_normal(n) * scale
                    x_try = x + perturbation
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < fbest:
                        fbest = f_try
                        x = x_try.copy()

        return x, fbest
