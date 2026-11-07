"""Combined optimizer: Rastrigin-aware exploration with CMA-ES and coordinate refinement."""
import numpy as np
import time


class Optimizer:
    """Rastrigin-aware optimizer combining lattice exploration, CMA-ES, and coordinate descent."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill

        best_x = None
        best_f = np.inf
        calls_used = 0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def try_point(x):
            nonlocal best_x, best_f, calls_used
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return calls_used < budget and time.time() < deadline

        def cma_es_refine(x0, sigma_init=0.5):
            """CMA-ES-like local refinement with simplified covariance adaptation."""
            nonlocal best_x, best_f, calls_used
            n = dim
            x = np.array(x0, dtype=float)
            sigma = sigma_init
            pop_size = max(4, int(4 + 3 * np.log(n)))
            C = np.eye(n)
            ps = np.zeros(n)
            
            # Evolution path parameters
            cs = (pop_size + 2) / (n + pop_size + 4)
            cmu = 2 / ((n + 2) ** 2 + pop_size)
            c_sigma = cs
            c_c = 2 * (cs - 1) + 1
            hc = 1.0 + 2.0 / (n + 1) - 1 / (n + 1)
            
            max_iter = 50
            for iteration in range(max_iter):
                if not time_ok():
                    break
                if calls_used >= budget:
                    break

                # Generate population
                offspring = []
                for i in range(pop_size):
                    if not time_ok() or calls_used >= budget:
                        break
                    z = rng.standard_normal(n)
                    # Simple transformation using current C (identity initially, adapted later)
                    xi = x + sigma * z
                    fi = try_point(xi)
                    offspring.append((fi, xi))
                
                if len(offspring) < pop_size:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                
                # Update best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()
                
                # Weighted recombination
                w = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
                w /= np.sum(w)
                mu_eff = 1.0 / np.sum(w ** 2)
                
                # Update mean
                x_new = np.zeros(n)
                for i in range(pop_size):
                    x_new += w[i] * offspring[i][1]
                
                # Check convergence
                if np.linalg.norm(x_new - x) < 1e-10 * sigma:
                    break
                
                # Update evolution paths
                y = (x_new - x) / sigma
                ps = (1 - c_sigma) * ps + c_sigma * y
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (iteration + 1))
                        < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1 - 1))
                
                if hsig:
                    ps = (1 - c_sigma) * ps + c_sigma * y
                else:
                    ps = (1 - c_sigma) * ps
                
                # Update covariance
                delta_hsig = (1 - hsig) * c_sigma * (2 - c_sigma)
                C = (1 + cmu * delta_hsig - cmu - c_sigma * (2 - c_sigma)) * C
                for i in range(pop_size):
                    vi = (offspring[i][1] - x) / sigma
                    C += cmu * w[i] * np.outer(vi, vi)
                C += cmu * delta_hsig * np.outer(ps, ps)
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(c_sigma / (5 * n) * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                x = x_new
                
                if sigma < 1e-6:
                    break
            
            return best_x, best_f

        # Phase 1: Coarse exploration exploiting Rastrigin structure
        n_explore = 50
        for i in range(n_explore):
            if not time_ok():
                break
            if i < 25:
                # Random exploration
                x = rng.uniform(-4.5, 4.5, dim)
            elif i < 37:
                # Integer-like points with small noise (Rastrigin minima at integers)
                x_int = rng.integers(-4, 5, dim).astype(float)
                x = x_int + rng.uniform(-0.2, 0.2, dim)
            else:
                # Half-integer points
                x_half = (rng.integers(-4, 5, dim) + 0.5).astype(float)
                x = x_half + rng.uniform(-0.1, 0.1, dim)
            try_point(x)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = try_point(best_x)

        # Phase 2: Coordinate descent sweeps to quickly find basin
        for step in [1.0, 0.5, 0.25]:
            if not time_ok():
                break
            improved = True
            max_sweeps = 3
            for _ in range(max_sweeps):
                if not time_ok():
                    break
                improved = False
                for d in range(dim):
                    if not time_ok():
                        break
                    for sign in [1, -1]:
                        if not time_ok():
                            break
                        x_trial = best_x.copy()
                        x_trial[d] += sign * step
                        f_trial = try_point(x_trial)
                        if f_trial < best_f:
                            improved = True

        # Phase 3: CMA-ES refinement
        if time_ok():
            cma_es_refine(best_x, sigma_init=0.3)

        # Phase 4: Fine-tuning with very small coordinate steps
        for ss in [0.05, 0.02, 0.01, 0.005]:
            if not time_ok():
                break
            improved = True
            max_fine = 2
            for _ in range(max_fine):
                if not time_ok():
                    break
                improved = False
                for d in range(dim):
                    if not time_ok():
                        break
                    for sign in [1, -1]:
                        if not time_ok():
                            break
                        x_trial = best_x.copy()
                        x_trial[d] += sign * ss
                        f_trial = try_point(x_trial)
                        if f_trial < best_f:
                            improved = True
                if not improved:
                    break

        return best_x, best_f
