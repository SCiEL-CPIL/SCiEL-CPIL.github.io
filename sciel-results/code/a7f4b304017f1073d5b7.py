"""Hybrid CMA-ES + Nelder-Mead optimizer for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

class Optimizer:
    """
    Optimizer for BBOB f22 (Gallagher 21 peaks).
    Strategy: 
    1. Global search with CMA-ES to find promising basins.
    2. Local refinement with Nelder-Mead and Coordinate Descent.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin before 300s kill
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        n = self.dim

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if not check_time():
                return np.inf
            calls_made += 1
            try:
                fx = func(x)
                fx = float(fx)
            except Exception:
                return np.inf
            if not np.isfinite(fx):
                fx = np.inf
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float).copy()
            return fx

        def nelder_mead(x_start, f_start, max_steps=60, initial_delta=0.5):
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += initial_delta
                xi = np.clip(xi, -5, 5)
                if np.allclose(xi, x_start):
                    xi[i] = x_start[i] + initial_delta * 0.1
                    xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f_start]
            for i in range(1, dim + 1):
                if not check_time() or calls_made >= self.budget:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)

            for step in range(max_steps):
                if not check_time() or calls_made >= self.budget:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                if not check_time() or calls_made >= self.budget:
                    break

                if fxr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or calls_made >= self.budget:
                                return
                            simplex_f[i] = safe_func(simplex[i])

                if np.max(simplex_f) - np.min(simplex_f) < 1e-14:
                    break

        def coo_local_search(x0, f0, max_iter=40):
            """Coordinate descent local search."""
            nonlocal calls_made, best_x, best_f
            x = x0.copy()
            f = f0
            step = 0.5
            for it in range(max_iter):
                if not check_time() or calls_made >= self.budget:
                    break
                improved = False
                for i in range(self.dim):
                    if not check_time() or calls_made >= self.budget:
                        break
                    for direction in [1, -1]:
                        x_trial = x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = safe_func(x_trial)
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        def cma_es(x0, sigma0, max_calls):
            """
            CMA-ES implementation.
            Returns best_x, best_f found locally.
            """
            nonlocal calls_made, best_x, best_f
            dim = n
            # Population size
            lambda_ = 4 + int(3 * np.log(dim))
            if lambda_ < 4:
                lambda_ = 4
            mu = lambda_ // 2

            # Weights
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            
            # Adaptation rates
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = (2 - cc + 2 * mueff / (dim - 1)) / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

            # State
            xmean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(dim)
            C_eigval = np.ones(dim)
            C_eigvec = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))
            hs = (1.0 + (1.5 + 1.0 / dim) * cc) / (1.0 - cc)

            best_x_local = xmean.copy()
            best_f_local = np.inf
            calls_used = 0
            no_improve = 0
            max_no_improve = 20 # Stagnation detection

            gen = 0
            max_gen = max_calls // lambda_ + 5
            
            # Initial evaluation of x0
            f0 = safe_func(xmean)
            calls_used += 1
            if f0 < best_f_local:
                best_f_local = f0
                best_x_local = xmean.copy()

            while calls_used < max_calls and gen < max_gen and check_time() and calls_made < self.budget:
                gen += 1
                # Generate population
                offspring = []
                for i in range(lambda_):
                    if calls_made + calls_used >= self.budget or not check_time():
                        break
                    z = np.random.randn(dim)
                    # Transform z using eigendecomposition
                    z_trans = C_eigvec @ (np.sqrt(C_eigval) * z)
                    x_trial = xmean + sigma * z_trans
                    x_trial = np.clip(x_trial, -5, 5)
                    offspring.append(x_trial)

                if len(offspring) < mu:
                    break

                # Evaluate
                fvals = []
                for x_trial in offspring:
                    if calls_made + calls_used >= self.budget or not check_time():
                        break
                    fx = safe_func(x_trial)
                    calls_used += 1
                    fvals.append(fx)
                    if fx < best_f_local:
                        best_f_local = fx
                        best_x_local = x_trial.copy()
                        no_improve = 0
                    else:
                        no_improve += 1

                if len(fvals) < mu:
                    break

                # Sort
                idx = np.argsort(fvals)
                offspring = [offspring[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Update mean
                xnew = np.sum([w[i] * offspring[i] for i in range(mu)], axis=0)
                
                # Update covariance
                v = (xnew - xmean) / sigma
                pc = (1 - cc) * pc + cc * v
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * v
                
                # Check termination
                if np.linalg.norm(ps) / chi_n > 1.4 + 2.0 / (dim + 1):
                    # Sigma increase
                    C = C + (cmu + cs * (1 - cs) * (1 + 2 * mueff / dim) - c1) * np.outer(pc, pc)
                    # Actually, standard CMA-ES logic for ps check is for termination or sigma adaptation
                    # For simplicity and robustness in short budget, we stick to standard updates
                    pass

                # Standard CMA-ES updates
                C = (1 + c1 + cmu - c1 - cs * (2 - cs)) * C + c1 * np.outer(pc, pc) + cmu * np.sum([w[i] * np.outer(v_i, v_i) for i, v_i in enumerate([(offspring[i]-xmean)/sigma for i in range(mu)])], axis=0)
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Eigen decomposition
                C_eigval, C_eigvec = np.linalg.eigh(C)
                C_eigval = np.maximum(C_eigval, 1e-10) # Numerical stability
                
                # Update sigma
                sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update mean
                xmean = xnew
                
                # Update global best
                if best_f_local < best_f:
                    best_f = best_f_local
                    best_x = best_x_local.copy()

                if no_improve > max_no_improve:
                    # Restart or break? With small budget, breaking might be safer to save calls for local search
                    # But CMA-ES is robust. Let's just break if we are very close to budget limit
                    if calls_used > max_calls * 0.8:
                        break

            return best_x_local, best_f_local

        # Phase 1: Global Exploration with CMA-ES
        # Allocate ~70% of budget to CMA-ES
        cma_budget = int(self.budget * 0.7)
        x0 = np.zeros(n)
        # Start CMA-ES from center with moderate sigma
        cma_es(x0, sigma0=1.0, max_calls=cma_budget)

        # Phase 2: Local Refinement with Nelder-Mead
        if check_time() and calls_made < self.budget:
            remaining = self.budget - calls_made
            nm_budget = min(100, remaining)
            if nm_budget > 10:
                nelder_mead(best_x.copy(), best_f, max_steps=nm_budget, initial_delta=0.5)

        # Phase 3: Coordinate Descent Polish
        if check_time() and calls_made < self.budget:
            remaining = self.budget - calls_made
            cd_budget = min(50, remaining)
            if cd_budget > 5:
                coo_local_search(best_x.copy(), best_f, max_iter=cd_budget)

        return best_x, best_f

# EVOLVE-BLOCK-END
