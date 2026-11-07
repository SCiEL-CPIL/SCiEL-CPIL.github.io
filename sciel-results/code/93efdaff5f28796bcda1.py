"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            try:
                x = np.asarray(x, dtype=np.float64).copy()
                if x.shape != (dim,):
                    x = np.clip(x.reshape(dim), -5, 5)
                else:
                    x = np.clip(x, -5, 5)
                val = func(x)
                total_calls += 1
                if np.isfinite(val):
                    val = float(val)
                    if val < best_f:
                        best_f = val
                        best_x = x.copy()
                    return val
                return None
            except Exception:
                total_calls += 1
                return None

        def cma_es(x0, sigma0, max_evals, rng):
            """Run CMA-ES from x0 with initial step size sigma0."""
            nonlocal total_calls, best_x, best_f, deadline

            n = dim
            # CMA-ES parameters
            lam = max(4, int(np.ceil(4 + 3 * np.log(n))))  # population size
            mu = lam // 2  # number of parents
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

            # State
            x_mean = np.array(x0, dtype=np.float64).copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            evals = 0

            while evals < max_evals and total_calls < budget and time.time() < deadline:
                # Generate population
                pops = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    xi = x_mean + sigma * (B @ (D * z))
                    pops.append(xi)

                # Evaluate
                fitnesses = []
                for xi in pops:
                    if total_calls >= budget or time.time() > deadline:
                        break
                    val = safe_call(xi)
                    if val is None:
                        val = np.inf
                    fitnesses.append(val)

                # Sort
                order = np.argsort(fitnesses)
                pops_sorted = [pops[i] for i in order]
                fitnesses_sorted = [fitnesses[i] for i in order]

                # Update mean
                x_mean_new = np.zeros(n)
                for i in range(mu):
                    x_mean_new += weights[i] * pops_sorted[i]
                x_mean = x_mean_new

                # Update covariance matrix
                y_inv = np.zeros((mu, n))
                for i in range(mu):
                    y_inv[i] = (pops_sorted[i] - x_mean) / sigma

                # Update step size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (B @ (D * (y_inv.mean(axis=0))))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(evals / (cs * (n + 1))))) < (1.4 + 2 / (n + 1)) * chi_n

                # Update evolution paths
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_inv.mean(axis=0)

                # Update covariance
                delta = 1 if not hsig else 0
                cov_update = (1 + c1 * delta) * (1 - c1)
                # Rank-1 update
                rank1 = np.outer(pc, pc)
                # Rank-mu update
                rankmu = np.zeros((n, n))
                for i in range(mu):
                    rankmu += weights[i] * np.outer(y_inv[i], y_inv[i])
                rankmu *= cmu / c1 if c1 > 0 else 1

                # Update B and D (eigenvalue decomposition)
                # For efficiency, we use a simplified update
                # In practice, we should do full eigen-decomposition periodically
                # Here we use a heuristic update
                if evals % lam == 0:
                    # Full eigen-decomposition every lam evaluations
                    C = B @ np.diag(D**2) @ B.T
                    C = (1 - c1) * C + c1 * (rank1 + cmu * rankmu)
                    # Ensure symmetry
                    C = (C + C.T) / 2
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    B = eigvecs
                    D = np.sqrt(eigvals)

                # Update step size
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10)

                evals += lam

            return best_x.copy() if best_x is not None else x0, best_f

        def nelder_mead(x0, max_evals=20, rng=None):
            """Nelder-Mead simplex method for local refinement."""
            nonlocal total_calls, best_x, best_f, deadline

            n = dim
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                step = 0.05 if i == 0 else 0.02
                xi[i] += step
                simplex.append(xi)

            # Evaluate initial simplex
            fvals = []
            for xi in simplex:
                if total_calls >= budget or time.time() > deadline:
                    break
                val = safe_call(xi)
                if val is None:
                    val = np.inf
                fvals.append(val)

            evals = n + 1
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink

            while evals < max_evals and total_calls < budget and time.time() < deadline:
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = safe_call(xr)
                if fr is None:
                    fr = np.inf
                evals += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = safe_call(xe)
                    if fe is None:
                        fe = np.inf
                    evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        fc = safe_call(xc)
                        if fc is None:
                            fc = np.inf
                        evals += 1
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                fi = safe_call(simplex[i])
                                if fi is None:
                                    fi = np.inf
                                fvals[i] = fi
                                evals += 1
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = safe_call(xc)
                        if fc is None:
                            fc = np.inf
                        evals += 1
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                fi = safe_call(simplex[i])
                                if fi is None:
                                    fi = np.inf
                                fvals[i] = fi
                                evals += 1

            return best_x.copy() if best_x is not None else x0, best_f

        # Generate initial points using stratified sampling
        rng = np.random.default_rng(42)

        # Start with center
        initial_points = [np.zeros(dim)]

        # Add stratified samples
        n_strata = 15
        for i in range(n_strata):
            for j in range(n_strata):
                x = np.array([(2 * i + 1) / (2 * n_strata) * 10 - 5,
                              (2 * j + 1) / (2 * n_strata) * 10 - 5,
                              rng.uniform(-4, 4),
                              rng.uniform(-4, 4),
                              rng.uniform(-4, 4)])
                initial_points.append(x)

        # Add random points
        for _ in range(10):
            initial_points.append(rng.uniform(-4, 4, size=dim))

        # Sort by distance from center
        initial_points.sort(key=lambda p: np.linalg.norm(p))

        # Phase 1: Evaluate initial points
        for x_init in initial_points:
            if total_calls >= budget or time.time() > deadline:
                break
            x_clipped = np.clip(x_init, -5, 5)
            safe_call(x_clipped)

        # Phase 2: CMA-ES from best point
        if best_x is not None and total_calls < budget and time.time() < deadline:
            remaining = budget - total_calls
            if remaining > 50:
                cma_es(best_x, 1.0, remaining, rng)

        # Phase 3: Nelder-Mead polish
        if best_x is not None and total_calls < budget and time.time() < deadline:
            remaining = budget - total_calls
            if remaining > 10:
                nelder_mead(best_x, min(remaining, 30), rng)

        # Final fallback
        if best_x is None:
            best_x = np.zeros(dim)
            val = safe_call(best_x)
            if val is not None:
                best_f = val

        return best_x, best_f
# EVOLVE-BLOCK-END
