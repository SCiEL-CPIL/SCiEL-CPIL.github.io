"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with Island Model for multimodal optimization on BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es_run(x0, sigma, n_evals, max_stagnation=15):
            """
            Runs CMA-ES from x0 with initial sigma for up to n_evals calls.
            Returns early if stagnation occurs.
            """
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return

            n = dim
            # Smaller population for efficiency in small budget
            lam = max(4, int(4 + 3 * np.log(n))) 
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            xmean = x0.copy()
            C = np.eye(n)
            
            # Precompute Cinv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eigvals, 1e-12))
            B = eigvecs
            Cinv_sqrt = B @ np.diag(1.0 / D) @ B.T

            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            evals_used = 0
            no_improve = 0
            local_best = best_f
            sigma_min = 1e-8
            sigma_max = 5.0

            for gen in range(n_evals // lam + 5):
                if calls_made >= budget or time.time() > deadline:
                    break

                if sigma < sigma_min:
                    break

                # Generate population
                ys = []
                for i in range(lam):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    y = xmean + sigma * Cinv_sqrt @ rng.standard_normal(n)
                    y = np.clip(y, -5, 5)
                    fy = call_func(y)
                    ys.append((y, fy))

                if len(ys) < 2:
                    break

                ys.sort(key=lambda t: t[1])

                # Check for improvement
                if ys[0][1] < local_best - 1e-12:
                    local_best = ys[0][1]
                    no_improve = 0
                else:
                    no_improve += 1

                # Restart if stagnating
                if no_improve > max_stagnation:
                    break

                # Update mean
                xnew = np.zeros(n)
                for i in range(mu):
                    xnew += weights[i] * ys[i][0]

                # Update step size control variable
                old_xmean = xmean.copy()
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * Cinv_sqrt @ (xnew - old_xmean)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals_used + 1) / n))
                         < (1.4 + 2 / (n + 1)) * chi_n)

                # Update evolution path
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * Cinv_sqrt @ (xnew - old_xmean)

                xmean = xnew

                # Update covariance matrix
                delta_hsig = (2 - cc) if hsig else 0.0
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((n, n))
                for i in range(mu):
                    zi = Cinv_sqrt @ (ys[i][0] - xmean)
                    rankmu += weights[i] * np.outer(zi, zi)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * rank1 + cmu * rankmu

                # Ensure PSD
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2

                # Update D and B
                eigvals, B = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                Cinv_sqrt = B @ np.diag(1.0 / D) @ B.T

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, sigma_min)
                sigma = min(sigma, sigma_max)

                evals_used += len(ys)

                if evals_used >= n_evals:
                    break

        # Phase 1: Global Scan
        # Use 20% of budget for initial diverse sampling
        n_scan = min(20, budget // 5)
        scan_points = []
        for _ in range(n_scan):
            if calls_made >= budget: break
            x = rng.uniform(-5, 5, size=dim)
            f = call_func(x)
            scan_points.append((x, f))
        
        # Sort and pick top 4 distinct-ish points
        scan_points.sort(key=lambda t: t[1])
        
        # Select starting points for islands
        starts = []
        for x, f in scan_points:
            if len(starts) == 0 or np.linalg.norm(x - starts[-1]) > 1.0:
                starts.append(x)
            if len(starts) >= 4:
                break
        
        # If we didn't find enough diverse points, just take the best ones
        while len(starts) < 4 and len(scan_points) > len(starts):
             starts.append(scan_points[len(starts)][0])

        # Phase 2: Island CMA-ES
        # Distribute remaining budget among islands
        remaining_budget = budget - calls_made
        n_islands = len(starts)
        if n_islands == 0:
             # Fallback: one big run
             cma_es_run(np.zeros(dim), 3.0, remaining_budget, max_stagnation=30)
        else:
            # Give each island roughly equal share, but let them run until stagnation
            budget_per_island = remaining_budget // n_islands + 10
            
            for i, x0 in enumerate(starts):
                if calls_made >= budget or time.time() > deadline:
                    break
                # Use a moderate sigma
                sigma0 = 1.5
                # Run CMA-ES
                cma_es_run(x0.copy(), sigma0, budget_per_island, max_stagnation=12)

        # Phase 3: Local Refinement
        # Use remaining budget for a tight local search on the global best
        if calls_made < budget and best_x is not None and time.time() < deadline:
            # Use a very small sigma and high precision
            # We can run a simple (1+1)-ES or a small CMA-ES
            # Let's use a small CMA-ES with lambda=4
            x0 = best_x.copy()
            sigma0 = 0.1
            cma_es_run(x0, sigma0, budget - calls_made, max_stagnation=5)

        return best_x, best_f
# EVOLVE-BLOCK-END
