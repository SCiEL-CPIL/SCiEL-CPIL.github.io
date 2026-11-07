"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with aggressive restarts for multimodal optimization on BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s limit
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

        def cma_es_run(x0, sigma, n_evals, stagnation_threshold=20, restart_interval=50):
            """
            Runs CMA-ES with aggressive restarts. Restarts if no improvement
            for stagnation_threshold steps or every restart_interval steps.
            """
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return

            n = dim
            lam = 10  # Small population for efficiency
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
            gen_count = 0

            for gen in range(n_evals // lam + 10):
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

                gen_count += 1

                # Aggressive restart condition
                if no_improve > stagnation_threshold or (no_improve > 0 and gen_count % restart_interval == 0):
                    if best_x is not None:
                        xmean = best_x.copy() + rng.normal(0, 0.3, size=n)
                        xmean = np.clip(xmean, -5, 5)
                        sigma = max(sigma * 1.5, 0.8)
                        C = np.eye(n)
                        eigvals, eigvecs = np.linalg.eigh(C)
                        D = np.sqrt(np.maximum(eigvals, 1e-12))
                        B = eigvecs
                        Cinv_sqrt = B @ np.diag(1.0 / D) @ B.T
                        ps = np.zeros(n)
                        pc = np.zeros(n)
                        no_improve = 0
                        local_best = best_f
                    else:
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

        # Phase 1: Global Scan for diverse starting points
        n_scan = min(30, budget // 4)
        scan_points = []
        for _ in range(n_scan):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f = call_func(x)
            scan_points.append((x, f))

        # Sort and pick top 3 diverse points
        scan_points.sort(key=lambda t: t[1])

        starts = []
        for x, f in scan_points:
            if len(starts) == 0 or np.linalg.norm(x - starts[-1]) > 1.5:
                starts.append(x)
            if len(starts) >= 3:
                break

        # If we didn't find enough diverse points, just take the best ones
        while len(starts) < 3 and len(scan_points) > len(starts):
            starts.append(scan_points[len(starts)][0])

        # Phase 2: CMA-ES with aggressive restarts on each island
        remaining_budget = budget - calls_made
        n_islands = len(starts)

        if n_islands == 0:
            cma_es_run(np.zeros(dim), 3.0, remaining_budget)
        else:
            # Give each island roughly equal share, with a bit extra for the best
            base_budget = remaining_budget // n_islands
            for i, x0 in enumerate(starts):
                if calls_made >= budget or time.time() > deadline:
                    break
                # Best island gets slightly more budget
                if i == 0:
                    island_budget = base_budget + (remaining_budget % n_islands) + 10
                else:
                    island_budget = base_budget
                sigma0 = 2.0  # Larger initial sigma for better exploration
                cma_es_run(x0.copy(), sigma0, island_budget,
                           stagnation_threshold=20, restart_interval=50)

        # Phase 3: Final local refinement around the best point found
        if calls_made < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 20:
                # Small local CMA-ES refinement
                cma_es_run(best_x.copy(), 0.3, min(remaining, 50),
                           stagnation_threshold=10, restart_interval=30)

        return best_x, best_f
# EVOLVE-BLOCK-END
