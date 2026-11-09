"""Combined program: CMA-ES with IPOP restarts and multi-start, plus (1+1)-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with IPOP restarts and (1+1)-ES refinement for BBOB f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # 300s limit, 10s safety margin
        dim = self.dim
        budget = self.budget

        # Wrap func to track calls and time
        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        def run_cma_es(x0, sigma0, max_evals, pop_size=None):
            """Run CMA-ES from x0 with sigma0 for up to max_evals evaluations."""
            nonlocal call_count, best_f, best_x, deadline, budget

            n = dim
            if pop_size is None:
                pop_size = 4 + int(3 * np.log(n))
            lam = pop_size
            mu = lam // 2
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            csig = (mueff + 2) / (n + mueff + 5)
            dsig = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + csig
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((n + 2) ** 2 + mueff))

            xmean = x0.copy().astype(np.float64)
            sigma = sigma0
            C = np.eye(n)
            Cinv = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chn = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            Cchol = np.linalg.cholesky(C)

            evals = 0

            while evals < max_evals:
                if time.time() >= deadline or call_count >= budget:
                    break

                # Generate and evaluate population
                pop = []
                for i in range(lam):
                    if time.time() >= deadline or call_count >= budget:
                        break
                    y = rng.standard_normal(n)
                    z = Cchol @ y
                    xi = np.clip(xmean + sigma * z, -5.0, 5.0)
                    fi = wrapped_func(xi)
                    evals += 1
                    pop.append((fi, xi))

                if len(pop) < mu:
                    break

                pop.sort(key=lambda t: t[0])
                sel = [p[1] for p in pop[:mu]]
                sel_f = [p[0] for p in pop[:mu]]

                if sel_f[0] < best_f:
                    best_f = sel_f[0]
                    best_x = sel[0].copy()

                xnew = np.sum(w[:, None] * np.array(sel), axis=0)
                xold = xmean.copy()
                diff = xnew - xold

                # Step size update
                ps = (1 - csig) * ps + np.sqrt(csig * (2 - csig) * mueff) / chn * (Cinv @ diff) / sigma
                sigma *= np.exp(csig / dsig * (np.linalg.norm(ps) / chn - 1))
                sigma = np.clip(sigma, 1e-12, 1e4)

                xmean = xnew

                # Covariance update
                if np.linalg.norm(ps) < (1.4 + 1.0 / csig) * chn:
                    v = diff / sigma
                    pc = (1 - cc) * pc + cc * v
                    r1 = np.outer(pc, pc)
                else:
                    r1 = np.eye(n)

                delta_h = (1 - cc) / cc
                delta_h = (1 + delta_h) * (1 - cc) / cc
                C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * (r1 - C) + cmu * np.sum(
                    w[:, None, None] * np.outer(np.array(sel) - xold, np.array(sel) - xold), axis=0
                ) / sigma ** 2

                # Ensure symmetry
                C = (C + C.T) / 2

                # Recompute Cholesky
                try:
                    Cchol = np.linalg.cholesky(C)
                    Cinv = np.linalg.inv(C)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    Cchol = np.eye(n)
                    Cinv = np.eye(n)

                # IPOP restart condition
                if evals >= 5 * lam and sigma < 1e-8:
                    break

            return xmean, sigma

        def run_11es(x0, sigma0, max_evals, window_size=15):
            """Run (1+1)-ES with 1/5 success rule adaptation."""
            nonlocal call_count, best_f, best_x, deadline, budget

            x = x0.copy()
            fx = wrapped_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = sigma0
            successes = []

            for step in range(max_evals):
                if time.time() >= deadline or call_count >= budget:
                    break

                # Generate candidate
                step_dir = rng.standard_normal(dim)
                candidate = np.clip(x + sigma * step_dir, -5, 5)
                f_candidate = wrapped_func(candidate)

                if f_candidate < fx:
                    # Success
                    x = candidate
                    fx = f_candidate
                    successes.append(1)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    successes.append(0)

                # Maintain window
                if len(successes) > window_size:
                    successes.pop(0)

                # Adapt sigma based on 1/5 success rule
                if len(successes) == window_size:
                    success_rate = sum(successes) / window_size
                    if success_rate > 0.2:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.9

                # Stop if sigma too small
                if sigma < 1e-8:
                    break

        # Phase 1: CMA-ES with IPOP restarts
        # Start with a few diverse initial points
        n_starts = 3
        starts = [np.zeros(dim)]
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-3, 3, dim))

        # Allocate budget: 70% to CMA-ES, 30% to refinement
        alloc_cma = int(budget * 0.7)
        steps_per_cma_start = alloc_cma // n_starts

        for i, x0 in enumerate(starts):
            if time.time() >= deadline or call_count >= budget:
                break

            remaining_for_start = min(steps_per_cma_start, budget - call_count)
            if remaining_for_start < 20:
                break

            # Run CMA-ES from this start
            run_cma_es(x0, 1.0, remaining_for_start)

        # Phase 2: Refinement with (1+1)-ES on the best point found
        if best_x is not None and call_count < budget:
            remaining = budget - call_count
            if remaining > 10:
                # Start refinement with smaller sigma
                run_11es(best_x, 0.1, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
