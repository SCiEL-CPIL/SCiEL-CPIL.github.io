"""Hybrid CMA-ES + (1+1)-ES with coordinate-descent polish for Schwefel f20."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Hybrid CMA-ES/(1+1)-ES with coordinate-descent polish on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 315.0  # safety margin before 330s kill
        calls = 0

        def F(x):
            nonlocal calls
            calls += 1
            return func(x)

        rng = np.random.default_rng(12345)

        # Start at center (0,...,0) which is often closer to min in [-4,4]^5
        best_x = np.zeros(dim)
        best_f = F(best_x)

        # Quick random scan to find a better starting point
        scan_budget = min(10, budget // 20)
        for _ in range(scan_budget):
            if calls >= budget or time.time() >= deadline - 10.0:
                break
            x_try = rng.uniform(-5, 5, dim)
            f_try = F(x_try)
            if f_try < best_f:
                best_f, best_x = f_try, x_try.copy()

        # ---- Coordinate descent polish (exploits Schwefel separability) ----
        def coord_descent_polish(x_start, f_start, max_calls):
            nonlocal best_x, best_f, calls
            x = x_start.copy()
            fx = f_start
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            
            # Adaptive step sizes per dimension
            steps = np.full(dim, 0.5)
            no_improve = 0
            
            while calls < budget and time.time() < deadline - 10.0:
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() >= deadline - 10.0:
                        break
                    # Try positive and negative steps
                    for sign in [1.0, -1.0]:
                        if calls >= budget or time.time() >= deadline - 10.0:
                            break
                        x_new = x.copy()
                        x_new[d] += sign * steps[d]
                        x_new[d] = np.clip(x_new[d], -5, 5)
                        f_new = F(x_new)
                        if f_new < fx:
                            x = x_new
                            fx = f_new
                            if fx < best_f:
                                best_f, best_x = fx, x.copy()
                            improved = True
                            no_improve = 0
                            # Reduce step slightly on success
                            steps[d] *= 0.9
                        else:
                            # No improvement, reduce step
                            steps[d] *= 0.7
                            if steps[d] < 1e-10:
                                steps[d] = 1e-10
                
                if not improved:
                    no_improve += 1
                    # If no improvement across all dims, shrink all steps
                    steps *= 0.5
                    if np.all(steps < 1e-12) or no_improve > 20:
                        break
                elif calls >= budget:
                    break
                
                # Reset no_improve if we made progress
                if improved:
                    no_improve = 0
            
            return x, fx

        # ---- (1+1)-ES phase for fine-tuning ----
        def es_1plus1(x_start, f_start, max_calls):
            nonlocal best_x, best_f, calls
            x = x_start.copy()
            fx = f_start
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            
            sigma = 0.1
            success_count = 0
            total_tries = 0
            stagnant = 0
            
            while calls < budget and time.time() < deadline - 10.0:
                # Generate one offspring
                z = rng.standard_normal(dim)
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)
                f_new = F(x_new)
                total_tries += 1
                
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    success_count += 1
                    stagnant = 0
                    # 1/5 success rule: adjust sigma
                    if total_tries >= 5:
                        rate = success_count / total_tries
                        if rate > 0.2:
                            sigma *= 1.1
                        elif rate < 0.2:
                            sigma *= 0.9
                        success_count = 0
                        total_tries = 0
                else:
                    stagnant += 1
                    if stagnant > 30:
                        break
                
                # Terminate if sigma too small
                if sigma < 1e-12:
                    break
            
            return x, fx

        # ---- CMA-ES single run helper ----
        def cma_run(x0, f0, budget_left):
            nonlocal best_x, best_f, calls
            n = dim
            # population size
            m = int(4 + 3 * np.log(n))
            # step size: start moderate
            sigma = 0.5
            # covariance
            C = np.eye(n)
            # evolution path
            ps = np.zeros(n)
            pc = np.zeros(n)
            # adaptation rates
            mu = m // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / w.sum()
            c_sigma = (mueff + 2) / ((n + 2) * (mueff + 2))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / ((n + 4) * (4 + mueff / n))
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            B = None
            D = None
            invsqrtC = None

            def update_ds():
                nonlocal B, D, invsqrtC
                ev, V = np.linalg.eigh(C)
                ev = np.maximum(ev, 1e-20)
                D = np.sqrt(ev)
                B = V
                invsqrtC = V @ np.diag(1.0 / np.sqrt(ev)) @ V.T

            update_ds()

            x = x0.copy()
            fx = f0
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            cma_calls_made = 0
            while calls < budget_left and (time.time() < deadline - 15.0):
                # sample offspring
                z = rng.standard_normal((m, n))
                # transform: x + sigma * B D z
                off = z @ (B * D).T
                off *= sigma
                y = x[None, :] + off
                y = np.clip(y, -5, 5)
                # evaluate
                fy = np.empty(m)
                for i in range(m):
                    if calls >= budget_left or time.time() >= deadline - 15.0:
                        break
                    fy[i] = F(y[i])
                # if we ran out, break
                if calls >= budget_left:
                    break
                # sort
                order = np.argsort(fy)
                # update best
                if fy[order[0]] < best_f:
                    best_f = fy[order[0]]
                    best_x = y[order[0]].copy()
                # recompute mean of best mu
                wsum = w.sum()
                mean_new = np.sum(w[:, None] * y[order[:mu]], axis=0) / wsum
                # evolution path sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (invsqrtC @ (mean_new - x) / sigma)
                # sigma update
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = min(sigma, 1e6)
                sigma = max(sigma, 1e-12)
                # covariance update
                yw = y[order[:mu]]
                diff = (yw - x) / sigma
                delta = (mean_new - x) / sigma
                pc = (1 - c_c) * pc + c_c * delta
                outer = np.outer(pc, pc)
                C = (1 + c_1 + 2 * c_mu - c_c) * C + c_1 * (outer - C) \
                    + c_mu * np.sum(w[:, None, None] * diff[:, None, :] * diff[:, None, :], axis=0)
                # recompute
                update_ds()
                # accept mean
                x = mean_new
                fx = fy[order[0]]
                cma_calls_made += m

                # Periodic coordinate descent polish every ~50 calls
                if cma_calls_made % 50 < m:
                    if calls < budget_left and time.time() < deadline - 20.0:
                        x, fx = coord_descent_polish(x, fx, budget_left)
                        # Reset CMA state partially after polish
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                        x = best_x.copy()
                        fx = best_f

                # termination: convergence
                if sigma < 1e-12:
                    break

            return x, fx

        # ---- Main search loop ----
        # Phase 1: CMA-ES with periodic coordinate descent
        cma_budget = int(budget * 0.7)
        x_end, f_end = cma_run(best_x.copy(), best_f, cma_budget)
        
        # Phase 2: (1+1)-ES for fine-tuning
        if calls < budget and time.time() < deadline - 15.0:
            x_end, f_end = es_1plus1(best_x.copy(), best_f, budget)
        
        # Phase 3: Final coordinate descent polish
        if calls < budget and time.time() < deadline - 10.0:
            x_end, f_end = coord_descent_polish(best_x.copy(), best_f, budget)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
