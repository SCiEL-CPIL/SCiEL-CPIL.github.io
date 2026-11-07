"""Hybrid optimizer for BBOB f20 (Schwefel): CMA-ES global search + Nelder-Mead local refinement.

Strategy:
  1. Use CMA-ES as the primary global search method. CMA-ES adapts its sampling
     distribution (mean, covariance, step size) to the landscape, which is
     particularly effective for multimodal, rotated, and shifted functions like
     Schwefel.
  2. After CMA-ES converges or the global budget is exhausted, switch to
     Nelder-Mead simplex method for high-precision local refinement around the
     best point found.
  3. Multi-start: run several short CMA-ES instances from diverse starting
     points, then refine the global best with Nelder-Mead.
  4. Strict wall-clock deadline guard with safety margin.
"""
import numpy as np
import time


class Optimizer:
    """CMA-ES + Nelder-Mead hybrid optimizer for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 330s kill

        call_count = [0]

        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_check():
            return time.time() < deadline - 5.0

        def budget_check():
            return call_count[0] < budget - 5

        # ===================== CMA-ES =====================
        def cma_es(start_mean, initial_sigma, max_iter):
            """Run CMA-ES from start_mean."""
            nonlocal best_x, best_f

            n = dim
            lam = max(4, 4 + int(3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

            mean = start_mean.copy()
            sigma = initial_sigma
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            f_mean = make_call(mean)
            if f_mean is None:
                return
            update_best(mean, f_mean)

            for it in range(max_iter):
                if not budget_check() or not time_check():
                    break

                # Sample points
                points = []
                z_list = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = B @ (D * z)
                    x = mean + sigma * y
                    x = np.clip(x, -5.0, 5.0)
                    points.append(x)
                    z_list.append(z)

                # Evaluate points
                fvals = []
                for x in points:
                    f = make_call(x)
                    if f is None:
                        break
                    fvals.append(f)
                    update_best(x, f)

                if len(fvals) < lam:
                    break

                # Sort
                idx = np.argsort(fvals)
                points = [points[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                z_sorted = [z_list[i] for i in idx]

                # Update mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * points[i]

                old_mean = mean.copy()
                old_B = B.copy()
                old_D = D.copy()

                diff = new_mean - old_mean
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (diff / sigma)

                norm_p_sigma = np.linalg.norm(p_sigma)
                h_sigma = 1.0 if norm_p_sigma / chi_n < (1 + 2 / (n + 1)) * c_1 else 0.0

                diff_z = (diff / old_D) @ old_B.T
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff_z

                y_pc = old_B @ (old_D * p_c)
                rank1_term = np.outer(y_pc, y_pc)

                sum_yy = np.zeros((n, n))
                for i in range(mu):
                    y_i = (points[i] - old_mean) / sigma
                    sum_yy += weights[i] * np.outer(y_i, y_i)

                C = (1 - c_1 - c_mu) * C + c_1 * (rank1_term + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum_yy
                C = (C + C.T) / 2.0

                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)

                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                mean = new_mean

                if np.linalg.norm(mean - old_mean) < 1e-10 and sigma < 1e-10:
                    break

        # ===================== Nelder-Mead =====================
        def nelder_mead(start_x, max_iter=50):
            """Run Nelder-Mead simplex method from start_x."""
            nonlocal best_x, best_f

            # Initialize simplex
            simplex = []
            f_start = make_call(start_x)
            if f_start is None:
                return
            update_best(start_x, f_start)
            simplex.append((start_x.copy(), f_start))

            for i in range(dim):
                x_i = start_x.copy()
                step = 0.5
                if x_i[i] + step > 5.0:
                    step = -0.5
                x_i[i] += step
                x_i = np.clip(x_i, -5.0, 5.0)

                f_i = make_call(x_i)
                if f_i is None:
                    return
                update_best(x_i, f_i)
                simplex.append((x_i.copy(), f_i))

                if not time_check():
                    return

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            shrink = 0.5

            for _ in range(max_iter):
                if not budget_check() or not time_check():
                    break

                simplex.sort(key=lambda t: t[1])

                x_worst = simplex[-1][0]
                f_worst = simplex[-1][1]
                x_best = simplex[0][0]
                f_best = simplex[0][1]

                if f_worst - f_best < 1e-10:
                    break

                x_centroid = np.mean([x for x, _ in simplex[:-1]], axis=0)

                # Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = make_call(x_ref)
                if f_ref is None:
                    break
                update_best(x_ref, f_ref)

                if f_ref < f_best:
                    # Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = make_call(x_exp)
                    if f_exp is None:
                        break
                    update_best(x_exp, f_exp)

                    if f_exp < f_ref:
                        simplex[-1] = (x_exp.copy(), f_exp)
                    else:
                        simplex[-1] = (x_ref.copy(), f_ref)
                elif f_ref < simplex[-2][1]:
                    # Accept reflection
                    simplex[-1] = (x_ref.copy(), f_ref)
                else:
                    # Contraction
                    if f_ref < f_worst:
                        # Outside contraction
                        x_con = x_centroid + rho * (x_ref - x_centroid)
                    else:
                        # Inside contraction
                        x_con = x_centroid - rho * (x_centroid - x_worst)

                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = make_call(x_con)
                    if f_con is None:
                        break
                    update_best(x_con, f_con)

                    if f_con < min(f_ref, f_worst):
                        simplex[-1] = (x_con.copy(), f_con)
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = (x_best + shrink * (simplex[i][0] - x_best), simplex[i][1])
                            f_sh = make_call(simplex[i][0])
                            if f_sh is None:
                                break
                            update_best(simplex[i][0], f_sh)
                            simplex[i] = (simplex[i][0].copy(), f_sh)
                        if not budget_check() or not time_check():
                            break

        # ===================== Main orchestration =====================

        # Phase 1: CMA-ES global search with multi-start
        # Allocate ~70% of budget to CMA-ES, ~30% to Nelder-Mead refinement
        cma_budget = int(budget * 0.7)
        
        # Generate diverse starting points for CMA-ES
        start_points = []
        
        # 1. Origin
        start_points.append(np.zeros(dim))
        
        # 2. Points near typical Schwefel minima (around 4.20968 in each dim)
        base = 4.20968
        for _ in range(5):
            p = np.full(dim, base)
            p += rng.normal(0, 0.5, dim)
            p = np.clip(p, -5.0, 5.0)
            start_points.append(p)
        
        # 3. Random points
        for _ in range(10):
            start_points.append(rng.uniform(-5, 5, dim))
        
        # 4. Points at various scales
        for scale in [1.0, 2.0, 3.0, 4.0]:
            for _ in range(3):
                p = rng.uniform(-scale, scale, dim)
                p = np.clip(p, -5.0, 5.0)
                start_points.append(p)
        
        # Shuffle to mix
        rng.shuffle(start_points)
        
        # Run CMA-ES from each starting point with limited iterations
        n_cma_restarts = len(start_points)
        # Each CMA-ES iteration costs lam + 1 calls (mean + lam samples)
        # For dim=5, lam=12, so ~13 calls per iteration
        # Budget for CMA-ES is cma_budget, so max_iter per restart ~ cma_budget / (13 * n_restarts)
        # But we want to spread budget: use shorter runs for more restarts
        max_iter_per_cma = max(5, min(30, cma_budget // (13 * max(1, n_cma_restarts))))
        
        for i in range(n_cma_restarts):
            if not budget_check() or not time_check():
                break
            
            start_x = start_points[i].copy()
            # Use a reasonable initial sigma
            initial_sigma = 2.0
            cma_es(start_x, initial_sigma, max_iter_per_cma)
        
        # Phase 2: Nelder-Mead local refinement
        # Use remaining budget for Nelder-Mead refinement around the best point
        if best_x is not None and budget_check() and time_check():
            # Refine around the global best
            nelder_mead(best_x, max_iter=30)
            
            # If budget allows, try a few more refinements from slightly perturbed points
            remaining = budget - call_count[0]
            if remaining > 30 and time_check():
                for _ in range(3):
                    if not budget_check() or not time_check():
                        break
                    refine_x = best_x.copy()
                    refine_x += rng.normal(0, 0.1, dim)
                    refine_x = np.clip(refine_x, -5.0, 5.0)
                    nelder_mead(refine_x, max_iter=15)
        
        return best_x, best_f
