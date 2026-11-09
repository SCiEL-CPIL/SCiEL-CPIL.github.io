"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES + diverse Nelder-Mead simplex ensemble + Rastrigin-aware refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Initial exploration - 10 random starts to find promising basins
        n_starts = 10
        for i in range(n_starts):
            if call_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            if fx is None:
                break

        # Phase 2: CMA-ES from best starting point, with stagnation detection
        remaining = budget - call_count
        if remaining > 40 and time.time() < deadline:
            x = best_x.copy()
            sigma = 1.5  # initial step size for Rastrigin scale

            # CMA-ES parameters tuned for dim=5
            mu = 3
            lambda_ = 10
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)

            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            stagnation_threshold = 12  # generations without improvement
            recent_best = []
            gen = 0
            while call_count < budget and time.time() < deadline:
                if gen % 50 == 0 and gen > 0:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-10)
                        C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                    except np.linalg.LinAlgError:
                        C_inv_sqrt = np.eye(dim)

                # Generate offspring
                offspring = []
                off_f = []
                for k in range(lambda_):
                    if call_count >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = x + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                    fy = call_func(y)
                    if fy is None:
                        break
                    off_f.append(fy)

                if len(off_f) < mu:
                    break

                # Sort by fitness
                sorted_idx = np.argsort(off_f)
                selected_x = [offspring[i] for i in sorted_idx[:mu]]
                selected_f = [off_f[i] for i in sorted_idx[:mu]]

                # Stagnation tracking
                if selected_f[0] < best_f:
                    recent_best = []
                else:
                    recent_best.append(best_f)

                # If stagnating, hand off to Nelder-Mead ensemble
                if len(recent_best) >= stagnation_threshold and call_count < budget - 60:
                    break

                # Weighted mean
                m = sum(w * xk for w, xk in zip(weights, selected_x))

                # Update evolution paths
                diff = m - x
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ diff)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * diff

                # Sigma adaptation
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) / chi_n
                if h_sigma < (1.4 + 2 / (dim + 1)):
                    sigma *= np.exp(c_sigma / chi_n * (h_sigma - 1))

                # Update covariance
                delta_d = (1 + c_mu * (mueff - 1)) * diff
                rank_mu = np.outer(delta_d, delta_d)
                p_c_norm_sq = np.linalg.norm(p_c)**2
                h_c = 1.0 if p_c_norm_sq > 1 / (1 - c_c)**2 else 0.0
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + h_c * np.eye(dim)) + c_mu * rank_mu

                x = m
                gen += 1

        # Phase 3: Diverse Nelder-Mead simplex ensemble
        # 5 independent simplexes at different basins to escape Rastrigin local minima
        if call_count < budget and time.time() < deadline:
            # Diverse starting points: best_x, center, and corner perturbations
            base_points = [best_x.copy()]
            base_points.append(np.zeros(dim))
            for i in range(3):
                corner = rng.uniform(-4.5, 4.5, dim)
                base_points.append(np.clip(corner, -5, 5))

            alpha = 0.5  # perturbation size for initial simplex
            alpha_nm = 1.0   # reflection
            gamma_nm = 2.0   # expansion
            rho_nm = 0.5     # contraction
            sigma_nm = 0.5   # shrink
            max_nm_iter = 25

            for base_pt in base_points:
                if call_count >= budget or time.time() > deadline:
                    break

                # Initialize Nelder-Mead simplex
                simplex = [base_pt.copy()]
                simplex_f = [call_func(base_pt)]
                if simplex_f[0] is None:
                    break

                for d in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    pt = base_pt.copy()
                    pt[d] += alpha
                    pt = np.clip(pt, -5, 5)
                    fpt = call_func(pt)
                    if fpt is None:
                        break
                    simplex.append(pt)
                    simplex_f.append(fpt)

                if len(simplex) < dim + 1:
                    continue

                # Nelder-Mead iterations
                for nm_iter in range(max_nm_iter):
                    if call_count >= budget or time.time() > deadline:
                        break

                    # Sort simplex by fitness
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    x_ref = centroid + alpha_nm * (centroid - simplex[-1])
                    x_ref = np.clip(x_ref, -5, 5)
                    f_ref = call_func(x_ref)
                    if f_ref is None:
                        break

                    if f_ref < simplex_f[0]:
                        # Expansion
                        x_exp = centroid + gamma_nm * (centroid - simplex[-1])
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = call_func(x_exp)
                        if f_exp is None:
                            break
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            simplex_f[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            simplex_f[-1] = f_ref
                    elif f_ref < simplex_f[-1]:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    else:
                        # Contraction
                        x_con = centroid + rho_nm * (simplex[-1] - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = call_func(x_con)
                        if f_con is None:
                            break
                        if f_con < simplex_f[-1]:
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_shrink = call_func(simplex[i])
                                if f_shrink is None:
                                    break
                                simplex_f[i] = f_shrink
                            if f_shrink is None:
                                break

        # Phase 4: Rastrigin-aware coordinate descent (converge to integer-aligned minima)
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f

            for step in [0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
                if call_count >= budget or time.time() > deadline:
                    break

                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if call_count >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            fx_new = call_func(x_new)
                            if fx_new is None:
                                break
                            if fx_new < fx - 1e-10:
                                x = x_new
                                fx = fx_new
                                improved = True
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
