"""Multi-start optimizer: Sobol global + Nelder-Mead fast local + CMA-ES adaptive refinement for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.stats import qmc


class Optimizer:
    """Multi-start optimizer for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def time_left():
            return deadline - time.time()

        def make_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global exploration with Sobol sequence
        n_global = 40
        if n_global > budget:
            n_global = budget

        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        m = 1
        while m < n_global:
            m *= 2
        sobol_points = sobol.random(m)
        sobol_points = sobol_points * 10.0 - 5.0

        # Track all evaluated points for multi-start
        evaluated_points = []
        evaluated_fvals = []

        for i in range(n_global):
            if calls_made >= budget or time_left() < 5:
                break
            x = sobol_points[i]
            f = make_call(x)
            if f is not None:
                evaluated_points.append(x.copy())
                evaluated_fvals.append(f)

        if best_x is None:
            make_call(np.zeros(dim))
            return best_x, best_f

        # Sort evaluated points by fitness (best first)
        if len(evaluated_points) > 0:
            sort_idx = np.argsort(evaluated_fvals)
            evaluated_points = [evaluated_points[i] for i in sort_idx]
            evaluated_fvals = [evaluated_fvals[i] for i in sort_idx]

        # Phase 2: Fast Nelder-Mead refinement on top candidates
        # This gives rapid early convergence for AOCC

        def nelder_mead(start_x, budget_nm, sigma_init=0.5):
            nonlocal calls_made, best_x, best_f

            if calls_made >= budget or time_left() < 2:
                return

            # Initialize simplex
            simplex = [start_x.copy()]
            for i in range(dim):
                vertex = start_x.copy()
                vertex[i] += sigma_init
                simplex.append(vertex)

            # Evaluate initial simplex
            f_values = []
            for i in range(dim + 1):
                if calls_made >= budget or time_left() < 1:
                    return
                f = make_call(simplex[i])
                if f is None:
                    return
                f_values.append(f)

            # Nelder-Mead loop
            max_iter = 200
            for _ in range(max_iter):
                if calls_made >= budget or time_left() < 1:
                    break

                # Sort simplex by function value
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]

                # Check convergence
                if abs(f_values[-1] - f_values[0]) < 1e-12:
                    break

                # Compute centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = make_call(x_r)
                if f_r is None:
                    break

                if f_r < f_values[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = make_call(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Reflection accepted
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    beta = 0.5
                    if f_r < f_values[-1]:
                        x_c = centroid + beta * (x_r - centroid)
                    else:
                        x_c = centroid - beta * (x_r - centroid)

                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = make_call(x_c)
                    if f_c is None:
                        break

                    if f_c < f_values[-1]:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls_made >= budget or time_left() < 1:
                                return
                            f_shrink = make_call(simplex[i])
                            if f_shrink is None:
                                return
                            f_values[i] = f_shrink

                        if np.max(f_values) - np.min(f_values) < 1e-10:
                            break

        # Run Nelder-Mead on top 3 candidates from Phase 1
        n_nm_starts = min(3, len(evaluated_points))
        for i in range(n_nm_starts):
            if calls_made >= budget or time_left() < 5:
                break
            nelder_mead(evaluated_points[i].copy(), budget - calls_made)

        # Phase 3: CMA-ES adaptive refinement on the current best
        # CMA-ES handles rotated landscapes better than Nelder-Mead
        if calls_made < budget and time_left() > 10:
            cma_es(best_x.copy(), budget - calls_made)

        return best_x, best_f

    def cma_es(self, start_x, remaining_budget):
        """Lightweight CMA-ES for adaptive refinement."""
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 15  # population size
        mu = lam // 2

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)

        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Initialize CMA-ES
        m = start_x.copy()
        sigma = 0.3  # smaller initial step for local refinement

        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        dcor = 1.0 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        # Eigen decomposition
        C_eigvals, C_eigvecs = np.linalg.eigh(C)

        def update_eig():
            nonlocal C, C_eigvals, C_eigvecs
            C_eigvals, C_eigvecs = np.linalg.eigh(C)
            C_eigvals = np.maximum(C_eigvals, 1e-12)

        rng = np.random.default_rng(123)

        # CMA-ES loop
        while calls_made < budget and time_left() > 2:
            # Compute square root of C
            sqrt_C_eigvals = np.sqrt(C_eigvals)
            C_sqrt = C_eigvecs @ np.diag(sqrt_C_eigvals)

            # Generate population
            Z = rng.standard_normal((lam, dim))
            Y = m[None, :] + sigma * (C_sqrt @ Z.T).T
            Y = np.clip(Y, -5.0, 5.0)

            # Evaluate population
            F = np.full(lam, np.inf)
            for i in range(lam):
                if calls_made >= budget or time_left() < 1:
                    break
                f = make_call(Y[i])
                if f is not None:
                    F[i] = f

            if calls_made >= budget or time_left() < 1:
                break

            # Check if any evaluation was made
            if np.all(np.isinf(F)):
                break

            # Sort by fitness
            valid = ~np.isinf(F)
            if not np.any(valid):
                break
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]

            # Update mean
            m_old = m.copy()
            valid_idx = [i for i in idx if not np.isinf(F[i])]
            if len(valid_idx) < mu:
                sel_Y = Y[valid_idx]
                sel_w = np.ones(len(valid_idx)) / len(valid_idx)
                m = np.sum(sel_w[:, None] * sel_Y, axis=0)
            else:
                Y_sorted_valid = Y_sorted[:mu]
                m = np.sum(w[:, None] * Y_sorted_valid, axis=0)

            # Update sigma, C, pc, ps
            if sigma > 1e-12:
                y = (m - m_old) / sigma
            else:
                y = np.zeros(dim)

            # Update ps
            hs = (math.sqrt(1 - math.exp(-2 * (cs + 1))) * (dim + 2) / 2 + cs)
            if np.linalg.norm(y) < (1 + 5.0 / (dim + 1)) * hs * math.sqrt(dim) * sigma:
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mu_eff) * y

            # Update pc
            hsig_threshold = 1.5 * math.sqrt(dim) * sigma
            if np.linalg.norm(ps) < hsig_threshold:
                pc = (1 - cc) * pc + math.sqrt(cc * (2 - cc) * mu_eff) * ps / sigma

            # Update C
            norm_ps_sq = np.dot(ps, ps)
            delta_h = (1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs) * (norm_ps_sq / dim - 1)

            rank1 = np.outer(y, y)

            # Rank-mu update
            if len(valid_idx) >= mu:
                Y_selected = Y_sorted[:mu]
            else:
                Y_selected = Y[valid_idx]
                while len(Y_selected) < mu:
                    Y_selected = np.vstack([Y_selected, m_old])

            Y_steps = (Y_selected[:mu] - m_old[None, :]) / max(sigma, 1e-12)
            rank_mu = np.sum(w[:, None, None] * np.einsum('ij,ik->ijk', Y_steps, Y_steps), axis=0)

            C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * rank1 + cmu * rank_mu
            C = (C + C.T) / 2.0
            update_eig()

            # Update sigma
            sigma = sigma * math.exp(cs / dcor * (np.linalg.norm(ps) / math.sqrt(dim) - 1))

            # Convergence check
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
