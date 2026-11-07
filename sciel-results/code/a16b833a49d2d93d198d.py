import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        evals = 0

        # CMA-ES parameters tuned for dim=5, budget=500
        # Population size
        lam = 15
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Chi-square approximation
        chi_lambda = np.sqrt(dim * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2)))

        def run_cma_es(x0, sigma0, remaining_budget):
            nonlocal best_x, best_f, evals

            m = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Initial Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            # Track best
            local_best_x = m.copy()
            local_best_f = func(m)
            evals += 1

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            # Stagnation tracking
            no_improve_count = 0
            stagnation_threshold = 10  # Generations without improvement

            gen = 0
            while evals + lam <= remaining_budget and (time.time() - start_time) < deadline:
                gen += 1

                # Sample lambda candidates using Cholesky
                Z = np.random.randn(lam, dim)
                X = m + sigma * (Z @ L.T)
                
                # Clip to box to avoid wasted evaluations on out-of-bounds points
                X = np.clip(X, -5, 5)

                # Evaluate
                F = np.array([func(x) for x in X])
                evals += lam

                # Sort
                order = np.argsort(F)
                F_sorted = F[order]
                X_sorted = X[order]

                # Update best
                if F_sorted[0] < local_best_f:
                    local_best_f = F_sorted[0]
                    local_best_x = X_sorted[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

                # Check for stagnation
                if no_improve_count > stagnation_threshold and gen > 5:
                    break

                # Update mean
                m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

                # Compute C^{-1/2} v using eigendecomposition (robust for dim=5)
                v = (m_new - m) / sigma
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    v_rot = eigvecs.T @ v
                    v_scaled = v_rot / np.sqrt(eigvals)
                    v_inv_half = eigvecs @ v_scaled
                except np.linalg.LinAlgError:
                    v_inv_half = v

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + \
                          np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_inv_half

                # Update sigma
                sigma_norm = np.linalg.norm(p_sigma) / chi_lambda
                sigma = sigma * np.exp(c_sigma / chi_lambda * (sigma_norm - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)

                # h_sigma
                h_sigma_thresh = (1 + 1.4 / mu_eff) * np.sqrt((1 - c_c) * (2 - c_c))
                h_sigma = 1.0 if sigma_norm < h_sigma_thresh else 0.0

                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v_inv_half

                # Update C
                Y = (X_sorted[:mu] - m) / sigma
                M2 = (Y.T * weights) @ Y
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * M2
                C = (C + C.T) / 2

                # Update Cholesky
                try:
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                except np.linalg.LinAlgError:
                    C += 1e-10 * np.eye(dim)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)
                        C = np.eye(dim)

                # Update mean
                m = m_new

            return local_best_x, local_best_f

        def coordinate_descent(x_start, max_evals, tol=1e-4):
            nonlocal best_x, best_f, evals
            
            x = x_start.copy()
            f_x = func(x)
            evals += 1
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()
            
            step = 1.0
            for _ in range(max_evals):
                if (time.time() - start_time) > deadline:
                    break
                improved = False
                for i in range(dim):
                    if evals >= budget:
                        return
                    # Try positive step
                    x_plus = x.copy()
                    x_plus[i] += step
                    x_plus = np.clip(x_plus, -5, 5)
                    f_plus = func(x_plus)
                    evals += 1
                    if f_plus < f_x:
                        x = x_plus
                        f_x = f_plus
                        improved = True
                        if f_x < best_f:
                            best_f = f_x
                            best_x = x.copy()
                        continue
                    # Try negative step
                    x_minus = x.copy()
                    x_minus[i] -= step
                    x_minus = np.clip(x_minus, -5, 5)
                    f_minus = func(x_minus)
                    evals += 1
                    if f_minus < f_x:
                        x = x_minus
                        f_x = f_minus
                        improved = True
                        if f_x < best_f:
                            best_f = f_x
                            best_x = x.copy()
                        continue
                    # No improvement in this coordinate, shrink step
                    step *= 0.5
                    if step < tol:
                        break
                if not improved:
                    break
            return x, f_x

        # Phase 1: Initial CMA-ES run
        x0 = np.zeros(dim)
        sigma0 = 2.0
        run_cma_es(x0, sigma0, budget)

        # Phase 2: If budget remains, perform coordinate descent refinement
        if evals < budget and (time.time() - start_time) < deadline:
            # Allocate remaining budget to coordinate descent, but cap it to ensure we don't waste time
            remaining = budget - evals
            # Use a portion for CD, keep some for a final small CMA-ES if needed? 
            # The prompt suggests CD exploits ill-conditioned structure.
            # Let's use all remaining budget for CD, but CD is cheap.
            # Actually, CD can be very fast. Let's just run it until budget or time runs out.
            # But we need to be careful not to exceed budget.
            # coordinate_descent checks budget internally.
            coordinate_descent(best_x, remaining)

        return best_x, best_f
