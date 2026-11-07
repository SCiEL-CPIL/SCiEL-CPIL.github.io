import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Leave 20 seconds margin for safety (evaluator kills at 330s)
        deadline = start_time + 310.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        def nelder_mead(func, x0, calls, best_x, best_f, max_iter=100, initial_step=0.2, deadline=None, time_ok=None):
            """Nelder-Mead simplex method for local refinement."""
            if time_ok is None:
                time_ok = lambda: True
            if deadline is None:
                deadline = float('inf')

            x = x0.copy()
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

            # Initialize simplex
            simplex = [x.copy()]
            for i in range(dim):
                v = x.copy()
                v[i] += initial_step
                simplex.append(v)

            fvals = [f] + [func(v) for v in simplex[1:]]
            calls += dim

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            min_step = 1e-10

            for _ in range(max_iter):
                if not time_ok() or calls >= budget:
                    break

                # Sort simplex by fitness
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Check convergence
                if abs(fvals[0] - fvals[-1]) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
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
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < max(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

                # Check if simplex is too small
                spread = np.max([np.linalg.norm(simplex[0] - simplex[i]) for i in range(1, dim + 1)])
                if spread < min_step:
                    break

            return calls, best_x, best_f

        # Initial evaluation
        x0 = np.zeros(dim)
        best_x = x0.copy()
        best_f = func(x0)
        calls = 1

        # Phase 1: CMA-ES for global exploration
        # Use ~60% of budget for CMA-ES, rest for local refinement
        cma_budget = int(budget * 0.6)

        # CMA-ES Parameters
        lam = 10
        mu = lam // 2

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        # State variables
        x = np.zeros(dim)
        sigma = 2.5  # Larger initial sigma for better exploration
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        last_improvement_call = 0
        stagnation_limit = 120

        while calls < cma_budget and time_ok():
            # Generate population
            y_list = []
            f_list = []

            # Compute B and D from C for sampling
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.clip(eigenvalues, 1e-12, None)
            D = np.sqrt(eigenvalues)
            B = eigenvectors

            for i in range(lam):
                if not time_ok() or calls >= cma_budget:
                    break
                z = np.random.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                y_list.append(y)
                f_list.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    last_improvement_call = calls

            if len(y_list) < mu:
                break

            # Sort by fitness
            idx = np.argsort(f_list)
            y_selected = np.array([y_list[i] for i in idx[:mu]])

            # Weighted mean
            m = np.sum(weights[:, None] * y_selected, axis=0)

            # Update evolution paths
            old_sigma = sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (m - x)))

            # Update pc
            gen_count = max(1, calls // lam)
            h_sigma = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen_count)) < (1.4 + 2 / (dim + 1)) * chi_n

            if h_sigma:
                pc = (1 - c_c) * pc + c_c * (m - x) / sigma
                delta_h = 1 - h_sigma
            else:
                pc = (1 - c_c) * pc
                delta_h = 1

            # Update Covariance Matrix
            C = (1 - c_c * (2 - c_c + delta_h)) * C + c_c * (2 - c_c + delta_h) * (pc @ pc.T)
            C = (C + C.T) / 2.0

            # Update Sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update Mean
            x = m

            # Safety check
            if sigma > 5.0:
                sigma = 5.0
            if sigma < 1e-4:
                break

            # Restart Logic
            if calls - last_improvement_call > stagnation_limit and calls < cma_budget * 0.7:
                x = best_x + np.random.normal(0, 0.5, dim)
                x = np.clip(x, -5, 5)
                sigma = 1.5
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                last_improvement_call = calls

        # Phase 2: Local refinement using Nelder-Mead
        if calls < budget and time_ok():
            calls, best_x, best_f = nelder_mead(func, best_x, calls, best_x, best_f,
                                                  max_iter=150, initial_step=0.15,
                                                  deadline=deadline, time_ok=time_ok)

        # Phase 3: If still have budget, do another round of CMA-ES with smaller sigma
        remaining = budget - calls
        if remaining > 50 and time_ok():
            # Restart CMA-ES from best point with smaller sigma
            x = best_x.copy()
            sigma = 0.5
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            last_improvement_call = calls

            while calls < budget and time_ok():
                y_list = []
                f_list = []

                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-12, None)
                D = np.sqrt(eigenvalues)
                B = eigenvectors

                for i in range(lam):
                    if not time_ok() or calls >= budget:
                        break
                    z = np.random.standard_normal(dim)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    y_list.append(y)
                    f_list.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        last_improvement_call = calls

                if len(y_list) < mu:
                    break

                idx = np.argsort(f_list)
                y_selected = np.array([y_list[i] for i in idx[:mu]])
                m = np.sum(weights[:, None] * y_selected, axis=0)

                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (m - x)))
                gen_count = max(1, (calls - (budget - remaining)) // lam)
                h_sigma = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen_count)) < (1.4 + 2 / (dim + 1)) * chi_n

                if h_sigma:
                    pc = (1 - c_c) * pc + c_c * (m - x) / sigma
                    delta_h = 1 - h_sigma
                else:
                    pc = (1 - c_c) * pc
                    delta_h = 1

                C = (1 - c_c * (2 - c_c + delta_h)) * C + c_c * (2 - c_c + delta_h) * (pc @ pc.T)
                C = (C + C.T) / 2.0
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                x = m

                if sigma > 2.0:
                    sigma = 2.0
                if sigma < 1e-5:
                    break

        return best_x, best_f
