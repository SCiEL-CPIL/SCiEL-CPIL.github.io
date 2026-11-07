"""CMA-ES with RBF surrogate for BBOB f4 (Büche-Rastrigin)"""
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
        deadline = start_time + 280

        call_count = 0
        best_x = None
        best_f = float('inf')
        all_x = []
        all_f = []

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x_clipped))
            all_x.append(x_clipped.copy())
            all_f.append(f)
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial exploration - stratified + random
        n_init = 25
        points = [np.zeros(dim)]
        for axis in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[axis] = sign * 2.5
                points.append(pt)
        while len(points) < n_init:
            points.append(rng.uniform(-4.0, 4.0, dim))

        for p in points:
            if call_count >= budget or time.time() > deadline:
                break
            try:
                do_call(p)
            except Exception:
                break

        # Phase 2: CMA-ES with restarts
        def cmaes(x0, sigma0, max_iter=80, pop_size=None):
            nonlocal call_count, best_f, best_x, deadline
            if call_count >= budget or time.time() > deadline:
                return
            
            if pop_size is None:
                pop_size = int(4 + 3 * np.log(dim))
            
            # CMA-ES parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # State
            mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Precompute
            def update_chol(C):
                nonlocal C_inv, C_sqrt, C_inv_sqrt
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                C_sqrt = L
                C_inv = np.linalg.inv(C)
                # Inverse square root via eigen
                eigvals, eigvecs = np.linalg.eigh(C_inv)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            update_chol(C)
            
            for gen in range(max_iter):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Sample
                samples = []
                fitnesses = []
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    x_sample = mean + sigma * (C_sqrt @ z)
                    x_sample = np.clip(x_sample, -4.9, 4.9)
                    try:
                        f = do_call(x_sample)
                    except Exception:
                        break
                    samples.append(x_sample)
                    fitnesses.append(f)
                
                if len(samples) < pop_size:
                    break
                
                # Sort by fitness
                order = np.argsort(fitnesses)
                samples = [samples[i] for i in order]
                fitnesses = [fitnesses[i] for i in order]
                
                # Update mean
                old_mean = mean.copy()
                mean = sum(weights[i] * samples[i] for i in range(mu))
                
                # Update sigma
                diff = mean - old_mean
                norm_diff = np.linalg.norm(C_inv_sqrt @ diff)
                if norm_diff > 0:
                    sigma *= np.exp(c_sigma / d_sigma * (norm_diff / chi_n - 1))
                
                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff) / sigma
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 1/dim) * chi_n:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma
                    delta_sig = (np.linalg.norm(pc)**2 / dim - 1) / (dim + 1)
                    C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_sig * C)
                    C += c_mu * sum(weights[i] * np.outer(samples[i] - old_mean, samples[i] - old_mean) for i in range(mu))
                    
                    # Ensure symmetry and positive definite
                    C = (C + C.T) / 2
                    try:
                        update_chol(C)
                    except np.linalg.LinAlgError:
                        C = C + 0.1 * np.eye(dim)
                        C = (C + C.T) / 2
                        try:
                            update_chol(C)
                        except np.linalg.LinAlgError:
                            C = np.eye(dim)
                            update_chol(C)
                
                # Stagnation check
                if gen > 10 and np.std(fitnesses[:mu]) < 1e-10:
                    break

        # Run CMA-ES from best point
        if best_x is not None and call_count < budget - 30 and time.time() < deadline:
            cmaes(best_x, 1.5, max_iter=60, pop_size=10)

        # Phase 3: RBF surrogate-guided exploration
        def rbf_predict(x_new):
            if len(all_x) < 2:
                return 0.0
            X = np.array(all_x)
            F = np.array(all_f)
            x_new = np.asarray(x_new, dtype=float)
            
            # RBF kernel
            eps = 0.5
            def phi(a, b):
                return np.exp(-eps * np.sum((a - b)**2, axis=-1))
            
            # Solve for weights
            K = phi(X[:, None, :], X[None, :, :])
            K += 1e-8 * np.eye(len(X))
            ones = np.ones((len(X), 1))
            A = np.hstack([K, ones])
            b_vec = np.hstack([F, [0.0]])
            try:
                w = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(A, b_vec, rcond=None)[0]
            
            k_new = phi(x_new, X)
            pred = np.sum(w[:len(X)] * k_new) + w[-1]
            return pred

        def rbf_variance(x_new):
            if len(all_x) < 2:
                return 1.0
            X = np.array(all_x)
            x_new = np.asarray(x_new, dtype=float)
            eps = 0.5
            k_new = np.exp(-eps * np.sum((x_new - X)**2, axis=1))
            # Simple variance estimate: distance to nearest points
            dists = np.sum((x_new - X)**2, axis=1)
            return np.exp(-0.5 * np.min(dists))

        # Interleave CMA-ES restarts with RBF-guided proposals
        n_cma_restarts = 4
        for r in range(n_cma_restarts):
            if call_count >= budget - 20 or time.time() > deadline:
                break
            
            # RBF-guided proposal: sample points and pick one with low predicted value
            n_candidates = 15
            candidates = []
            for _ in range(n_candidates):
                if call_count >= budget or time.time() > deadline:
                    break
                # Mix: some near best, some random, some RBF-guided
                if r % 3 == 0:
                    x_cand = best_x + rng.normal(0, 0.8, dim)
                elif r % 3 == 1:
                    x_cand = rng.uniform(-4.0, 4.0, dim)
                else:
                    # RBF-guided: try to minimize predicted value
                    x_cand = rng.uniform(-4.0, 4.0, dim)
                    # Refine with a few local steps
                    for _ in range(3):
                        pred = rbf_predict(x_cand)
                        grad_approx = np.zeros(dim)
                        for i in range(dim):
                            x_plus = x_cand.copy()
                            x_plus[i] += 0.1
                            x_minus = x_cand.copy()
                            x_minus[i] -= 0.1
                            grad_approx[i] = (rbf_predict(x_plus) - rbf_predict(x_minus)) / 0.2
                        x_cand -= 0.3 * grad_approx
                        x_cand = np.clip(x_cand, -4.5, 4.5)
                
                x_cand = np.clip(x_cand, -4.5, 4.5)
                candidates.append(x_cand)
            
            if candidates:
                # Evaluate the best candidate according to RBF
                best_cand_idx = 0
                best_pred = float('inf')
                for i, xc in enumerate(candidates):
                    pred = rbf_predict(xc)
                    var = rbf_variance(xc)
                    score = pred - 0.5 * var  # Lower is better (exploit + explore)
                    if score < best_pred:
                        best_pred = score
                        best_cand_idx = i
                
                x_start = candidates[best_cand_idx]
                try:
                    do_call(x_start)
                except Exception:
                    pass
            
            # CMA-ES from this start
            if call_count < budget - 20 and time.time() < deadline:
                cmaes(x_start, 1.0, max_iter=40, pop_size=8)

        # Phase 4: Final fine-tuning with coordinate descent
        if best_x is not None and call_count < budget - 5 and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            step_size = 0.3
            for _ in range(20):
                if call_count >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += direction * step_size
                        try:
                            f_try = do_call(x_try)
                        except Exception:
                            break
                        if f_try < f:
                            x[i] = x_try[i]
                            f = f_try
                            improved = True
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-5:
                        break
            best_x = x
            best_f = f

        return best_x, best_f
