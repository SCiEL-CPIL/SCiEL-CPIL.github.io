"""Hybrid surrogate-guided exploration with multi-start local refinement for BBOB f4."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        call_count = 0
        best_x = np.zeros(dim, dtype=float)
        best_f = float('inf')
        rng = np.random.default_rng(42)

        # Data storage for surrogate
        X_data = []
        Y_data = []

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            X_data.append(x.copy())
            Y_data.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return time.time() < deadline

        # ---------- Simple GP Surrogate ----------
        class GPSurrogate:
            def __init__(self, dim):
                self.dim = dim
                self.l = np.ones(dim) / 1.5  # length scales
                self.s2 = 1.0  # signal variance
                self.n2 = 1e-6  # noise variance

            def _rbf_kernel(self, X1, X2):
                """RBF kernel with ARD lengthscales."""
                diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2 / (self.l ** 2), axis=2)
                return np.exp(-0.5 * sq_dist)

            def fit(self, X, y):
                self.X = X
                self.y = y
                n = len(y)
                K = self._rbf_kernel(X, X) + self.n2 * np.eye(n)
                self.alpha = np.linalg.solve(K, y)

            def predict(self, X_new):
                K_star = self._rbf_kernel(X_new, self.X)
                mean = K_star @ self.alpha
                K_ss = self._rbf_kernel(X_new, X_new)
                var = self.s2 - np.sum(K_star * (K_star * self.alpha[:, np.newaxis]), axis=1)
                var = np.maximum(var, 1e-8)
                return mean, np.sqrt(var)

        # ---------- Rastrigin-aware local search ----------
        def local_search(x0, f0, max_calls=60, fine=False):
            nonlocal call_count, best_f, best_x
            x = np.asarray(x0, dtype=float).copy()
            f = float(f0)
            calls_used = 0

            if fine:
                step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            else:
                step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05]

            for step in step_sizes:
                if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                    break
                improved = True
                while improved and calls_used < max_calls and call_count < budget - 5 and time_ok():
                    improved = False
                    for i in range(dim):
                        if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                            break
                        base = x[i]
                        for delta in [step, -step]:
                            if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                                break
                            c = base + delta
                            if not (-4.95 <= c <= 4.95):
                                continue
                            x[i] = c
                            cf = do_call(x)
                            calls_used += 1
                            if cf < f:
                                f = cf
                                improved = True
                            else:
                                x[i] = base
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        # ---------- Phase 0: Initial exploration ----------
        init_pts = [np.zeros(dim)]
        for axis in range(dim):
            for s in (-2.0, -1.0, 1.0, 2.0):
                p = np.zeros(dim)
                p[axis] = s
                init_pts.append(p)
        while len(init_pts) < 20:
            init_pts.append(rng.uniform(-4.0, 4.0, dim))

        for p in init_pts:
            if call_count >= budget or not time_ok():
                break
            p = np.clip(p, -4.95, 4.95)
            f = do_call(p)

        # ---------- Phase 1: Surrogate-guided exploration + local refinement ----------
        reserve = 40
        max_rounds = 5
        for round_idx in range(max_rounds):
            if call_count >= budget - reserve or not time_ok():
                break

            # Refine current best with local search
            local_search(best_x, best_f, max_calls=40, fine=False)

            if call_count >= budget - reserve or not time_ok():
                break

            # Build surrogate from collected data
            if len(X_data) >= 5:
                X_arr = np.array(X_data)
                Y_arr = np.array(Y_data)
                gp = GPSurrogate(dim)
                gp.fit(X_arr, Y_arr)

                # Generate candidate points using acquisition function
                n_candidates = 10
                candidates = []
                for _ in range(n_candidates):
                    # Sample from a grid + random
                    if _ < 5:
                        # Grid-based sampling
                        coords = rng.uniform(-4.5, 4.5, dim)
                    else:
                        # Random sampling
                        coords = rng.uniform(-4.95, 4.95, dim)
                    candidates.append(coords)

                # Evaluate acquisition function (Expected Improvement)
                best_cand = None
                best_acq = -float('inf')
                for cand in candidates:
                    mean, std = gp.predict(cand.reshape(1, -1))
                    mean, std = mean[0], std[0]
                    # Expected Improvement
                    z = (best_f - mean) / max(std, 1e-8)
                    from scipy.stats import norm
                    ei = (best_f - mean) * norm.cdf(z) + std * norm.pdf(z)
                    if ei > best_acq:
                        best_acq = ei
                        best_cand = cand

                # Evaluate the best candidate
                if best_cand is not None:
                    f_cand = do_call(best_cand)

                    # Quick refine from new seed
                    if call_count < budget - reserve and time_ok():
                        local_search(best_cand, f_cand, max_calls=25, fine=False)
            else:
                # Not enough data for surrogate, do random exploration
                n_rand = 5
                for _ in range(n_rand):
                    if call_count >= budget - reserve or not time_ok():
                        break
                    p = rng.uniform(-4.5, 4.5, dim)
                    f = do_call(p)

        # ---------- Phase 2: Multi-start Powell refinement ----------
        if call_count < budget - 1 and time_ok():
            # Sort by function value
            if len(X_data) > 0:
                X_arr = np.array(X_data)
                Y_arr = np.array(Y_data)
                sort_idx = np.argsort(Y_arr)

                # Take top 5 points
                top_k = min(5, len(sort_idx))
                top_points = X_arr[sort_idx[:top_k]]

                for x0 in top_points:
                    if call_count >= budget - 1 or not time_ok():
                        break
                    try:
                        res = minimize(
                            do_call,
                            x0,
                            method='Powell',
                            options={'maxiter': 15, 'xtol': 1e-5, 'ftol': 1e-5}
                        )
                    except Exception:
                        pass

        # ---------- Phase 3: Final fine-tune ----------
        if call_count < budget - 1 and time_ok():
            local_search(best_x, best_f, max_calls=100, fine=True)

        return best_x, best_f
