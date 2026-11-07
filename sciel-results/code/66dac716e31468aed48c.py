# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid multi-objective optimizer: LHS init, diverse Tchebycheff decomposition, and Nelder-Mead polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Record all evaluated points
        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count += 1
            return x, f

        # Phase 1: Latin Hypercube initialization
        n_lhs = 20
        lhs_X = np.empty((n_lhs, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_lhs)
            lhs_X[:, d] = (perm + 0.5) / n_lhs
        lhs_X = lo + (hi - lo) * lhs_X

        for x in lhs_X:
            if eval_count >= self.budget or time.time() > deadline:
                break
            evaluate(x)

        # Phase 2: Tchebycheff decomposition with diverse weight vectors
        # This covers different parts of the Pareto front
        if eval_count < self.budget and time.time() < deadline:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            # Find non-dominated points
            mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                for j in range(len(F_arr)):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            nd_idx = np.where(mask)[0]

            if len(nd_idx) > 0:
                # Get objective ranges for normalization
                f_min = np.min(F_arr[nd_idx], axis=0)
                f_max = np.max(F_arr[nd_idx], axis=0)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1.0

                # Create diverse weight vectors for 2 objectives
                # Use a set of weights that cover the front well
                n_weights = min(10, (self.budget - eval_count) // 3)
                if n_weights > 0:
                    # Create weight vectors: from (1,0) to (0,1) with emphasis on extremes
                    weight_vectors = []
                    # Include extreme weights
                    weight_vectors.append(np.array([1.0, 0.0]))
                    weight_vectors.append(np.array([0.0, 1.0]))
                    # Add intermediate weights
                    for k in range(1, n_weights - 1):
                        w1 = k / (n_weights - 1)
                        w2 = 1.0 - w1
                        weight_vectors.append(np.array([w1, w2]))
                    # Trim to fit budget
                    weight_vectors = weight_vectors[:n_weights]

                    # For each weight vector, do a local search from the best starting point
                    for w in weight_vectors:
                        if eval_count >= self.budget or time.time() > deadline:
                            break

                        # Find best starting point for this weight
                        # Score: normalized weighted sum
                        F_norm = (F_arr[nd_idx] - f_min) / f_range
                        scores = np.sum(F_norm * w, axis=1)
                        best_local = np.argmin(scores)
                        x0 = X_arr[nd_idx[best_local]].copy()

                        # Tchebycheff objective: max_i (w_i * |f_i - z_i^min|)
                        # But we use a smooth approximation via log-sum-exp
                        def tchebycheff_obj(x):
                            x = np.clip(x, lo, hi)
                            f = np.array(problem(x))
                            # Normalize objectives
                            f_norm = (f - f_min) / f_range
                            # Weighted objectives
                            weighted = w * f_norm
                            # Soft max (log-sum-exp) for smoothness
                            # Use a moderate temperature for smoothness
                            temp = 0.1
                            max_w = np.max(weighted)
                            log_sum_exp = max_w + temp * np.log(np.sum(np.exp((weighted - max_w) / temp)))
                            return log_sum_exp

                        try:
                            res = minimize(tchebycheff_obj, x0, method='Nelder-Mead',
                                           bounds=list(zip(lo, hi)),
                                           options={'maxiter': 30, 'xatol': 1e-5, 'fatol': 1e-7})
                            if eval_count < self.budget:
                                cx = np.clip(res.x, lo, hi)
                                cf = np.array(problem(cx))
                                all_X.append(cx.copy())
                                all_F.append(cf.copy())
                                eval_count += 1
                        except Exception:
                            pass

        # Phase 3: Final Nelder-Mead polish on top non-dominated points
        if eval_count < self.budget and time.time() < deadline:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            # Find non-dominated
            mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                for j in range(len(F_arr)):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            nd_idx = np.where(mask)[0]

            if len(nd_idx) > 0:
                # Sort by sum of objectives
                order = np.argsort(np.sum(F_arr[nd_idx], axis=1))
                top_n = min(5, len(nd_idx), (self.budget - eval_count) // 2)

                for idx in order[:top_n]:
                    if eval_count >= self.budget or time.time() > deadline:
                        break

                    x0 = X_arr[idx].copy()

                    def objective(x):
                        x = np.clip(x, lo, hi)
                        f = np.array(problem(x))
                        return np.sum(f)

                    try:
                        res = minimize(objective, x0, method='Nelder-Mead',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                        if eval_count < self.budget:
                            cx = np.clip(res.x, lo, hi)
                            cf = np.array(problem(cx))
                            all_X.append(cx.copy())
                            all_F.append(cf.copy())
                            eval_count += 1
                    except Exception:
                        pass

        return all_X
# EVOLVE-BLOCK-END
