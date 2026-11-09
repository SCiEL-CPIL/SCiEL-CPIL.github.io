"""k-NN surrogate with epsilon-greedy exploration/exploitation for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """k-NN surrogate-guided optimizer for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(7777)

        best_x = None
        best_f = np.inf
        X_all = []
        Y_all = []

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            X_all.append(x.copy())
            Y_all.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return calls_made < budget and (time.time() - start_time) < (deadline - start_time)

        # ---- Phase 1: Initial exploration with structured + quasi-random points ----
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = int(index)
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_sequence(n, dim):
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
            pts = []
            for i in range(1, n + 1):
                pts.append(np.array([halton(i, b) for b in bases]))
            return np.array(pts) * 10.0 - 5.0

        # Structured points
        structured = [np.zeros(dim)]
        for d in range(dim):
            for v in [-3.0, 0.0, 3.0]:
                p = np.zeros(dim)
                p[d] = v
                structured.append(p)
        # Some diagonal / corner-ish points
        for mask in [(1, 0, 1, 0, 1), (0, 1, 0, 1, 0), (1, 1, 0, 0, 1), (0, 0, 1, 1, 0),
                     (1, 0, 0, 1, 0), (0, 1, 1, 0, 1), (1, 1, 1, 0, 0), (0, 0, 1, 1, 1)]:
            p = np.array([(-2.5 if m else 2.5) for m in mask])
            structured.append(p)

        n_halton = 40
        init_pts = halton_sequence(n_halton, dim)

        all_init = np.vstack([np.array(structured), init_pts])
        all_init = np.clip(all_init, -5.0, 5.0)

        # Deduplicate
        unique_init = [all_init[0]]
        for p in all_init[1:]:
            if np.all([np.linalg.norm(p - u) > 0.15 for u in unique_init]):
                unique_init.append(p)
                if len(unique_init) >= 55:
                    break

        for p in unique_init:
            if not time_ok():
                break
            evaluate(p)

        # ---- Phase 2: k-NN surrogate guided search ----
        K = 5

        def knn_predict(x, X, Y, k=K):
            """Predict using weighted k-NN (distance-weighted mean of k nearest)."""
            if len(X) == 0:
                return 0.0
            X_arr = np.array(X) if not isinstance(X, np.ndarray) else X
            Y_arr = np.array(Y) if not isinstance(Y, np.ndarray) else Y
            n = len(X_arr)
            if n <= k:
                # Weighted by distance
                d2 = np.sum((X_arr - x) ** 2, axis=1)
                w = 1.0 / (d2 + 1e-6)
                return float(np.dot(w, Y_arr) / np.sum(w))
            # Find k nearest
            d2 = np.sum((X_arr - x) ** 2, axis=1)
            idx = np.argpartition(d2, k)[:k]
            d2_k = d2[idx]
            Y_k = Y_arr[idx]
            w = 1.0 / (d2_k + 1e-6)
            return float(np.dot(w, Y_k) / np.sum(w))

        def knn_predict_batch(Xq, X, Y, k=K):
            """Predict for multiple query points."""
            X_arr = np.array(X) if not isinstance(X, np.ndarray) else X
            Y_arr = np.array(Y) if not isinstance(Y, np.ndarray) else Y
            n = len(X_arr)
            results = np.zeros(len(Xq))
            for i, xq in enumerate(Xq):
                results[i] = knn_predict(xq, X_arr, Y_arr, k)
            return results

        def max_min_distance(x, X, k=K):
            """Distance to the k-th nearest neighbor (for exploration)."""
            if len(X) == 0:
                return 10.0
            X_arr = np.array(X) if not isinstance(X, np.ndarray) else X
            d2 = np.sum((X_arr - x) ** 2, axis=1)
            if len(X_arr) <= k:
                return float(np.sqrt(np.max(d2)))
            idx = np.argpartition(d2, k)[:k]
            return float(np.sqrt(np.max(d2[idx])))

        # Main loop
        x_current = best_x.copy() if best_x is not None else np.zeros(dim)
        f_current = best_f
        es_sigma = 0.8
        n_success = 0
        n_trials = 0
        stall_count = 0
        prev_best = best_f
        epsilon = 0.5  # exploration rate

        # Track convergence
        best_f_history = [best_f]

        while time_ok():
            n_trials += 1
            # Decay epsilon over time (more exploitation later)
            progress = min(1.0, n_trials / max(1, (budget - calls_made) * 0.6))
            epsilon = 0.6 * (1.0 - progress) + 0.1 * progress

            # Generate candidate pool
            n_cands = 8
            candidates = []

            # ES-style candidates around current best
            for _ in range(3):
                c = x_current + es_sigma * rng.standard_normal(dim)
                c = np.clip(c, -5.0, 5.0)
                candidates.append(c)

            # Candidate near global best
            if best_x is not None:
                c = best_x + es_sigma * 0.5 * rng.standard_normal(dim)
                c = np.clip(c, -5.0, 5.0)
                candidates.append(c)

            # Random exploration candidates
            for _ in range(2):
                c = rng.uniform(-5.0, 5.0, size=dim)
                candidates.append(c)

            # Surrogate-guided candidates: sample and pick best predicted
            if len(X_all) >= K + 2:
                for _ in range(3):
                    c = best_x + es_sigma * 1.5 * rng.standard_normal(dim) if best_x is not None else rng.uniform(-5, 5, dim)
                    c = np.clip(c, -5.0, 5.0)
                    candidates.append(c)

            # Select candidate using epsilon-greedy
            if len(candidates) > 0:
                Xq = np.array(candidates)
                # Compute surrogate predictions
                preds = np.array([knn_predict(c, X_all, Y_all, K) for c in Xq])
                # Compute exploration scores (max-min distance)
                expl_scores = np.array([max_min_distance(c, X_all, K) for c in Xq])

                # Normalize
                if np.max(preds) - np.min(preds) > 1e-10:
                    preds_norm = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))
                else:
                    preds_norm = np.zeros_like(preds)
                if np.max(expl_scores) - np.min(expl_scores) > 1e-10:
                    expl_norm = (expl_scores - np.min(expl_scores)) / (np.max(expl_scores) - np.min(expl_scores))
                else:
                    expl_norm = np.zeros_like(expl_scores)

                # Epsilon-greedy: with prob epsilon pick max exploration, else min prediction
                if rng.random() < epsilon:
                    idx = int(np.argmax(expl_norm))
                else:
                    idx = int(np.argmin(preds_norm))

                x_candidate = Xq[idx]
            else:
                x_candidate = np.clip(x_current + es_sigma * rng.standard_normal(dim), -5.0, 5.0)

            f_cand = evaluate(x_candidate)

            # Update current point if improved
            if f_cand < f_current - 1e-10:
                x_current = x_candidate.copy()
                f_current = f_cand
                n_success += 1
                es_sigma = min(es_sigma * 1.1, 2.0)
                stall_count = 0
            else:
                es_sigma = max(es_sigma * 0.9, 0.05)
                stall_count += 1

            # 1/5 success rule
            if n_trials % 5 == 0:
                rate = n_success / 5.0
                if rate > 0.2:
                    es_sigma *= 1.05
                elif rate < 0.1:
                    es_sigma *= 0.95
                es_sigma = max(0.03, min(es_sigma, 2.5))
                n_success = 0

            # Restart if stalled
            if stall_count > 30:
                if len(X_all) >= 10:
                    good_idx = [i for i, y in enumerate(Y_all) if y < best_f + 0.5]
                    if good_idx:
                        gi = rng.choice(good_idx)
                        x_current = X_all[gi].copy()
                        f_current = Y_all[gi]
                        es_sigma = max(es_sigma, 0.4)
                    else:
                        x_current = rng.uniform(-4, 4, size=dim)
                        f_current = np.inf
                        es_sigma = 0.8
                stall_count = 0

            # Convergence check
            if es_sigma < 0.03:
                break

        # ---- Phase 3: Nelder-Mead final polish ----
        if time_ok() and best_x is not None and calls_made < budget:
            nm_x = best_x.copy()
            nm_f = best_f
            nm_sigma = 0.05

            simplex = [nm_x.copy()]
            for d in range(dim):
                p = nm_x.copy()
                p[d] += nm_sigma
                simplex.append(p)
            simplex = np.array(simplex)
            fvals = [nm_f]
            for i in range(1, dim + 1):
                if not time_ok() or calls_made >= budget:
                    break
                fvals.append(evaluate(simplex[i]))

            for it in range(30):
                if not time_ok() or calls_made >= budget:
                    break
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                if calls_made >= budget:
                    break
                fr = evaluate(xr)

                if fr < fvals[0]:
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    if calls_made >= budget:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        break
                    fe = evaluate(xe)
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
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    if calls_made >= budget:
                        break
                    fc = evaluate(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, dim + 1):
                            if calls_made >= budget:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = evaluate(simplex[i])

                if max(fvals) - min(fvals) < 1e-8:
                    break

        # ---- Phase 4: Fine-tuning ----
        if time_ok() and best_x is not None:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.02
            for _ in range(20):
                if not time_ok() or calls_made >= budget:
                    break
                y = x_polish + sigma_p * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    sigma_p = max(0.003, sigma_p * 0.9)
                else:
                    sigma_p = max(0.002, sigma_p * 0.85)

        return best_x, best_f
# EVOLVE-BLOCK-END
