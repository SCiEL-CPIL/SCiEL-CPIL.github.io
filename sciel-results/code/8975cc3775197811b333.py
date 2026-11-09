"""Surrogate-assisted multi-start optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Surrogate-assisted multi-start optimizer for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

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
        for mask in [(1, 0, 1, 0, 1), (0, 1, 0, 1, 0), (1, 1, 0, 0, 1), (0, 0, 1, 1, 0),
                     (1, 0, 0, 1, 0), (0, 1, 1, 0, 1), (1, 1, 1, 0, 0), (0, 0, 1, 1, 1)]:
            p = np.array([(-2.5 if m else 2.5) for m in mask])
            structured.append(p)

        n_halton = 30
        init_pts = halton_sequence(n_halton, dim)

        all_init = np.vstack([np.array(structured), init_pts])
        all_init = np.clip(all_init, -5.0, 5.0)

        unique_init = [all_init[0]]
        for p in all_init[1:]:
            if np.all([np.linalg.norm(p - u) > 0.15 for u in unique_init]):
                unique_init.append(p)
                if len(unique_init) >= 50:
                    break

        for p in unique_init:
            if not time_ok():
                break
            evaluate(p)

        # ---- k-NN Surrogate with adaptive width ----
        K = 10

        def knn_predict(x, k=K):
            if len(X_all) == 0:
                return 0.0
            X_arr = np.array(X_all)
            Y_arr = np.array(Y_all)
            n = len(X_arr)
            d2 = np.sum((X_arr - x) ** 2, axis=1)
            if n <= k:
                w = 1.0 / (d2 + 1e-6)
                return float(np.dot(w, Y_arr) / np.sum(w))
            idx = np.argpartition(d2, k)[:k]
            d2_k = d2[idx]
            Y_k = Y_arr[idx]
            w = 1.0 / (d2_k + 1e-6)
            return float(np.dot(w, Y_k) / np.sum(w))

        def acquisition_score(x, k=K):
            """Combined exploitation + exploration score."""
            if len(X_all) == 0:
                return 10.0
            X_arr = np.array(X_all)
            d2 = np.sum((X_arr - x) ** 2, axis=1)
            
            # Exploitation: predicted value (lower is better)
            if len(X_arr) <= k:
                w = 1.0 / (d2 + 1e-6)
                pred = float(np.dot(w, Y_arr) / np.sum(w))
            else:
                idx = np.argpartition(d2, k)[:k]
                d2_k = d2[idx]
                Y_k = Y_arr[idx]
                w = 1.0 / (d2_k + 1e-6)
                pred = float(np.dot(w, Y_k) / np.sum(w))
            
            # Exploration: distance to nearest neighbor (higher is better)
            nearest_d = float(np.sqrt(np.min(d2)))
            
            # Adaptive exploration weight based on number of samples
            n_samples = len(X_all)
            if n_samples < 50:
                expl_weight = 2.0
            elif n_samples < 150:
                expl_weight = 1.0
            elif n_samples < 300:
                expl_weight = 0.5
            else:
                expl_weight = 0.3
            
            # Normalize and combine
            score = pred - expl_weight * nearest_d * 0.5
            return score

        # ---- Phase 2: Main optimization loop with surrogate-guided search ----
        x_current = best_x.copy() if best_x is not None else np.zeros(dim)
        f_current = best_f
        es_sigma = 0.8
        stall_count = 0
        no_improve_count = 0
        prev_best = best_f

        # Multi-start points for restarts
        restart_pool = []
        if best_x is not None:
            restart_pool.append(best_x.copy())

        while time_ok():
            # Generate candidate pool
            n_cands = 12
            candidates = []

            # ES-style candidates around current best
            for _ in range(4):
                c = x_current + es_sigma * rng.standard_normal(dim)
                c = np.clip(c, -5.0, 5.0)
                candidates.append(c)

            # Candidate near global best
            if best_x is not None:
                c = best_x + es_sigma * 0.4 * rng.standard_normal(dim)
                c = np.clip(c, -5.0, 5.0)
                candidates.append(c)

            # Random exploration
            for _ in range(3):
                c = rng.uniform(-5.0, 5.0, size=dim)
                candidates.append(c)

            # Surrogate-guided: sample broadly and pick by acquisition score
            if len(X_all) >= K + 5:
                for _ in range(5):
                    # Sample around best with varying radii
                    radius = es_sigma * rng.choice([0.5, 1.0, 1.5, 2.0])
                    base = best_x if best_x is not None else np.zeros(dim)
                    c = base + radius * rng.standard_normal(dim)
                    c = np.clip(c, -5.0, 5.0)
                    candidates.append(c)

            if len(candidates) > 0:
                Xq = np.array(candidates)

                # Compute acquisition scores
                acq_scores = np.array([acquisition_score(c) for c in Xq])

                # Select candidate with best acquisition score
                idx = int(np.argmin(acq_scores))
                x_candidate = Xq[idx]
            else:
                x_candidate = np.clip(x_current + es_sigma * rng.standard_normal(dim), -5.0, 5.0)

            f_cand = evaluate(x_candidate)

            # Update current point and sigma
            if f_cand < f_current - 1e-10:
                x_current = x_candidate.copy()
                f_current = f_cand
                es_sigma = min(es_sigma * 1.1, 2.0)
                stall_count = 0
            else:
                es_sigma = max(es_sigma * 0.93, 0.05)
                stall_count += 1

            # Track improvement for restart logic
            if best_f < prev_best - 1e-8:
                no_improve_count = 0
                prev_best = best_f
                if best_x is not None and len(restart_pool) < 10:
                    restart_pool.append(best_x.copy())
            else:
                no_improve_count += 1

            # Restart if stalled
            if stall_count > 30 or no_improve_count > 60:
                if restart_pool:
                    # Pick a random point from restart pool
                    gi = rng.randint(len(restart_pool))
                    x_current = restart_pool[gi].copy()
                    f_current = evaluate(x_current)
                    es_sigma = max(es_sigma, 0.5)
                else:
                    x_current = rng.uniform(-4, 4, size=dim)
                    f_current = evaluate(x_current)
                    es_sigma = 0.8
                stall_count = 0
                no_improve_count = 0

            if es_sigma < 0.03:
                break

        # ---- Phase 3: Nelder-Mead final polish ----
        if time_ok() and best_x is not None and calls_made < budget - 15:
            nm_x = best_x.copy()
            nm_f = best_f
            nm_sigma = 0.1

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

            for it in range(50):
                if not time_ok() or calls_made >= budget:
                    break
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

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

                if max(fvals) - min(fvals) < 1e-9:
                    break

            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

        # ---- Phase 4: Fine-tuning with coordinate descent ----
        if time_ok() and best_x is not None and calls_made < budget:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.04

            for _ in range(30):
                if not time_ok() or calls_made >= budget:
                    break
                improved = False
                for d in range(dim):
                    if not time_ok() or calls_made >= budget:
                        break
                    for step in [sigma_p, -sigma_p]:
                        y = x_polish.copy()
                        y[d] += step
                        y = np.clip(y, -5.0, 5.0)
                        fy = evaluate(y)
                        if fy < f_polish - 1e-12:
                            x_polish = y.copy()
                            f_polish = fy
                            improved = True
                            break
                if not improved:
                    y = x_polish + sigma_p * 0.5 * rng.standard_normal(dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = evaluate(y)
                    if fy < f_polish:
                        x_polish = y
                        f_polish = fy

                if improved:
                    sigma_p = max(0.003, sigma_p * 0.85)
                else:
                    sigma_p = max(0.002, sigma_p * 0.8)

            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
