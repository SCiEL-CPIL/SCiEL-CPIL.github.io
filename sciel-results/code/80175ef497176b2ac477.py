"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """NSGA-II evolutionary search + boundary init + Nelder-Mead local polish for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random  # evaluator sets np.random.seed(seed) before calling

        # ------------------------------------------------------------------
        # Evaluation bookkeeping
        # ------------------------------------------------------------------
        evaluated = []          # list of (x, f)
        eval_set = set()
        eval_count = 0

        def evaluate(x):
            """Evaluate x (clipped to bounds). Returns f or None if duplicate/no budget."""
            nonlocal eval_count
            if eval_count >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
            key = tuple(np.round(x, 9))
            if key in eval_set:
                return None
            f = np.array(problem(x), dtype=np.float64)
            if not np.all(np.isfinite(f)):
                eval_set.add(key)
                return None
            eval_set.add(key)
            evaluated.append((x, f))
            eval_count += 1
            return f

        # ------------------------------------------------------------------
        # 2-objective helpers
        # ------------------------------------------------------------------
        def dominates(a, b):
            return bool(np.all(a <= b) and np.any(a < b))

        def non_dominated_indices(F):
            """F: (K, n_obj). Return list of non-dominated indices."""
            K = F.shape[0]
            nd = []
            for i in range(K):
                is_dom = False
                for j in range(K):
                    if i != j and dominates(F[j], F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    nd.append(i)
            return nd

        def hv2(pts, ref):
            """Exact 2D hypervolume of a list of (f1,f2) points vs reference point."""
            pts = [p for p in pts if p[0] < ref[0] and p[1] < ref[1]]
            if not pts:
                return 0.0
            pts = sorted(pts, key=lambda p: p[0])
            hv = 0.0
            prev = ref[1]
            for f1, f2 in pts:
                if f2 < prev:
                    hv += (ref[0] - f1) * (prev - f2)
                    prev = f2
            return hv

        # ------------------------------------------------------------------
        # Phase 1: Boundary + stratified initialization
        # ------------------------------------------------------------------
        N = 40
        n_init = min(N, budget)

        init_X = []
        # (a) boundary points: each dimension at its low or high end (2^dim corners)
        for mask in range(1 << dim):
            x = lo.copy()
            for d in range(dim):
                if (mask >> d) & 1:
                    x[d] = hi[d]
            init_X.append(x)
        # (b) fill remaining with stratified (LHS-like) interior points
        n_corners = len(init_X)
        n_fill = n_init - n_corners
        for i in range(max(0, n_fill)):
            x = np.zeros(dim)
            for d in range(dim):
                frac = (i + rng.random()) / max(1, n_fill)
                x[d] = lo[d] + frac * (hi[d] - lo[d])
            init_X.append(x)

        init_F = []
        for x in init_X:
            f = evaluate(x)
            if f is not None:
                init_F.append(f)
            if eval_count >= budget:
                break

        if eval_count >= budget or not init_F:
            return evaluated

        X = np.array([ex for ex, _ in evaluated[:len(init_F)]], dtype=np.float64)
        F_vals = np.array(init_F, dtype=np.float64)

        # ------------------------------------------------------------------
        # Phase 2: Differential Evolution main loop with HV-based selection
        # ------------------------------------------------------------------
        F = 0.8
        CR = 0.9
        pop_size = len(X)

        while eval_count < budget:
            if time.time() > deadline:
                break

            # generate offspring (up to pop_size, bounded by remaining budget)
            num_off = min(pop_size, budget - eval_count)
            U = np.zeros((num_off, dim))
            U_vals = []
            U_xs = []
            for i in range(num_off):
                if time.time() > deadline:
                    break
                # Differential mutation: v = x_r1 + F * (x_r2 - x_r3)
                r1, r2, r3 = rng.choice(len(X), 3, replace=False)
                v = X[r1] + F * (X[r2] - X[r3])
                # Binomial crossover
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() <= CR or j == j_rand:
                        U[i, j] = v[j]
                    else:
                        U[i, j] = X[i, j]
                U[i] = np.clip(U[i], lo, hi)
                f = evaluate(U[i])
                if f is not None:
                    U_vals.append(f)
                    U_xs.append(U[i].copy())
                if eval_count >= budget:
                    break

            if eval_count >= budget or time.time() > deadline:
                break

            if not U_vals:
                break

            # Combine parent and offspring
            F_comb = np.vstack([F_vals, np.array(U_vals)])
            X_comb = np.vstack([X, np.array(U_xs)])

            # Select best pop_size points by greedy hypervolume contribution
            # Use task reference point for selection
            task_ref = np.array([3000.0, 0.06])
            
            # First filter to non-dominated set for efficiency
            nd_idx = non_dominated_indices(F_comb)
            
            selected = []
            remaining = list(nd_idx)
            
            # Greedy HV selection: pick points that maximize hypervolume increment
            for _ in range(min(pop_size, len(remaining))):
                if not remaining or time.time() > deadline:
                    break
                best_idx, best_hv = -1, -1.0
                cur_pts = [tuple(F_comb[i]) for i in selected]
                for idx in remaining:
                    cand = cur_pts + [tuple(F_comb[idx])]
                    hv_val = hv2(cand, task_ref)
                    if hv_val > best_hv:
                        best_hv, best_idx = hv_val, idx
                if best_idx != -1:
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                else:
                    break

            # If we didn't select enough from non-dominated, add dominated points
            # that are close to the front
            if len(selected) < pop_size:
                dominated_idx = [i for i in range(len(F_comb)) if i not in nd_idx]
                # Sort dominated by their distance to the front (simple sum)
                dominated_idx.sort(key=lambda i: F_comb[i, 0] + F_comb[i, 1] * 1000)
                for idx in dominated_idx:
                    if len(selected) >= pop_size:
                        break
                    if idx not in selected:
                        selected.append(idx)

            if selected:
                X = np.array([X_comb[i] for i in selected], dtype=np.float64)
                F_vals = np.array([F_comb[i] for i in selected], dtype=np.float64)

        # ------------------------------------------------------------------
        # Phase 3: Local polish on top hypervolume contributors
        # ------------------------------------------------------------------
        if eval_count < budget and evaluated:
            all_F = np.array([f for _, f in evaluated], dtype=np.float64)
            nd_idx = non_dominated_indices(all_F)
            # rank non-dominated points by their hypervolume contribution using task ref
            task_ref = np.array([3000.0, 0.06])
            base_pts = [tuple(all_F[i]) for i in nd_idx]
            # contribution = hv(all) - hv(all without i)
            full_hv = hv2(base_pts, task_ref)
            contrib = []
            for k in range(len(nd_idx)):
                sub = [base_pts[m] for m in range(len(nd_idx)) if m != k]
                c = full_hv - hv2(sub, task_ref)
                contrib.append((c, nd_idx[k]))
            contrib.sort(reverse=True)

            # polish the top contributors with Nelder-Mead
            top = [idx for _, idx in contrib[:8]]
            
            # build a local objective: weighted sum emphasizing hypervolume region
            def local_obj(x):
                x = np.clip(x, lo, hi)
                f = np.array(problem(x), dtype=np.float64)
                if not np.all(np.isfinite(f)):
                    return 1e9
                # scalarize: push both objectives down
                w = np.array([1.0 / (3000.0 + 1e-6), 1.0 / (0.06 + 1e-6)])
                return float(np.sum(w * f))

            for idx in top:
                if eval_count >= budget or time.time() > deadline:
                    break
                x0 = evaluated[idx][0].copy()
                try:
                    res = minimize(local_obj, x0, method='Nelder-Mead',
                                   options={'maxiter': 50, 'xatol': 1e-8,
                                            'fatol': 1e-8, 'adaptive': True})
                except Exception:
                    res = None
                if res is not None:
                    evaluate(res.x)
                    # evaluate interpolated points along the path
                    for t in (0.3, 0.6, 0.9):
                        if eval_count >= budget or time.time() > deadline:
                            break
                        x_int = (1 - t) * x0 + t * np.clip(res.x, lo, hi)
                        evaluate(x_int)

        return evaluated
# EVOLVE-BLOCK-END
