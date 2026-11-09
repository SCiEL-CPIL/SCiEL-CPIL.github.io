"""Multi-objective optimizer for RE37 using Multi-Objective Nelder-Mead (MONM) simplex."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: MONM simplex + scalarized local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        used = 0
        all_X = []
        all_F = []

        def eval_point(x):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        def time_ok(margin=5.0):
            return (time.time() + margin) < deadline

        def is_non_dominated(f_new, F_archive):
            """Check if f_new is non-dominated by any point in F_archive."""
            for f_old in F_archive:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return False
            return True

        def add_to_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points."""
            if is_non_dominated(f, all_F):
                all_X.append(x.copy())
                all_F.append(f.copy())
                # Remove points dominated by new point
                keep_X = []
                keep_F = []
                for xi, fi in zip(all_X, all_F):
                    if np.all(f <= fi) and np.any(f < fi):
                        continue
                    keep_X.append(xi)
                    keep_F.append(fi)
                all_X.clear()
                all_F.clear()
                all_X.extend(keep_X)
                all_F.extend(keep_F)

        def dominates(a, b):
            """True if a dominates b (a <= b in all objectives, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        # Phase 1: Initial diverse sampling (LHS-like)
        n_init = min(30, self.budget // 5)
        X_init = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            for i in range(n_init):
                X_init[i, d] = lo[d] + span[d] * (i + np.random.random()) / n_init

        for x in X_init:
            if not time_ok(10.0):
                break
            f = eval_point(np.clip(x, lo, hi))
            add_to_archive(np.clip(x, lo, hi), f)

        # Phase 2: Multi-Objective Nelder-Mead (MONM)
        # Build initial simplex around a good starting point
        # Use a regular simplex in 4D

        def build_simplex(center, scale=0.1):
            """Build a regular simplex around center."""
            n = self.dim
            vertices = [center.copy()]
            # Standard regular simplex vertices
            e = np.zeros(n)
            for i in range(n):
                e = np.zeros(n)
                e[i] = 1.0
                # Orthogonalize against previous vertices
                for j in range(len(vertices) - 1):
                    vj = vertices[j] - center
                    proj = np.dot(e, vj) / (np.dot(vj, vj) + 1e-15)
                    e -= proj * vj
                e_norm = np.linalg.norm(e)
                if e_norm < 1e-10:
                    e = np.random.random(n) - 0.5
                    e_norm = np.linalg.norm(e)
                e = e / e_norm
                vertices.append(center + scale * span * e)
            return np.array(vertices)

        # Run MONM from multiple starting points
        n_starts = min(8, max(3, (self.budget - used) // 20))
        for s in range(n_starts):
            if not time_ok(10.0):
                break
            if used >= self.budget - 20:
                break

            # Choose starting point
            if all_X and np.random.random() < 0.6:
                center = all_X[np.random.randint(len(all_X))].copy()
            else:
                center = lo + span * np.random.random(self.dim)
            center = np.clip(center, lo, hi)

            # Build initial simplex
            simplex_X = build_simplex(center, scale=0.15)
            simplex_F = []
            for vx in simplex_X:
                vx = np.clip(vx, lo, hi)
                f = eval_point(vx)
                simplex_F.append(f)
                add_to_archive(vx, f)
                if not time_ok(5.0):
                    break

            if len(simplex_F) < self.dim + 1:
                continue

            simplex_F = np.array(simplex_F)
            simplex_X = np.array(simplex_X)

            # MONM iterations
            n_iter = min(25, (self.budget - used) // (self.dim + 1))
            for it in range(n_iter):
                if not time_ok(3.0):
                    break
                if used >= self.budget - 5:
                    break

                # Sort vertices by scalarized quality (use average objective)
                # For multi-objective, we need a different approach
                # Use a weighted sum with random weights to determine "best" and "worst"
                w = np.random.dirichlet(np.ones(self.n_obj))
                scores = simplex_F @ w
                order = np.argsort(scores)
                simplex_X = simplex_X[order]
                simplex_F = simplex_F[order]

                best_X = simplex_X[0]
                worst_X = simplex_X[-1]
                worst_F = simplex_F[-1]

                # Centroid of all but worst
                centroid = np.mean(simplex_X[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - worst_X)
                x_ref = np.clip(x_ref, lo, hi)
                f_ref = eval_point(x_ref)

                if not dominates(f_ref, worst_F) and not is_non_dominated(f_ref, all_F):
                    # Reflection is not better, try contraction
                    gamma = 0.5
                    x_con = centroid + gamma * (worst_X - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = eval_point(x_con)

                    if not dominates(f_con, worst_F):
                        # Shrink toward best
                        for i in range(1, self.dim + 1):
                            simplex_X[i] = best_X + 0.5 * (simplex_X[i] - best_X)
                            simplex_X[i] = np.clip(simplex_X[i], lo, hi)
                            simplex_F[i] = eval_point(simplex_X[i])
                            add_to_archive(simplex_X[i], simplex_F[i])
                            if not time_ok(2.0):
                                break
                    else:
                        simplex_X[-1] = x_con
                        simplex_F[-1] = f_con
                        add_to_archive(x_con, f_con)
                else:
                    # Reflection is good, try expansion
                    if not is_non_dominated(f_ref, all_F) or dominates(f_ref, worst_F):
                        beta = 2.0
                        x_exp = centroid + beta * (x_ref - centroid)
                        x_exp = np.clip(x_exp, lo, hi)
                        f_exp = eval_point(x_exp)
                        if dominates(f_exp, f_ref):
                            simplex_X[-1] = x_exp
                            simplex_F[-1] = f_exp
                            add_to_archive(x_exp, f_exp)
                        else:
                            simplex_X[-1] = x_ref
                            simplex_F[-1] = f_ref
                            add_to_archive(x_ref, f_ref)
                    else:
                        simplex_X[-1] = x_ref
                        simplex_F[-1] = f_ref
                        add_to_archive(x_ref, f_ref)

        # Phase 3: Scalarized local polish from archive points
        n_polish = min(30, self.budget - used - 10)
        if n_polish > 0 and all_F:
            for _ in range(n_polish):
                if not time_ok(5.0):
                    break
                if used >= self.budget - 5:
                    break

                idx = np.random.randint(len(all_X))
                x_seed = all_X[idx].copy()
                w = np.random.dirichlet(np.ones(self.n_obj))

                # Small Nelder-Mead on scalarized objective
                def scalarized(x):
                    f = eval_point(x)
                    return np.dot(w, f)

                # Manual Nelder-Mead for speed
                x0 = x_seed + 0.05 * span * np.random.standard_normal(self.dim)
                x0 = np.clip(x0, lo, hi)

                # Build small simplex
                nm_simplex = [x0.copy()]
                for d in range(self.dim):
                    xp = x0.copy()
                    step = 0.05 * span[d]
                    xp[d] += step
                    xp = np.clip(xp, lo, hi)
                    nm_simplex.append(xp)

                nm_F = [eval_point(vx) for vx in nm_simplex]
                nm_F_scalar = [np.dot(w, f) for f in nm_F]

                for nm_it in range(15):
                    if not time_ok(2.0):
                        break
                    if used >= self.budget - 3:
                        break

                    order = np.argsort(nm_F_scalar)
                    nm_simplex = [nm_simplex[i] for i in order]
                    nm_F_scalar = [nm_F_scalar[i] for i in order]

                    worst = nm_simplex[-1]
                    centroid = np.mean(nm_simplex[:-1], axis=0)

                    x_ref = centroid + 1.0 * (centroid - worst)
                    x_ref = np.clip(x_ref, lo, hi)
                    f_ref = eval_point(x_ref)
                    f_ref_s = np.dot(w, f_ref)

                    if f_ref_s < nm_F_scalar[-1]:
                        x_exp = centroid + 2.0 * (x_ref - centroid)
                        x_exp = np.clip(x_exp, lo, hi)
                        f_exp = eval_point(x_exp)
                        f_exp_s = np.dot(w, f_exp)
                        if f_exp_s < f_ref_s:
                            nm_simplex[-1] = x_exp
                            nm_F_scalar[-1] = f_exp_s
                        else:
                            nm_simplex[-1] = x_ref
                            nm_F_scalar[-1] = f_ref_s
                    else:
                        x_con = centroid + 0.5 * (worst - centroid)
                        x_con = np.clip(x_con, lo, hi)
                        f_con = eval_point(x_con)
                        f_con_s = np.dot(w, f_con)
                        if f_con_s < nm_F_scalar[-1]:
                            nm_simplex[-1] = x_con
                            nm_F_scalar[-1] = f_con_s
                        else:
                            for i in range(1, self.dim + 1):
                                nm_simplex[i] = nm_simplex[0] + 0.5 * (nm_simplex[i] - nm_simplex[0])
                                nm_simplex[i] = np.clip(nm_simplex[i], lo, hi)
                                nm_F_scalar[i] = np.dot(w, eval_point(nm_simplex[i]))

                # Add best point from NM to archive
                best_idx = np.argmin(nm_F_scalar)
                add_to_archive(nm_simplex[best_idx], eval_point(nm_simplex[best_idx]))

        # Phase 4: Fill remaining budget with random sampling + quick polish
        while used < self.budget - 5 and time_ok(3.0):
            x_rand = lo + span * np.random.random(self.dim)
            f_rand = eval_point(x_rand)
            add_to_archive(x_rand, f_rand)

            if used < self.budget - 8 and time_ok(2.0):
                w = np.random.dirichlet(np.ones(self.n_obj))
                x_pert = x_rand + 0.03 * span * np.random.standard_normal(self.dim)
                x_pert = np.clip(x_pert, lo, hi)
                f_pert = eval_point(x_pert)
                add_to_archive(x_pert, f_pert)

        # Build final result
        if all_X:
            X_final = np.array(all_X)
            F_final = np.array(all_F)
        else:
            X_final = np.array([])
            F_final = np.array([])

        return X_final

# EVOLVE-BLOCK-END
