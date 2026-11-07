"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using stratified sampling + local Nelder-Mead refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 30s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3000.0, 0.06])

        def eval_point(x):
            """Evaluate a point, clip to bounds, return objective vector."""
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            return x, f

        def hypervolume_loss(f):
            """
            Compute a loss based on how far f is from contributing to hypervolume.
            We want to minimize objectives. A point is useful if it's below ref in all dims.
            Loss = sum of (ref_i - f_i) for each dim where f_i < ref_i, but we want to MINIMIZE this.
            Actually, for local search we want to push f towards the origin (0,0) in objective space.
            Simple loss: sum of max(0, f_i - 0) for each objective? No, that doesn't account for trade-offs.
            
            Better approach: minimize the L1 distance to the ideal point (0,0) in objective space,
            but only for points that are below the reference point.
            Or even simpler: just minimize each objective independently? No, that's not multi-objective.
            
            Let's use a scalarization: minimize f[0] + f[1] (weighted sum).
            But this might miss parts of the Pareto front.
            
            Alternative: for local search, we can use the fact that we want to find points on the Pareto front.
            A good scalarization is to minimize the distance to the reference point in objective space,
            but only for points that are dominated by the reference point.
            
            Let's use: loss = sum((ref_i - f_i)^2 for i where f_i < ref_i)
            This encourages f to be as small as possible in each dimension, up to the reference point.
            """
            if np.any(f >= ref):
                return np.sum(f)  # Penalize points above reference
            return np.sum((ref - f) ** 2)

        def nelder_mead(x0, max_evals=15):
            """
            Local Nelder-Mead optimization starting from x0.
            Returns the best point found and its objective value.
            """
            if time.time() > deadline:
                return x0, None

            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(self.dim):
                xi = x0.copy()
                step = 0.05 * (hi - lo)[i]
                xi[i] += step
                simplex.append(np.clip(xi, lo, hi))

            # Evaluate simplex
            F = []
            for x in simplex:
                _, f = eval_point(x)
                F.append(f)
                if time.time() > deadline:
                    return x0, None

            # Sort by hypervolume loss
            losses = [hypervolume_loss(f) for f in F]
            order = np.argsort(losses)
            simplex = [simplex[i] for i in order]
            F = [F[i] for i in order]
            losses = [losses[i] for i in order]

            best_x = simplex[0]
            best_f = F[0]
            evals_used = self.dim + 1

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            while evals_used < max_evals and time.time() < deadline:
                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                _, fr = eval_point(xr)
                evals_used += 1
                lr = hypervolume_loss(fr)

                if losses[0] <= lr < losses[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    F[-1] = fr
                    losses[-1] = lr
                    if lr < losses[0]:
                        best_x = xr
                        best_f = fr
                    # Re-sort
                    order = np.argsort(losses)
                    simplex = [simplex[i] for i in order]
                    F = [F[i] for i in order]
                    losses = [losses[i] for i in order]
                elif lr < losses[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lo, hi)
                    _, fe = eval_point(xe)
                    evals_used += 1
                    le = hypervolume_loss(fe)
                    if le < lr:
                        simplex[-1] = xe
                        F[-1] = fe
                        losses[-1] = le
                        if le < losses[0]:
                            best_x = xe
                            best_f = fe
                    else:
                        simplex[-1] = xr
                        F[-1] = fr
                        losses[-1] = lr
                    # Re-sort
                    order = np.argsort(losses)
                    simplex = [simplex[i] for i in order]
                    F = [F[i] for i in order]
                    losses = [losses[i] for i in order]
                else:
                    # Contraction
                    if lr < losses[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, lo, hi)
                    _, fc = eval_point(xc)
                    evals_used += 1
                    lc = hypervolume_loss(fc)

                    if lc < max(lr, losses[-1]):
                        simplex[-1] = xc
                        F[-1] = fc
                        losses[-1] = lc
                        if lc < losses[0]:
                            best_x = xc
                            best_f = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            _, fi = eval_point(simplex[i])
                            evals_used += 1
                            losses[i] = hypervolume_loss(fi)
                            F[i] = fi
                        if evals_used >= max_evals:
                            break

                    # Re-sort
                    order = np.argsort(losses)
                    simplex = [simplex[i] for i in order]
                    F = [F[i] for i in order]
                    losses = [losses[i] for i in order]

            return best_x, best_f

        # Phase 1: Stratified random sampling
        n_initial = 40
        # Create stratified samples
        n_per_dim = max(1, int(round(n_initial ** (1.0 / self.dim))))
        grid = np.meshgrid(*[np.linspace(lo[i], hi[i], n_per_dim) for i in range(self.dim)], indexing='ij')
        all_points = np.array([g.ravel() for g in grid]).T

        # Randomly select n_initial points from the grid
        if len(all_points) >= n_initial:
            indices = np.random.choice(len(all_points), size=n_initial, replace=False)
            initial_points = all_points[indices]
        else:
            initial_points = all_points

        # Evaluate initial points
        all_X = []
        all_F = []
        for x in initial_points:
            x, f = eval_point(x)
            all_X.append(x)
            all_F.append(f)
            if time.time() > deadline:
                break

        # Phase 2: Local refinement around best points
        # Select top candidates for local search
        if all_F:
            F_arr = np.array(all_F)
            # Sort by hypervolume loss
            losses = [hypervolume_loss(f) for f in F_arr]
            order = np.argsort(losses)
            
            # Pick top candidates for local search
            n_local = min(10, len(order))
            candidates = []
            for i in order[:n_local]:
                candidates.append(all_X[i])

            # Perform local search on each candidate
            for x0 in candidates:
                if time.time() > deadline:
                    break
                best_x, best_f = nelder_mead(x0, max_evals=15)
                if best_f is not None:
                    all_X.append(best_x)
                    all_F.append(best_f)

        # Phase 3: Additional random sampling with local refinement
        remaining_budget = self.budget - len(all_X)
        n_additional = min(30, remaining_budget)
        for _ in range(n_additional):
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(self.dim)
            x, f = eval_point(x)
            all_X.append(x)
            all_F.append(f)

            # Quick local refinement
            if time.time() < deadline - 10:
                best_x, best_f = nelder_mead(x, max_evals=8)
                if best_f is not None:
                    all_X.append(best_x)
                    all_F.append(best_f)

        # Ensure we don't exceed budget
        all_X = all_X[:self.budget]
        all_F = all_F[:self.budget]

        return all_X
# EVOLVE-BLOCK-END
