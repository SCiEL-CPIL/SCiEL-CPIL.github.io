"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Grid-based multi-objective optimizer with local refinement for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin: evaluator kills at 330s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.RandomState(42)

        used = 0
        records = []

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            x_clipped = np.clip(x, lo, hi)
            used += 1
            f = np.array(problem(x_clipped), dtype=float)
            records.append(f)
            return f

        def dominated(f1, f2):
            """Check if f1 dominates f2 (all <= and at least one <)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_non_dominated(f, records_list):
            """Check if f is non-dominated w.r.t. records_list."""
            for r in records_list:
                if dominated(r, f):
                    return False
            return True

        def nelder_mead_refine(x0, problem, bounds, max_iter=50, tol=1e-8):
            """Local refinement using Nelder-Mead on a weighted scalarization."""
            nonlocal used
            if used >= self.budget:
                return x0

            # Get current objective values to determine weights
            f0 = np.array(problem(np.clip(x0, lo, hi)), dtype=float)
            if not np.all(np.isfinite(f0)):
                return x0

            # Use a simple weighted sum based on inverse of current values
            weights = 1.0 / (np.abs(f0) + 1e-6)
            weights /= np.sum(weights)

            def scalarized(x):
                nonlocal used
                if used >= self.budget:
                    return float('inf')
                x_c = np.clip(x, lo, hi)
                used += 1
                f = np.array(problem(x_c), dtype=float)
                records.append(f)
                if not np.all(np.isfinite(f)):
                    return float('inf')
                return float(np.dot(weights, f))

            # Nelder-Mead parameters
            x_init = x0.copy()
            simplex = [x_init]
            for i in range(self.dim):
                dx = x_init.copy()
                dx[i] += 0.05 * (hi[i] - lo[i])
                simplex.append(np.clip(dx, lo, hi))

            f_simplex = [scalarized(s) for s in simplex]
            if used >= self.budget:
                return x_init

            best_idx = np.argmin(f_simplex)
            best_x = simplex[best_idx].copy()
            best_f = f_simplex[best_idx]

            for _ in range(max_iter):
                if time.time() > deadline or used >= self.budget:
                    break

                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                if f_simplex[-1] - f_simplex[0] < tol:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, lo, hi)
                f_ref = scalarized(x_ref)
                if used >= self.budget:
                    break

                if f_ref < f_simplex[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = scalarized(x_exp)
                    if used >= self.budget:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = scalarized(x_con)
                    if used >= self.budget:
                        break
                    if f_con < f_simplex[-1]:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if used >= self.budget:
                                break
                            f_simplex[i] = scalarized(simplex[i])
                        if used >= self.budget:
                            break

                # Track best
                best_idx_local = np.argmin(f_simplex)
                if f_simplex[best_idx_local] < best_f:
                    best_f = f_simplex[best_idx_local]
                    best_x = simplex[best_idx_local].copy()

            return best_x

        # Phase 1: Structured grid for initial coverage
        # Use a grid that covers the space uniformly
        # For 12 dims, we need to be clever about grid resolution
        # Use a stratified sampling approach

        # First, do a coarse grid to find promising regions
        # We'll use a hypercube grid with limited points per dimension
        # Given budget=200, let's use:
        # - ~100 for initial structured sampling
        # - ~100 for local refinement

        # Stratified sampling: divide each dimension into segments
        n_per_dim = 3  # 3^12 = 531441, too many
        # Instead, use a more efficient approach: Latin Hypercube-like sampling

        # Phase 1: Random + stratified initial population
        n_initial = 80
        # Stratified sampling: ensure coverage in each dimension
        # Use a simple approach: random but with stratification in key dimensions

        # Actually, let's use a smarter approach:
        # 1. Sample from the center and edges to find the Pareto front region
        # 2. Then refine locally

        # Initial diverse sampling
        candidates = []

        # Center point
        center = (lo + hi) / 2.0
        candidates.append(center)

        # Corner points (but only a subset due to dimensionality)
        # Use a space-filling approach: Halton-like sequence
        def halton_sequence(n, dims):
            """Generate n points in [0,1]^dims using Halton sequence."""
            points = []
            for i in range(1, n + 1):
                point = []
                for d in range(1, dims + 1):
                    f = 1.0
                    p = i
                    while p > 0:
                        p, r = divmod(p, d)
                        f /= d
                        point.append(r * f)
                    points.append(point)
                points = points  # keep as list of lists
            return np.array(points).T if len(points) > 0 else np.array([]).reshape(0, dims)

        # Generate Halton points for structured coverage
        n_halton = 60
        halton_pts = []
        for i in range(1, n_halton + 1):
            point = []
            for d in range(1, self.dim + 1):
                f = 1.0
                p = i
                rad = 0.0
                while p > 0:
                    p, r = divmod(p, d)
                    f /= d
                    rad += r * f
                point.append(rad)
            halton_pts.append(point)
        halton_pts = np.array(halton_pts)

        # Map to bounds
        for hp in halton_pts:
            x = lo + hp * (hi - lo)
            candidates.append(x)

        # Add some random points for diversity
        n_random = n_initial - 1 - n_halton
        if n_random > 0:
            for _ in range(n_random):
                x = lo + rng.random(self.dim) * (hi - lo)
                candidates.append(x)

        # Evaluate initial candidates
        initial_X = []
        initial_F = []
        for x in candidates:
            if used >= self.budget or time.time() > deadline:
                break
            x_c = np.clip(x, lo, hi)
            f = evaluate(x_c)
            if f is not None:
                initial_X.append(x_c)
                initial_F.append(f)

        if len(initial_X) == 0:
            # Fallback: just return what we have
            if records:
                return records[0]
            return lo

        # Phase 2: Local refinement on the most promising points
        # Identify non-dominated points and refine them
        initial_F_arr = np.array(initial_F)
        initial_X_arr = np.array(initial_X)

        # Find non-dominated points
        non_dom_indices = []
        for i in range(len(initial_F)):
            is_nd = True
            for j in range(len(initial_F)):
                if i == j:
                    continue
                if dominated(initial_F[j], initial_F[i]):
                    is_nd = False
                    break
            if is_nd:
                non_dom_indices.append(i)

        # Sort non-dominated points by their sum of objectives (proxy for quality)
        if non_dom_indices:
            sums = [np.sum(initial_F[i]) for i in non_dom_indices]
            non_dom_indices.sort(key=lambda i: sums[i])
        else:
            # Fallback: use all points
            non_dom_indices = list(range(len(initial_F)))
            sums = [np.sum(initial_F[i]) for i in non_dom_indices]
            non_dom_indices.sort(key=lambda i: sums[i])

        # Refine the top points with local optimization
        n_refine = min(15, len(non_dom_indices))
        for idx in non_dom_indices[:n_refine]:
            if used >= self.budget or time.time() > deadline:
                break

            x0 = initial_X_arr[idx]
            # Try refinement with different weight vectors to explore different parts of PF
            # Use the current objective values to guide weights
            f0 = initial_F_arr[idx]

            # Try a few different scalarizations
            n_local_trials = 2
            for trial in range(n_local_trials):
                if used >= self.budget or time.time() > deadline:
                    break

                # Vary the weights slightly
                if trial == 0:
                    weights = 1.0 / (np.abs(f0) + 1e-6)
                else:
                    # Perturb weights
                    perturb = 1.0 + 0.3 * rng.standard_normal(self.n_obj)
                    perturb = np.abs(perturb)
                    weights = (1.0 / (np.abs(f0) + 1e-6)) * perturb

                weights /= np.sum(weights)

                # Run Nelder-Mead with these weights
                def scalarized_local(x, w=weights):
                    nonlocal used
                    if used >= self.budget:
                        return float('inf')
                    x_c = np.clip(x, lo, hi)
                    used += 1
                    f = np.array(problem(x_c), dtype=float)
                    records.append(f)
                    if not np.all(np.isfinite(f)):
                        return float('inf')
                    return float(np.dot(w, f))

                # Nelder-Mead setup
                x_init = x0.copy()
                simplex = [x_init]
                for i in range(self.dim):
                    dx = x_init.copy()
                    step = 0.1 * (hi[i] - lo[i])
                    dx[i] += step
                    simplex.append(np.clip(dx, lo, hi))

                f_simplex = [scalarized_local(s) for s in simplex]
                if used >= self.budget:
                    continue

                for _ in range(30):  # limited iterations per local optimization
                    if time.time() > deadline or used >= self.budget:
                        break

                    order = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in order]
                    f_simplex = [f_simplex[i] for i in order]

                    if f_simplex[-1] - f_simplex[0] < 1e-10:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    x_ref = centroid + (centroid - simplex[-1])
                    x_ref = np.clip(x_ref, lo, hi)
                    f_ref = scalarized_local(x_ref)
                    if used >= self.budget:
                        break

                    if f_ref < f_simplex[0]:
                        x_exp = centroid + 2.0 * (centroid - simplex[-1])
                        x_exp = np.clip(x_exp, lo, hi)
                        f_exp = scalarized_local(x_exp)
                        if used >= self.budget:
                            break
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            f_simplex[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            f_simplex[-1] = f_ref
                    elif f_ref < f_simplex[-2]:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                    else:
                        x_con = centroid + 0.5 * (simplex[-1] - centroid)
                        x_con = np.clip(x_con, lo, hi)
                        f_con = scalarized_local(x_con)
                        if used >= self.budget:
                            break
                        if f_con < f_simplex[-1]:
                            simplex[-1] = x_con
                            f_simplex[-1] = f_con
                        else:
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                if used >= self.budget:
                                    break
                                f_simplex[i] = scalarized_local(simplex[i])
                            if used >= self.budget:
                                break

        # Phase 3: If budget remains, do additional random sampling + quick local search
        if used < self.budget and time.time() < deadline:
            # Add more diverse points
            n_extra = min(30, self.budget - used - 50)  # leave room for refinement
            for _ in range(n_extra):
                if used >= self.budget or time.time() > deadline:
                    break
                x = lo + rng.random(self.dim) * (hi - lo)
                f = evaluate(x)
                if f is None:
                    break

        # If still have budget, refine the best points found so far
        if used < self.budget and time.time() < deadline:
            # Get all recorded points
            if records:
                all_F = np.array(records)
                # Find non-dominated
                nd_idx = []
                for i in range(len(all_F)):
                    is_nd = True
                    for j in range(len(all_F)):
                        if i == j:
                            continue
                        if dominated(all_F[j], all_F[i]):
                            is_nd = False
                            break
                    if is_nd:
                        nd_idx.append(i)

                if nd_idx:
                    # Sort by sum
                    nd_sums = [np.sum(all_F[i]) for i in nd_idx]
                    nd_idx.sort(key=lambda i: nd_sums[i])

                    # Refine top few more
                    n_more_refine = min(5, len(nd_idx), (self.budget - used) // 20)
                    for idx in nd_idx[:n_more_refine]:
                        if used >= self.budget or time.time() > deadline:
                            break

                        # We don't have the corresponding x, so skip detailed refinement
                        # Instead, do a small random perturbation search
                        base_f = all_F[idx]
                        best_x_local = None
                        best_f_local = base_f.copy()

                        # Try random perturbations
                        n_perturb = min(10, self.budget - used)
                        for _ in range(n_perturb):
                            if used >= self.budget or time.time() > deadline:
                                break
                            # Random point near the center of the bounds
                            x = lo + rng.random(self.dim) * (hi - lo)
                            f = evaluate(x)
                            if f is not None and np.all(f <= best_f_local) and np.any(f < best_f_local):
                                best_f_local = f
                                best_x_local = x

        return records[0] if records else lo

# EVOLVE-BLOCK-END
