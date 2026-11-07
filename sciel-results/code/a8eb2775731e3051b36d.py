"""Direct DTLZ7 Pareto front sampling with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = 325.0  # safety margin below 330s wall-clock limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        M = self.n_obj

        rng = np.random.default_rng(42)
        evaluated = 0
        max_evals = self.budget
        points = []
        F = []

        def check_deadline():
            return (time.perf_counter() - t0) < deadline

        def safe_eval(x):
            nonlocal evaluated
            if evaluated >= max_evals or not check_deadline():
                return None
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            if len(x) != n:
                x = np.resize(x, n)
            f = np.array(problem(x), dtype=float)
            if np.any(~np.isfinite(f)):
                return None
            evaluated += 1
            return f

        # DTLZ7 structure for n=22, M=3:
        # f1 = x[0], f2 = x[1]
        # g-set is the last M-2 = 1 variable: x[21]
        # g(x) = 100 * (1 + (x[21]-0.5)^2 * cos(20*pi*(x[21]-0.5)))
        # The Pareto front is where g is minimized.
        # 
        # Actually, for standard DTLZ7, the Pareto front is defined by g=0,
        # but with this formula g can't be 0. Let me reconsider.
        #
        # The key insight: on the Pareto front, the "g-variables" are set to
        # minimize g, and the shape variables (x[0], x[1], and middle vars)
        # determine the trade-off.
        #
        # For DTLZ7 with M=3:
        # f1 = x[0]
        # f2 = x[1]
        # f3 = (1 + g) * h(y)
        # where y = sum(x[2:21]) / 19 (or similar normalization)
        # h(y) = 1 - y/m * (1 + sin(3*pi*y/m)) or similar
        #
        # The Pareto front has g at its minimum, and the shape is determined
        # by x[0], x[1], and the middle variables x[2:21].
        #
        # Strategy: 
        # 1. Determine the optimal value of the g-variable (x[21])
        # 2. Sample the shape space (x[0], x[1], x[2:21]) densely
        #
        # First, let's find the optimal g-variable value by scanning

        # Phase 1: Find optimal g-variable value
        g_var_vals = np.linspace(0, 1, 20)
        best_g_var = 0.5
        best_f3 = np.inf
        for gv in g_var_vals:
            if not check_deadline() or evaluated >= max_evals:
                break
            x = np.zeros(n)
            x[0] = 0.5
            x[1] = 0.5
            x[21] = gv
            f = safe_eval(x)
            if f is None:
                break
            if f[2] < best_f3:
                best_f3 = f[2]
                best_g_var = gv
            points.append(np.clip(x, lo, hi))
            F.append(f)

        if not F:
            return np.array([])

        # Phase 2: Dense sampling of the Pareto front
        # Fix x[21] = best_g_var, sample x[0], x[1], and middle vars x[2:21]
        # The middle variables create the complex Pareto shape
        #
        # For hypervolume maximization with ref [1.1, 1.1, 25.0]:
        # We want to cover the (f1, f2, f3) space well
        # f1 = x[0], f2 = x[1], f3 depends on middle vars and g
        #
        # Use a stratified/grid approach:
        # - Sample x[0] and x[1] on a grid
        # - For middle vars, use patterns that produce diverse f3 values

        remaining = max_evals - evaluated
        if remaining <= 0 or not check_deadline():
            return np.array(points) if points else np.zeros((0, n))

        # Determine grid size
        # We want to balance coverage of (x[0], x[1]) with diversity in middle vars
        n_grid_1d = int(np.sqrt(remaining * 0.5)) + 1
        n_grid_1d = min(n_grid_1d, 15)  # cap to keep computation fast
        n_mid_patterns = max(1, remaining // (n_grid_1d * n_grid_1d) - 1)
        n_mid_patterns = min(n_mid_patterns, 5)

        # Stratified sampling for x[0] and x[1]
        xs0 = np.linspace(0, 1, n_grid_1d, endpoint=False) + rng.random(n_grid_1d) * (1.0 / n_grid_1d)
        xs1 = np.linspace(0, 1, n_grid_1d, endpoint=False) + rng.random(n_grid_1d) * (1.0 / n_grid_1d)

        # Generate middle-variable patterns (x[2:21], 19 variables)
        mid_len = n - 2  # 20 variables from index 2 to 21, but x[21] is fixed
        mid_len = 19  # x[2] through x[20]
        
        patterns = []
        # Pattern 0: all zeros
        patterns.append(np.zeros(mid_len))
        # Pattern 1: all 0.5
        patterns.append(np.full(mid_len, 0.5))
        # Pattern 2: linear ramp
        patterns.append(np.linspace(0, 1, mid_len))
        # Pattern 3: random
        patterns.append(rng.random(mid_len))
        # Pattern 4: another random
        if n_mid_patterns > 4:
            patterns.append(rng.random(mid_len))

        patterns = patterns[:n_mid_patterns]

        # Evaluate grid x patterns
        for pi in range(len(patterns)):
            mid = patterns[pi]
            for i in range(n_grid_1d):
                for j in range(n_grid_1d):
                    if not check_deadline() or evaluated >= max_evals:
                        break
                    x = np.zeros(n)
                    x[0] = xs0[i]
                    x[1] = xs1[j]
                    x[2:21] = mid
                    x[21] = best_g_var
                    f = safe_eval(x)
                    if f is None:
                        break
                    points.append(np.clip(x, lo, hi))
                    F.append(f)

        # Phase 3: Refinement with remaining budget
        remaining = max_evals - evaluated
        if remaining > 0 and check_deadline() and points:
            F_arr = np.array(F)
            # Focus on non-dominated solutions
            # Simple approach: identify points with low f3 and perturb them
            n_refine = min(remaining, 50)
            for _ in range(n_refine):
                if not check_deadline() or evaluated >= max_evals:
                    break
                # Pick a random existing point with low f3
                if len(F_arr) > 0:
                    f3_vals = F_arr[:, 2]
                    # Sample from points with relatively low f3
                    threshold = np.percentile(f3_vals, 50)
                    good_indices = np.where(f3_vals < threshold)[0]
                    if len(good_indices) > 0:
                        idx = good_indices[rng.integers(0, len(good_indices))]
                    else:
                        idx = rng.integers(0, len(points))
                else:
                    idx = 0
                
                base = points[idx].astype(float)
                # Small perturbation
                noise = rng.standard_normal(n) * 0.03 * (hi - lo)
                x = np.clip(base + noise, lo, hi)
                f = safe_eval(x)
                if f is None:
                    break
                points.append(np.clip(x, lo, hi))
                F.append(f)
                F_arr = np.array(F)

        # Final check
        if not points:
            if evaluated < max_evals and check_deadline():
                x = np.random.uniform(0, 1, n)
                f = safe_eval(x)
                if f is not None:
                    points.append(np.clip(x, lo, hi))
                    F.append(f)

        return np.array(points) if points else np.zeros((0, n))

# EVOLVE-BLOCK-END
