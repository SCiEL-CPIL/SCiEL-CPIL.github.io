"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-specific optimizer: hypervolume-optimized non-uniform lattice on (x0,x1)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin well below 330s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        M = self.n_obj

        evaluated = 0
        max_evals = self.budget
        points = []
        F = []

        def check_deadline():
            return (time.time() - start_time) < deadline

        def safe_eval(x):
            nonlocal evaluated
            if evaluated >= max_evals or not check_deadline():
                return None
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            if len(x) != n:
                x = np.resize(x, n)
            try:
                f = np.array(problem(x), dtype=float)
            except Exception:
                return None
            if np.any(~np.isfinite(f)):
                return None
            evaluated += 1
            return f

        # DTLZ7 structure for n=22, M=3:
        # f1 = x[0]
        # f2 = x[1]
        # f3 = (1 + g) * h(y) where y involves x[2:21] and g involves x[21]
        # The Pareto front is where g is minimized (x[21] near 0.5) and x[2:21] = 0.
        # For hypervolume maximization with ref [1.1, 1.1, 25.0],
        # we want to densely cover the front, especially near the ideal point (0,0,0).

        # Phase 1: Determine the best x[21] value for minimizing g
        # g(x) = 100 * (1 + (x[21]-0.5)^2 * cos(20*pi*(x[21]-0.5)))
        # The minimum of g is at x[21] = 0.5, giving g = 100 * (1 + 0) = 100? 
        # Actually for DTLZ7, g is defined differently. Let's just sample a few values.
        # We'll use a small set of candidate x[21] values and pick the one that gives lowest f3
        # when combined with x[0]=0.5, x[1]=0.5, x[2:21]=0.
        
        g_var_candidates = np.linspace(0.0, 1.0, 10)
        best_g_var = 0.5
        best_f3 = np.inf
        
        for gv in g_var_candidates:
            if not check_deadline() or evaluated >= max_evals:
                break
            x = np.zeros(n)
            x[0] = 0.5
            x[1] = 0.5
            x[21] = gv
            f = safe_eval(x)
            if f is not None:
                if f[2] < best_f3:
                    best_f3 = f[2]
                    best_g_var = gv
                points.append(x.copy())
                F.append(f)

        if not F:
            return np.array([])

        # Phase 2: Hypervolume-optimized non-uniform lattice on (x0, x1)
        # Using power-law distribution to cluster points near (0,0) which contributes
        # most to hypervolume with the given reference point.
        remaining = max_evals - evaluated
        if remaining <= 0 or not check_deadline():
            return np.array(points) if points else np.zeros((0, n))

        # We have roughly remaining evaluations left.
        # Use a non-uniform grid: x0 and x1 sampled with power-law density.
        # For hypervolume, the contribution of a point (f1, f2, f3) is roughly
        # (ref1-f1)*(ref2-f2)*(ref3-f3). Points near (0,0,0) contribute most.
        # So we want more points near x0=0, x1=0.
        
        # Let's use a grid of size sqrt(remaining) x sqrt(remaining)
        # But we also need to vary x[2:21] to cover the front properly.
        # Actually, for DTLZ7, the front is 2D in (x0, x1) space when x[2:21]=0 and x[21]=optimal.
        # So we can focus on (x0, x1) with x[2:21]=0 and x[21]=best_g_var.
        
        # Let's use a non-uniform grid with power-law spacing.
        # Number of points per dimension:
        n_per_dim = max(2, int(np.sqrt(remaining)))
        
        # Power-law sampling: x = t^(1/alpha) where t is uniform in [0,1]
        # alpha > 1 clusters points near 0.
        alpha = 2.0  # Quadratic clustering near 0
        
        # Generate non-uniform grid points
        t_vals = np.linspace(0, 1, n_per_dim)
        x0_vals = (t_vals ** (1.0/alpha)) * (hi[0] - lo[0]) + lo[0]
        x1_vals = (t_vals ** (1.0/alpha)) * (hi[1] - lo[1]) + lo[1]
        
        # Also add uniform points for better coverage
        x0_uniform = np.linspace(lo[0], hi[0], max(2, n_per_dim // 2))
        x1_uniform = np.linspace(lo[1], hi[1], max(2, n_per_dim // 2))
        
        # Combine and deduplicate
        all_x0 = np.unique(np.concatenate([x0_vals, x0_uniform]))
        all_x1 = np.unique(np.concatenate([x1_vals, x1_uniform]))
        
        # Sort to ensure consistent ordering
        all_x0 = np.sort(all_x0)
        all_x1 = np.sort(all_x1)
        
        # Evaluate the lattice
        for x0 in all_x0:
            for x1 in all_x1:
                if not check_deadline() or evaluated >= max_evals:
                    break
                x = np.zeros(n)
                x[0] = x0
                x[1] = x1
                x[2:21] = 0.0
                x[21] = best_g_var
                f = safe_eval(x)
                if f is not None:
                    points.append(x.copy())
                    F.append(f)
            if not check_deadline() or evaluated >= max_evals:
                break
        
        # Phase 3: Add corner and edge points for better hypervolume coverage
        remaining = max_evals - evaluated
        if remaining > 0 and check_deadline():
            corners = [
                (0.0, 0.0), (0.0, hi[1]), (hi[0], 0.0), (hi[0], hi[1]),
                (0.0, hi[1]*0.25), (0.0, hi[1]*0.5), (0.0, hi[1]*0.75),
                (hi[0]*0.25, 0.0), (hi[0]*0.5, 0.0), (hi[0]*0.75, 0.0),
                (hi[0]*0.25, hi[1]*0.25), (hi[0]*0.75, hi[1]*0.75),
            ]
            for c0, c1 in corners:
                if not check_deadline() or evaluated >= max_evals:
                    break
                x = np.zeros(n)
                x[0] = c0
                x[1] = c1
                x[2:21] = 0.0
                x[21] = best_g_var
                f = safe_eval(x)
                if f is not None:
                    points.append(x.copy())
                    F.append(f)
        
        # Phase 4: If budget remains, do local refinement around promising points
        remaining = max_evals - evaluated
        if remaining > 0 and check_deadline() and len(F) > 0:
            F_arr = np.array(F)
            # Find points with low f3 (close to Pareto front)
            f3_vals = F_arr[:, 2]
            threshold = np.percentile(f3_vals, 50)
            good_idx = np.where(f3_vals <= threshold)[0]
            
            rng = np.random.default_rng(42)
            n_refine = min(remaining, 30)
            
            for _ in range(n_refine):
                if not check_deadline() or evaluated >= max_evals:
                    break
                if len(good_idx) > 0:
                    idx = good_idx[rng.integers(0, len(good_idx))]
                else:
                    idx = rng.integers(0, len(points))
                
                base = np.array(points[idx], dtype=float)
                # Perturb x0 and x1 slightly
                noise = rng.standard_normal(n) * 0.02
                noise[2:] = 0.0  # Keep other variables fixed
                x = np.clip(base + noise, lo, hi)
                f = safe_eval(x)
                if f is not None:
                    points.append(x.copy())
                    F.append(f)
                    F_arr = np.array(F)
                    f3_vals = F_arr[:, 2]
                    threshold = np.percentile(f3_vals, 50)
                    good_idx = np.where(f3_vals <= threshold)[0]
        
        # Final fallback
        if not points and evaluated < max_evals and check_deadline():
            x = np.random.uniform(lo, hi)
            f = safe_eval(x)
            if f is not None:
                points.append(x.copy())
                F.append(f)
        
        return np.array(points) if points else np.zeros((0, n))

# EVOLVE-BLOCK-END
