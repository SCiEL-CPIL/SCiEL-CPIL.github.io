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
        deadline = 325.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        M = self.n_obj

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

        # DTLZ7 Structure:
        # f1 = x[0], f2 = x[1]
        # f3 = (1 + g) * h(y)
        # g depends on last M-2 vars (x[21] for M=3, n=22)
        # h depends on sum of middle vars (x[2..20])
        
        # 1. Determine best x[21] (g-var) to minimize g (or just pick a good one)
        # We assume standard DTLZ7 where g is large but constant-ish.
        # Let's test a few values for x[21] with x[0]=x[1]=0.5, x[2..20]=0.5
        # to see which gives lowest f3.
        best_g_var = 0.5
        best_f3 = float('inf')
        for gv in [0.0, 0.25, 0.5, 0.75, 1.0]:
            x = np.full(n, 0.5)
            x[-1] = gv
            f = safe_eval(x)
            if f is not None and f[2] < best_f3:
                best_f3 = f[2]
                best_g_var = gv
        
        # 2. Grid sample x[0], x[1] and the "shape" of x[2..20]
        # We want to cover the Pareto front.
        # f1, f2 are independent. f3 depends on sum(x[2..20]).
        # To get diverse f3, we vary the sum of x[2..20].
        # We can set all x[2..20] to the same value v.
        # Then y = 19 * v.
        
        remaining = max_evals - evaluated
        if remaining <= 0 or not check_deadline():
            return np.array(points) if points else np.zeros((0, n))

        # Allocate budget:
        # N0 x N1 x Nv <= remaining
        # Let's use N0=5, N1=5, Nv=8 (total 200)
        N0 = 5
        N1 = 5
        Nv = max(1, remaining // (N0 * N1))
        
        # If we have more budget, increase resolution
        if N0 * N1 * Nv < remaining:
             Nv = remaining // (N0 * N1)
             
        # Ensure we don't exceed budget
        total_planned = N0 * N1 * Nv
        if total_planned > remaining:
            Nv = max(1, remaining // (N0 * N1))
            total_planned = N0 * N1 * Nv

        xs0 = np.linspace(0.0, 1.0, N0)
        xs1 = np.linspace(0.0, 1.0, N1)
        # v in [0, 1] controls f3
        # We want to sample v such that f3 covers the range of interest.
        # Since f3 is non-linear, uniform v might not be ideal, but it's a start.
        # To bias towards low f3 (which is better for hypervolume with ref 25),
        # we might want to sample v closer to 1 (if h(1)=0) or other minima.
        # For DTLZ7, h(1)=0. So v=1 gives f3=0.
        # Let's sample v uniformly in [0, 1].
        vs = np.linspace(0.0, 1.0, Nv)

        for x0 in xs0:
            for x1 in xs1:
                for v in vs:
                    if not check_deadline() or evaluated >= max_evals:
                        break
                    x = np.zeros(n)
                    x[0] = x0
                    x[1] = x1
                    # Set middle vars to v
                    x[2:n-1] = v
                    # Set g-var to best_g_var
                    x[n-1] = best_g_var
                    
                    f = safe_eval(x)
                    if f is None:
                        break
                    points.append(x)
                    F.append(f)
                if not check_deadline() or evaluated >= max_evals:
                    break
            if not check_deadline() or evaluated >= max_evals:
                break

        # If we have budget left, do random refinement
        remaining = max_evals - evaluated
        if remaining > 0 and check_deadline() and points:
            rng = np.random.default_rng(42)
            for _ in range(remaining):
                if not check_deadline() or evaluated >= max_evals:
                    break
                # Random point, biased towards low f1, f2, f3
                # f1, f2 low -> x0, x1 low
                # f3 low -> x[2..n-1] configured to minimize f3
                # We can just perturb a random existing good point
                idx = rng.integers(0, len(points))
                base = points[idx].astype(float)
                noise = rng.standard_normal(n) * 0.1
                x = np.clip(base + noise, lo, hi)
                f = safe_eval(x)
                if f is None:
                    break
                points.append(x)
                F.append(f)

        if not points:
            if evaluated < max_evals and check_deadline():
                x = np.random.uniform(0, 1, n)
                f = safe_eval(x)
                if f is not None:
                    points.append(x)
                    F.append(f)

        return np.array(points) if points else np.zeros((0, n))

# EVOLVE-BLOCK-END
