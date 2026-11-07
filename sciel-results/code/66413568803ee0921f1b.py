# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE37 (4 variables, 3 objectives, bounds [0,1]).
    Strategy:
      1. Halton sequence for deterministic space-filling initial design.
      2. Archive-based non-dominated point tracking with dominance checks.
      3. Local refinement of promising non-dominated points using Nelder-Mead
         on weighted-sum scalarization with diverse weight vectors.
      4. Deadline guard to stay within 300s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim, skip=0):
        """Generate Halton sequence points in [0,1]^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2
            for i in range(n):
                idx = i + skip + 1
                f = 1.0
                r = 0.0
                while idx > 0:
                    idx, rem = divmod(idx, base)
                    f /= base
                    r += f * rem
                points[i, d] = r
        return points

    def _is_dominated(self, f, archive_f):
        """Check if f is dominated by any point in archive_f."""
        for af in archive_f:
            if np.all(af <= f) and np.any(af < f):
                return True
        return False

    def _add_to_archive(self, x, f, archive_x, archive_f):
        """Add point to archive if not dominated, remove dominated points."""
        # Check if new point is dominated
        for af in archive_f:
            if np.all(af <= f) and np.any(af < f):
                return False  # Don't add if dominated
        
        # Remove points dominated by new point
        new_archive_x = []
        new_archive_f = []
        for i in range(len(archive_f)):
            if not (np.all(f <= archive_f[i]) and np.any(f < archive_f[i])):
                new_archive_x.append(archive_x[i])
                new_archive_f.append(archive_f[i])
        
        # Add new point
        new_archive_x.append(x)
        new_archive_f.append(f)
        
        archive_x[:] = new_archive_x
        archive_f[:] = new_archive_f
        return True

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate weight vectors for weighted-sum scalarization
        n_weights_per_dim = 6
        weights = []
        if n_obj == 3:
            for i in range(n_weights_per_dim + 1):
                for j in range(n_weights_per_dim + 1 - i):
                    k = n_weights_per_dim - i - j
                    w = np.array([i, j, k], dtype=float) / n_weights_per_dim
                    if w.sum() > 0:
                        weights.append(w)
        else:
            rng_w = np.random.RandomState(0)
            for _ in range(50):
                w = rng_w.dirichlet(np.ones(n_obj))
                weights.append(w)

        weights = np.array(weights)
        n_weights = len(weights)

        # Phase 1: Halton sequence initialization
        n_halton = min(budget // 2, 100)
        halton_points = self._halton(n_halton, dim)
        halton_points = lo + halton_points * (hi - lo)

        archive_x = []
        archive_f = []
        all_points = []
        all_fvals = []
        eval_count = 0

        # Evaluate Halton points
        for i in range(n_halton):
            if time.time() > deadline or eval_count >= budget:
                break
            x = halton_points[i]
            f = np.array(problem(x), dtype=float)
            all_points.append(x)
            all_fvals.append(f)
            eval_count += 1
            self._add_to_archive(x, f, archive_x, archive_f)

        # Phase 2: Local refinement of non-dominated points
        from scipy.optimize import minimize

        # Select top non-dominated points for refinement
        n_refine = min(len(archive_x), budget - eval_count - n_weights // 2, 20)
        
        # Create diverse weight vectors for refinement
        rng = np.random.RandomState(42)
        refine_weights = []
        for _ in range(n_weights):
            w = rng.dirichlet(np.ones(n_obj))
            refine_weights.append(w)
        refine_weights = np.array(refine_weights)

        # Refine each archive point with multiple weight vectors
        for ax in list(archive_x):
            if time.time() > deadline or eval_count >= budget:
                break
            
            for w in refine_weights[:5]:  # Use first 5 weights per point
                if eval_count >= budget:
                    break
                
                x0 = ax.copy()
                
                def objective(x, w=w):
                    nonlocal eval_count
                    if time.time() > deadline:
                        return 1e10
                    if eval_count >= budget:
                        return 1e10
                    fx = np.array(problem(x), dtype=float)
                    eval_count += 1
                    return np.dot(w, fx)

                try:
                    res = minimize(objective, x0, method='Nelder-Mead',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 30, 'xatol': 1e-5, 'fatol': 1e-7})
                    if res.fun < 1e9:
                        x_best = np.clip(res.x, lo, hi)
                        f_best = np.array(problem(x_best), dtype=float)
                        eval_count += 1
                        all_points.append(x_best)
                        all_fvals.append(f_best)
                        self._add_to_archive(x_best, f_best, archive_x, archive_f)
                except Exception:
                    pass

                if eval_count >= budget:
                    break

        # Phase 3: Fill remaining budget with random sampling
        if eval_count < budget:
            rng_fill = np.random.RandomState(123)
            for _ in range(budget - eval_count):
                if time.time() > deadline:
                    break
                x = lo + rng_fill.random(dim) * (hi - lo)
                f = np.array(problem(x), dtype=float)
                all_points.append(x)
                all_fvals.append(f)
                eval_count += 1
                self._add_to_archive(x, f, archive_x, archive_f)

        if not all_points:
            return lo

        all_points = np.array(all_points)
        all_fvals = np.array(all_fvals)

        return all_points[0]
# EVOLVE-BLOCK-END
