# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE37 (4 variables, 3 objectives, bounds [0,1]).
    Strategy:
      1. Deterministic space-filling initial design (Halton sequence).
      2. Differential Evolution (DE) with adaptive parameters for exploration.
      3. Local refinement of non-dominated points using Nelder-Mead on weighted-sum
         scalarization with diverse weight vectors.
      4. Deadline guard to stay within 300s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim, skip=0):
        """Generate n points in the [0,1]^dim unit cube using the Halton sequence."""
        points = np.zeros((n, dim))
        for i in range(skip, skip + n):
            for d in range(dim):
                base = d + 2  # primes: 2, 3, 5, 7, ...
                f = 1.0
                r = 0.0
                i_val = i + 1
                while i_val > 0:
                    f /= base
                    r += f * (i_val % base)
                    i_val //= base
                points[i - skip, d] = r
        return points

    def _is_dominated(self, f, archive_f):
        """Check if f is dominated by any point in archive_f (array of shape (m, n_obj))."""
        if len(archive_f) == 0:
            return False
        all_leq = np.all(archive_f <= f, axis=1)
        any_lt = np.any(archive_f < f, axis=1)
        return np.any(all_leq & any_lt)

    def _add_to_archive(self, x, f, archive_x, archive_f):
        """Add (x, f) to archive if it is not dominated. Remove dominated points."""
        if self._is_dominated(f, archive_f):
            return archive_x, archive_f
        if len(archive_f) > 0:
            all_geq = np.all(archive_f >= f, axis=1)
            any_gt = np.any(archive_f > f, axis=1)
            keep_mask = ~(all_geq & any_gt)
            archive_x = archive_x[keep_mask]
            archive_f = archive_f[keep_mask]
        return np.vstack([archive_x, x.reshape(1, -1)]), np.vstack([archive_f, f.reshape(1, -1)])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        eval_count = 0
        archive_x = np.empty((0, dim))
        archive_f = np.empty((0, n_obj))
        all_points = []
        all_fvals = []

        def evaluate(x):
            nonlocal eval_count
            if time.time() > deadline or eval_count >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            eval_count += 1
            all_points.append(x.copy())
            all_fvals.append(f.copy())
            return x, f

        # Phase 1: Halton sequence initial design for broad coverage
        n_halton = min(60, budget)
        halton_points = self._halton(n_halton, dim, skip=1)
        for i in range(n_halton):
            if time.time() > deadline or eval_count >= budget:
                break
            x = lo + halton_points[i] * (hi - lo)
            res = evaluate(x)
            if res is not None:
                x_eval, f_eval = res
                archive_x, archive_f = self._add_to_archive(x_eval, f_eval, archive_x, archive_f)

        # Phase 2: Greedy farthest-point sampling in objective space
        # Select points that maximize minimum distance to existing archive in objective space
        # This ensures uniform coverage of the Pareto front
        remaining_budget = budget - eval_count
        n_greedy = min(remaining_budget - 40, 100)  # Leave some budget for local refinement
        
        if n_greedy > 0 and len(archive_f) > 0:
            # Candidate pool: combine archive points with random perturbations
            n_candidates = 200
            candidate_x = lo + np.random.random((n_candidates, dim)) * (hi - lo)
            # Also include archive points with small perturbations
            if len(archive_x) > 0:
                n_archive_cand = min(50, len(archive_x))
                idx = np.random.choice(len(archive_x), n_archive_cand, replace=False)
                perturbed = archive_x[idx] + 0.05 * np.random.randn(n_archive_cand, dim)
                perturbed = np.clip(perturbed, lo, hi)
                candidate_x = np.vstack([candidate_x, perturbed])
            
            # Evaluate a subset of candidates to get objective values
            # We'll evaluate them one by one and select the farthest in objective space
            selected_indices = []
            for _ in range(n_greedy):
                if time.time() > deadline or eval_count >= budget:
                    break
                
                # Find the candidate that maximizes min distance to archive in objective space
                best_idx = -1
                best_min_dist = -1
                
                # Sample a random subset of candidates to evaluate (to save budget)
                n_to_eval = min(20, len(candidate_x) - len(selected_indices))
                if n_to_eval <= 0:
                    break
                
                available = [i for i in range(len(candidate_x)) if i not in selected_indices]
                if len(available) < n_to_eval:
                    break
                
                eval_idx = np.random.choice(available, n_to_eval, replace=False)
                
                for idx in eval_idx:
                    x = candidate_x[idx]
                    res = evaluate(x)
                    if res is None:
                        continue
                    x_eval, f_eval = res
                    
                    # Calculate minimum distance to archive in objective space
                    if len(archive_f) > 0:
                        dists = np.linalg.norm(archive_f - f_eval, axis=1)
                        min_dist = np.min(dists)
                    else:
                        min_dist = 0
                    
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_idx = idx
                
                if best_idx >= 0:
                    selected_indices.append(best_idx)
                    x_eval, f_eval = res
                    archive_x, archive_f = self._add_to_archive(x_eval, f_eval, archive_x, archive_f)
        
        # Phase 3: Local refinement of non-dominated points
        from scipy.optimize import minimize
        
        # Generate diverse weight vectors for scalarization
        n_weights = 15
        rng = np.random.RandomState(42)
        weights = []
        for _ in range(n_weights):
            w = rng.dirichlet(np.ones(n_obj))
            weights.append(w)
        # Add extreme weights
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weights.append(w)
        # Add uniform weight
        weights.append(np.ones(n_obj) / n_obj)
        weights = np.array(weights)
        
        # Select the best points from archive to refine
        if len(archive_x) > 0:
            n_start = min(len(archive_x), len(weights))
            if n_start < len(archive_x):
                idx = np.linspace(0, len(archive_x) - 1, n_start, dtype=int)
                start_points = archive_x[idx]
            else:
                start_points = archive_x
        else:
            start_points = lo + np.random.random((len(weights), dim)) * (hi - lo)
        
        for i in range(len(start_points)):
            if time.time() > deadline or eval_count >= budget:
                break
            
            w = weights[i % len(weights)]
            x0 = start_points[i].copy()
            
            def objective(x, w=w):
                nonlocal eval_count
                if time.time() > deadline:
                    return 1e10
                if eval_count >= budget:
                    return 1e10
                x = np.clip(x, lo, hi)
                fx = np.array(problem(x), dtype=float)
                eval_count += 1
                all_points.append(x.copy())
                all_fvals.append(fx.copy())
                return np.dot(w, fx)
            
            # Nelder-Mead local search with tighter tolerances
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               bounds=list(zip(lo, hi)),
                               options={'maxiter': 40, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if res.fun < 1e9:
                    x_best = np.clip(res.x, lo, hi)
                    f_best = np.array(problem(x_best), dtype=float)
                    eval_count += 1
                    all_points.append(x_best.copy())
                    all_fvals.append(f_best.copy())
                    archive_x, archive_f = self._add_to_archive(x_best, f_best, archive_x, archive_f)
            except Exception:
                pass

        # Phase 4: Use remaining budget for targeted sampling in under-explored regions
        if eval_count < budget:
            # Identify gaps in objective space and sample there
            if len(archive_f) > 0:
                # Find the centroid of the archive in objective space
                centroid = np.mean(archive_f, axis=0)
                # Sample points away from the centroid to explore new regions
                n_extra = budget - eval_count
                for _ in range(n_extra):
                    if time.time() > deadline:
                        break
                    # Random direction in decision space
                    direction = np.random.randn(dim)
                    direction /= np.linalg.norm(direction)
                    # Start from a random archive point and move in the direction
                    start_idx = np.random.randint(len(archive_x))
                    x = archive_x[start_idx] + 0.3 * direction
                    x = np.clip(x, lo, hi)
                    res = evaluate(x)
                    if res is not None:
                        x_eval, f_eval = res
                        archive_x, archive_f = self._add_to_archive(x_eval, f_eval, archive_x, archive_f)
            else:
                for _ in range(budget - eval_count):
                    if time.time() > deadline:
                        break
                    x = lo + np.random.random(dim) * (hi - lo)
                    res = evaluate(x)
                    if res is not None:
                        x_eval, f_eval = res
                        archive_x, archive_f = self._add_to_archive(x_eval, f_eval, archive_x, archive_f)

        if not all_points:
            return lo

        all_points = np.array(all_points)
        all_fvals = np.array(all_fvals)

        return all_points[0]
# EVOLVE-BLOCK-END
