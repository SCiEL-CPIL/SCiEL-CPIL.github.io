import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        # Phase 1: Initialization with LHS
        n_init = min(20, budget)
        if n_init >= budget:
            for _ in range(budget):
                x = lo + (hi - lo) * np.random.random(dim)
                eval_point(x)
            return np.array(all_X)

        # Generate LHS points
        X_init = self._lhs(n_init, dim)
        X_init = lo + X_init * (hi - lo)
        
        # Generate random weight vectors for decomposition
        weights = np.random.random((n_init, n_obj))
        weights /= weights.sum(axis=1, keepdims=True)

        # Evaluate initial points
        for i in range(n_init):
            if used >= budget:
                break
            eval_point(X_init[i])

        # Phase 2: Local Search (Hooke-Jeeves) on each initial point
        # Distribute remaining budget
        remaining = budget - used
        if remaining > 0:
            evals_per_point = remaining // n_init
            # Ensure at least 1 eval per point if possible, or just do what we can
            for i in range(n_init):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x_start = X_init[i].copy()
                w = weights[i]
                
                self._hooke_jeeves(
                    eval_point, x_start, w, lo, hi, 
                    max_evals=evals_per_point, deadline=deadline
                )

        # Phase 3: Use any leftover budget for random search
        if used < budget:
            while used < budget:
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.random(dim)
                eval_point(x)

        return np.array(all_X)

    def _lhs(self, n, dim):
        """Latin Hypercube Sampling in [0, 1]^dim"""
        pts = np.zeros((n, dim))
        for d in range(dim):
            # Divide [0, 1] into n intervals
            intervals = np.linspace(0, 1, n + 1)
            # Pick a random point in each interval
            offsets = np.random.random(n)
            pts[:, d] = intervals[:-1] + offsets * (intervals[1] - intervals[0])
            # Shuffle to decorrelate dimensions
            pts[:, d] = pts[np.random.permutation(n), d]
        return pts

    def _hooke_jeeves(self, eval_func, x_start, weights, lo, hi, max_evals=10, deadline=None):
        """
        Hooke-Jeeves local search minimizing Tchebycheff scalarization.
        """
        dim = len(x_start)
        x = x_start.copy()
        
        # Initial step size: 10% of range
        steps = 0.1 * (hi - lo)
        
        # Get initial fitness
        f_current = np.array(eval_func(x.copy()), dtype=np.float64)
        
        # Ideal point estimate (min of objectives seen so far, or just 0)
        # For Tchebycheff, we need a reference. Let's use the min of the current point's objectives
        # or a running min. Since we don't have global mins easily, we use the current point's values as reference for simplicity,
        # or better, just minimize the weighted sum? 
        # Tchebycheff: max_i (w_i * (f_i - z_i))
        # If we don't know z*, we can use the min f_i observed in this local search or a fixed reference.
        # Let's use the min of the current population if available, but here we are in a local search.
        # Let's just use the weighted sum for simplicity and robustness, or Tchebycheff with z=0.
        # WFG objectives are usually positive. Let's assume z=0.
        
        def scalarize(f):
            # Tchebycheff with z=0
            return np.max(weights * f)

        best_score = scalarize(f_current)
        
        evals_used = 1 # Count the initial eval? No, eval_func was called.
        # The loop below calls eval_func.
        
        for _ in range(max_evals - 1):
            if deadline is not None and time.time() > deadline:
                break
            
            improved = False
            for d in range(dim):
                # Try positive step
                x_new = x.copy()
                x_new[d] += steps[d]
                x_new = np.clip(x_new, lo, hi)
                
                f_new = np.array(eval_func(x_new.copy()), dtype=np.float64)
                score_new = scalarize(f_new)
                
                if score_new < best_score:
                    x = x_new
                    f_current = f_new
                    best_score = score_new
                    improved = True
                    # Expand step in this dimension
                    steps[d] *= 2.0
                    # Check bounds
                    if x[d] >= hi[d]:
                        steps[d] = (hi[d] - x[d]) * 0.5
                    continue # Skip negative try? No, standard HJ tries both.
                
                # Try negative step
                x_new = x.copy()
                x_new[d] -= steps[d]
                x_new = np.clip(x_new, lo, hi)
                
                f_new = np.array(eval_func(x_new.copy()), dtype=np.float64)
                score_new = scalarize(f_new)
                
                if score_new < best_score:
                    x = x_new
                    f_current = f_new
                    best_score = score_new
                    improved = True
                    # Expand step in this dimension
                    steps[d] *= 2.0
                    if x[d] <= lo[d]:
                        steps[d] = (x[d] - lo[d]) * 0.5

            if not improved:
                # Shrink all steps
                steps *= 0.5
                # If steps are too small, stop
                if np.all(steps < 1e-8 * (hi - lo)):
                    break

    def _halton_sequence(self, n, dim, offset=0.0):
        """
        Generate n points in dim dimensions using the Halton sequence.
        Returns array of shape (n, dim) with values in [0, 1).
        """
        # First n primes for Halton bases
        primes = self._first_primes(dim)
        
        points = np.zeros((n, dim))
        for d in range(dim):
            base = primes[d]
            for i in range(n):
                points[i, d] = self._halton_value(i + 1 + int(offset * 1000), base)
        
        return points

    def _halton_value(self, i, base):
        """
        Compute the i-th value in the Halton sequence with given base.
        """
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _first_primes(self, n):
        """
        Return the first n prime numbers.
        """
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes

    def _get_non_dominated_indices(self, F):
        """
        Return indices of non-dominated points in F.
        """
        n = F.shape[0]
        if n == 0:
            return np.array([], dtype=int)
        
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        
        return np.where(~is_dominated)[0]

    def _nelder_mead(self, eval_func, x_start, f_start, lo, hi, max_evals=50, deadline=None):
        """
        Simple Nelder-Mead style optimization for multi-objective.
        We minimize the sum of objectives as a scalar surrogate.
        Returns (best_x, best_f).
        """
        dim = len(x_start)
        n_obj = len(f_start)
        
        # Initialize simplex
        alpha = 0.05  # Step size for initial simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x_start
        f0 = f_start.copy()
        
        for i in range(dim):
            simplex[i + 1] = x_start.copy()
            step = alpha * (hi[i] - lo[i])
            simplex[i + 1, i] += step
            simplex[i + 1] = np.clip(simplex[i + 1], lo, hi)
        
        # Evaluate simplex
        f_values = [f0]
        for i in range(1, dim + 1):
            if deadline is not None and time.time() > deadline:
                break
            f_i = np.array(eval_func(simplex[i].copy()), dtype=np.float64)
            f_values.append(f_i)
        
        # We need to track evaluations - but eval_func already does that
        # Let's use a wrapper that doesn't double-count
        # Actually, eval_func already increments used and records, so we just call it
        
        # Nelder-Mead iterations
        # We'll minimize the sum of objectives as a scalar
        best_x = x_start.copy()
        best_f = f_start.copy()
        best_score = f_start.sum()
        
        for iteration in range(max_evals // (dim + 1)):
            if deadline is not None and time.time() > deadline:
                break
            
            # Sort simplex by objective sum
            scores = [fv.sum() for fv in f_values]
            order = np.argsort(scores)
            
            simplex_sorted = simplex[order]
            f_sorted = [f_values[i] for i in order]
            scores_sorted = [scores[i] for i in order]
            
            # Check for convergence
            if scores_sorted[0] >= scores_sorted[-1] - 1e-10:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex_sorted[:-1], axis=0)
            centroid = np.clip(centroid, lo, hi)
            
            # Reflection
            worst = simplex_sorted[-1]
            f_worst = f_sorted[-1]
            f_worst_score = scores_sorted[-1]
            
            x_ref = centroid + (centroid - worst)
            x_ref = np.clip(x_ref, lo, hi)
            f_ref = np.array(eval_func(x_ref.copy()), dtype=np.float64)
            f_ref_score = f_ref.sum()
            
            if f_ref_score < scores_sorted[0]:
                # Expansion
                x_exp = centroid + 2.0 * (x_ref - centroid)
                x_exp = np.clip(x_exp, lo, hi)
                f_exp = np.array(eval_func(x_exp.copy()), dtype=np.float64)
                f_exp_score = f_exp.sum()
                
                if f_exp_score < f_ref_score:
                    simplex_sorted[-1] = x_exp
                    f_sorted[-1] = f_exp
                    scores_sorted[-1] = f_exp_score
                else:
                    simplex_sorted[-1] = x_ref
                    f_sorted[-1] = f_ref
                    scores_sorted[-1] = f_ref_score
            elif f_ref_score < f_worst_score:
                # Accept reflection
                simplex_sorted[-1] = x_ref
                f_sorted[-1] = f_ref
                scores_sorted[-1] = f_ref_score
            else:
                # Contraction
                x_con = centroid + 0.5 * (worst - centroid)
                x_con = np.clip(x_con, lo, hi)
                f_con = np.array(eval_func(x_con.copy()), dtype=np.float64)
                f_con_score = f_con.sum()
                
                if f_con_score < f_worst_score:
                    simplex_sorted[-1] = x_con
                    f_sorted[-1] = f_con
                    scores_sorted[-1] = f_con_score
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex_sorted[i] = simplex_sorted[0] + 0.5 * (simplex_sorted[i] - simplex_sorted[0])
                        simplex_sorted[i] = np.clip(simplex_sorted[i], lo, hi)
                        f_shrink = np.array(eval_func(simplex_sorted[i].copy()), dtype=np.float64)
                        f_sorted[i] = f_shrink
                        scores_sorted[i] = f_shrink.sum()
            
            # Update simplex and f_values
            simplex = simplex_sorted
            f_values = f_sorted
            
            # Track best
            min_idx = np.argmin(scores_sorted)
            if scores_sorted[min_idx] < best_score:
                best_score = scores_sorted[min_idx]
                best_x = simplex[min_idx].copy()
                best_f = f_values[min_idx].copy()
        
        return best_x, best_f
