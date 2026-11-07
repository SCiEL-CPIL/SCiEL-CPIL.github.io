# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using stratified sampling with focused local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        evaluated_points = []
        evaluated_f = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            evaluated_points.append(x.copy())
            evaluated_f.append(f.copy())
            eval_count += 1
            return f

        # Phase 1: Stratified sampling for efficient coverage
        # Use a combination of corners, midpoints, and stratified random points
        rng = np.random.default_rng(42)
        
        # Start with structured points: corners and edge midpoints
        structured = []
        # Corners
        for i in range(2 ** self.dim):
            coords = []
            for d in range(self.dim):
                if (i >> d) & 1:
                    coords.append(hi[d])
                else:
                    coords.append(lo[d])
            structured.append(np.array(coords))
        # Center
        structured.append((lo + hi) / 2.0)
        # Edge midpoints
        for d in range(self.dim):
            for val in [lo[d] + 0.25 * (hi[d] - lo[d]), lo[d] + 0.75 * (hi[d] - lo[d])]:
                for e in range(self.dim):
                    if e != d:
                        pt = (lo + hi) / 2.0
                        pt[d] = val
                        structured.append(pt)
        
        # Add stratified random points to reach ~40 total
        n_structured = len(structured)
        n_random = max(0, 40 - n_structured)
        for _ in range(n_random):
            # Stratified: divide each dimension into 4 bins, pick one per bin
            x = np.zeros(self.dim)
            for d in range(self.dim):
                bin_idx = rng.integers(0, 4)
                x[d] = lo[d] + (bin_idx + rng.random()) * (hi[d] - lo[d]) / 4.0
            structured.append(x)
        
        # Evaluate structured points
        for x in structured:
            result = evaluate(x)
            if result is None:
                break

        if eval_count >= self.budget or time.time() > deadline:
            return np.array(evaluated_points)

        # Build array of evaluated points
        F_arr = np.array(evaluated_f)
        X_arr = np.array(evaluated_points)

        # Phase 2: Identify non-dominated points
        def is_pareto_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            if len(F_set) == 0:
                return False
            le = np.all(F_set <= f, axis=1)
            lt = np.any(F_set < f, axis=1)
            return np.any(le & lt)

        # Find non-dominated points
        pareto_indices = []
        for i, f in enumerate(F_arr):
            others = np.delete(F_arr, i, axis=0)
            if not is_pareto_dominated(f, others):
                pareto_indices.append(i)

        # Select refinement targets: best per objective + top Pareto points
        refinement_targets = []
        for obj_idx in range(self.n_obj):
            best_idx = np.argmin(F_arr[:, obj_idx])
            if best_idx not in refinement_targets:
                refinement_targets.append(best_idx)

        # Add Pareto points for diversity, sorted by a proxy for quality
        for idx in pareto_indices:
            if idx not in refinement_targets and len(refinement_targets) < 8:
                refinement_targets.append(idx)

        # Phase 3: Systematic weight vectors for 3-objective optimization
        # Use a regular grid on the 2-simplex for weight vectors
        n_weights_per_axis = 3  # 3 levels per axis gives 6 points on the 2-simplex
        weights = []
        for i in range(n_weights_per_axis + 1):
            for j in range(n_weights_per_axis + 1 - i):
                k = n_weights_per_axis - i - j
                w = np.array([i, j, k], dtype=float) / n_weights_per_axis
                weights.append(w)
        
        # Add some random weights for diversity
        rng2 = np.random.default_rng(123)
        for _ in range(4):
            w = rng2.dirichlet(np.ones(self.n_obj) * 0.8)
            weights.append(w)
        
        weights = np.array(weights)
        
        # Sort weights to prioritize balanced ones first
        weight_order = np.argsort(weights[:, 0] + weights[:, 1] + weights[:, 2] * 2)
        weights = weights[weight_order]

        # Phase 4: Local refinement using Nelder-Mead with weighted sums
        for target_idx in refinement_targets:
            if eval_count >= self.budget or time.time() > deadline:
                break

            x0 = X_arr[target_idx].copy()

            for w in weights:
                if eval_count >= self.budget or time.time() > deadline:
                    break

                # Define scalarized objective
                def make_scalar_obj(w_vec):
                    def scalar_obj(x):
                        if time.time() > deadline:
                            return 1e10
                        x = np.clip(x, lo, hi)
                        f = np.array(problem(x), dtype=float)
                        evaluated_points.append(x.copy())
                        evaluated_f.append(f.copy())
                        eval_count += 1
                        return float(f @ w_vec)
                    return scalar_obj

                scalar_obj = make_scalar_obj(w)

                # Run Nelder-Mead with moderate iterations
                try:
                    result = minimize(scalar_obj, x0, method='Nelder-Mead',
                                      options={'maxiter': 25, 'xatol': 1e-5, 'fatol': 1e-5,
                                               'initial_simplex_size': 0.15})
                except Exception:
                    pass

        # Phase 5: Final polish pass on the best non-dominated points
        if eval_count < self.budget and time.time() < deadline:
            F_final = np.array(evaluated_f)
            # Find non-dominated points in final set
            final_pareto = []
            for i, f in enumerate(F_final):
                others = np.delete(F_final, i, axis=0)
                if not is_pareto_dominated(f, others):
                    final_pareto.append(i)

            # Select top points for final polish
            polish_targets = final_pareto[:min(6, len(final_pareto))]

            # Use a few more weight vectors for final refinement
            polish_weights = []
            rng3 = np.random.default_rng(999)
            for _ in range(4):
                w = rng3.dirichlet(np.ones(self.n_obj) * 0.6)
                polish_weights.append(w)
            polish_weights.append(np.ones(self.n_obj) / self.n_obj)
            polish_weights = np.array(polish_weights)

            for target_idx in polish_targets:
                if eval_count >= self.budget or time.time() > deadline:
                    break

                x0 = evaluated_points[target_idx].copy()

                for w in polish_weights:
                    if eval_count >= self.budget or time.time() > deadline:
                        break

                    def make_polish_obj(w_vec):
                        def polish_obj(x):
                            if time.time() > deadline:
                                return 1e10
                            x = np.clip(x, lo, hi)
                            f = np.array(problem(x), dtype=float)
                            evaluated_points.append(x.copy())
                            evaluated_f.append(f.copy())
                            eval_count += 1
                            return float(f @ w_vec)
                        return polish_obj

                    polish_obj = make_polish_obj(w)

                    try:
                        result = minimize(polish_obj, x0, method='Nelder-Mead',
                                          options={'maxiter': 30, 'xatol': 1e-6, 'fatol': 1e-6,
                                                   'initial_simplex_size': 0.05})
                    except Exception:
                        pass

        return np.array(evaluated_points)
# EVOLVE-BLOCK-END
