"""Multi-objective optimizer for RE34 using multi-start Nelder-Mead local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer for 3-objective RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 325.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # We have 200 evaluations. Use 10 restarts of 20 evals each.
        n_restarts = 10
        evals_per_restart = self.budget // n_restarts  # 20
        used = 0

        # Generate diverse starting points using Latin hypercube sampling
        # for better coverage of the [1,3]^5 space
        lhs_points = []
        for i in range(n_restarts):
            # Spread points across the space
            for d in range(self.dim):
                # LHS: divide [0,1] into n_restarts strata, pick one per stratum
                u = rng.random()
                x_d = (i + u) / n_restarts
                lhs_points.append(x_d)
        
        # Reshape to (n_restarts, dim)
        lhs_starts = np.array(lhs_points).reshape(n_restarts, self.dim)
        # Map to actual bounds
        lhs_starts = lo + (hi - lo) * lhs_starts

        # We'll keep track of all evaluated points to compute hypervolume later
        all_points = []

        def nelder_mead(problem, x0, bounds, max_evals, used_counter, all_points, deadline, rng):
            """
            Run Nelder-Mead on a scalarized objective (sum of normalized objectives).
            Returns the best point found and updates used_counter, all_points.
            """
            lo_b, hi_b = bounds
            dim = len(x0)
            n_obj = 3

            # Initialize simplex: n+1 points
            # Start with x0, then perturb each dimension
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                step = 0.1 * (hi_b[i] - lo_b[i])
                xi[i] += step
                xi = np.clip(xi, lo_b, hi_b)
                simplex.append(xi)

            # Evaluate initial simplex
            def evaluate(x):
                x = np.clip(x, lo_b, hi_b)
                f = problem(x)
                all_points.append(f)
                used_counter[0] += 1
                return f

            f_values = []
            for i in range(len(simplex)):
                if used_counter[0] >= max_evals or time.time() >= deadline:
                    break
                f_values.append(evaluate(simplex[i]))

            if len(f_values) < 2:
                return x0, f_values[0] if f_values else None

            # Scalarize: use weighted sum with adaptive weights
            # For simplicity, use equal weights but could try Tchebycheff
            def scalarize(f):
                return np.sum(f)

            # Main Nelder-Mead loop
            best_x = simplex[0].copy()
            best_f = f_values[0]
            best_scalar = scalarize(best_f)

            # Sort simplex by scalar objective
            order = np.argsort([scalarize(f) for f in f_values])
            simplex = [simplex[i].copy() for i in order]
            f_values = [f_values[i].copy() for i in order]

            alpha, gamma, rho = 1.0, 2.0, 0.5  # Standard NM parameters
            iteration = 0
            max_iterations = max_evals - len(simplex)

            while iteration < max_iterations and used_counter[0] < max_evals and time.time() < deadline:
                # Current worst point
                worst_idx = len(simplex) - 1
                second_worst_idx = worst_idx - 1

                # Centroid of all but worst
                centroid = np.mean([simplex[i] for i in range(worst_idx)], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[worst_idx])
                x_r = np.clip(x_r, lo_b, hi_b)

                if used_counter[0] >= max_evals or time.time() >= deadline:
                    break
                f_r = evaluate(x_r)
                s_r = scalarize(f_r)

                if s_r < scalarize(f_values[0]):
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, lo_b, hi_b)
                    if used_counter[0] >= max_evals or time.time() >= deadline:
                        # Keep reflection
                        simplex[worst_idx] = x_r
                        f_values[worst_idx] = f_r
                        break
                    f_e = evaluate(x_e)
                    s_e = scalarize(f_e)
                    if s_e < s_r:
                        simplex[worst_idx] = x_e
                        f_values[worst_idx] = f_e
                    else:
                        simplex[worst_idx] = x_r
                        f_values[worst_idx] = f_r
                elif s_r < scalarize(f_values[worst_idx]):
                    simplex[worst_idx] = x_r
                    f_values[worst_idx] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[worst_idx] - centroid)
                    x_c = np.clip(x_c, lo_b, hi_b)
                    if used_counter[0] >= max_evals or time.time() >= deadline:
                        break
                    f_c = evaluate(x_c)
                    s_c = scalarize(f_c)
                    if s_c < scalarize(f_values[worst_idx]):
                        simplex[worst_idx] = x_c
                        f_values[worst_idx] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo_b, hi_b)
                            if used_counter[0] >= max_evals or time.time() >= deadline:
                                break
                            f_values[i] = evaluate(simplex[i])

                # Update best
                for i in range(len(simplex)):
                    s_i = scalarize(f_values[i])
                    if s_i < best_scalar:
                        best_scalar = s_i
                        best_f = f_values[i].copy()
                        best_x = simplex[i].copy()

                # Re-sort simplex
                order = np.argsort([scalarize(f) for f in f_values])
                simplex = [simplex[i].copy() for i in order]
                f_values = [f_values[i].copy() for i in order]

                iteration += 1

            return best_x, best_f

        # Run multi-start
        used_counter = [used]
        for r in range(n_restarts):
            if used_counter[0] >= self.budget or time.time() >= deadline:
                break

            x0 = lhs_starts[r]
            # Add small random perturbation for diversity
            x0 = x0 + 0.05 * (hi - lo) * rng.standard_normal(self.dim)
            x0 = np.clip(x0, lo, hi)

            nelder_mead(problem, x0, (lo, hi), evals_per_restart, used_counter, all_points, deadline, rng)

        # Ensure we don't exceed budget
        # The used_counter tracks evaluations, so we're safe

        return all_points
# EVOLVE-BLOCK-END
