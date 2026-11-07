import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        call_count = [0]
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal best_x, best_f
            if call_count[0] >= self.budget:
                return float('inf')
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            fx = float(func(x))
            call_count[0] += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_left():
            return deadline - time.time()

        # Phase 1: Latin Hypercube Sampling for global coverage
        # Büche-Rastrigin has periodic structure with period 2
        # Sample points that cover the box well
        n_lhs = 40
        lhs_points = []
        for i in range(n_lhs):
            # Latin hypercube: one point per stratum in each dimension
            point = np.zeros(self.dim)
            for d in range(self.dim):
                # Stratified sampling: divide each dimension into n_lhs strata
                # Pick a random position within the stratum
                stratum = i
                pos = (stratum + rng.random()) / n_lhs
                point[d] = -5 + 10 * pos
            lhs_points.append(point)
        
        # Also add some points near integer values (Rastrigin minima are near integers)
        # Büche-Rastrigin: f(x) = sum(10*(x_i^2 - 1) + cos(2*pi*x_i)) + 10*dim
        # Minima occur where x_i is near integer values
        for d in range(self.dim):
            for val in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
                point = np.zeros(self.dim)
                point[d] = val
                lhs_points.append(point)
        
        # Add origin
        lhs_points.append(np.zeros(self.dim))
        
        # Evaluate LHS points
        for point in lhs_points:
            if time_left() < 5 or call_count[0] >= self.budget:
                break
            point = np.clip(point, -5, 5)
            fx = wrapped_func(point)
        
        if best_x is None:
            return np.zeros(self.dim), 0.0

        # Phase 2: Nelder-Mead refinement from best point
        remaining_budget = self.budget - call_count[0]
        if remaining_budget > 20 and time_left() > 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f

            # Initialize simplex with adaptive step size
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = x
            
            # Use smaller initial step for Rastrigin (period ~2)
            step = 0.8
            for i in range(self.dim):
                simplex[i + 1] = x.copy()
                if x[i] < 5 - step:
                    simplex[i + 1][i] += step
                else:
                    simplex[i + 1][i] -= step
                simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)

            simplex_f = np.empty(self.dim + 1)
            for i in range(self.dim + 1):
                if time_left() < 5 or call_count[0] >= self.budget:
                    break
                simplex_f[i] = wrapped_func(simplex[i])

            # Check if we managed to evaluate the whole simplex
            if np.any(~np.isfinite(simplex_f)):
                # Fill in missing values
                for i in range(self.dim + 1):
                    if not np.isfinite(simplex_f[i]):
                        simplex_f[i] = wrapped_func(simplex[i])

            # Nelder-Mead iterations with adaptive parameters
            max_iter_nm = remaining_budget // 2
            for _ in range(max_iter_nm):
                if time_left() < 5:
                    break
                if call_count[0] >= self.budget:
                    break

                # Sort simplex
                idx = np.argsort(simplex_f)
                simplex = simplex[idx]
                simplex_f = simplex_f[idx]

                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)

                if simplex_f[0] < f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                elif f_r < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                else:
                    # Contraction
                    beta = 0.5
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + beta * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - beta * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_func(x_c)

                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = wrapped_func(simplex[i])
                            if time_left() < 5 or call_count[0] >= self.budget:
                                break

        # Phase 3: Local grid search around best point to fine-tune
        remaining_budget = self.budget - call_count[0]
        if remaining_budget > 10 and time_left() > 5 and best_x is not None:
            # Fine grid search around best point
            best_local = best_x.copy()
            best_local_f = best_f
            
            for d in range(self.dim):
                if time_left() < 5 or call_count[0] >= self.budget:
                    break
                # Try small perturbations in each dimension
                for delta in [-0.1, -0.05, 0.05, 0.1]:
                    if time_left() < 5 or call_count[0] >= self.budget:
                        break
                    candidate = best_local.copy()
                    candidate[d] += delta
                    candidate = np.clip(candidate, -5, 5)
                    fx = wrapped_func(candidate)
                    if fx < best_local_f:
                        best_local_f = fx
                        best_local = candidate.copy()
            
            if best_local_f < best_f:
                best_x = best_local
                best_f = best_local_f

        return best_x, best_f
