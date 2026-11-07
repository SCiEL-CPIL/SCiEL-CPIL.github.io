import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(start_x, max_iter=200):
            nonlocal eval_count
            if eval_count >= budget - 10:
                return

            x0 = np.clip(start_x, -5, 5).copy()
            f0 = eval_func(x0)
            if eval_count >= budget:
                return

            # Initialize simplex
            simplex = [x0.copy()]
            func_vals = [f0]

            # Create initial simplex with small perturbations
            for i in range(d):
                x_new = x0.copy()
                # Use adaptive initial step size
                if abs(x0[i]) > 0.1:
                    x_new[i] += 0.1 * abs(x0[i])
                else:
                    x_new[i] += 0.1
                x_new = np.clip(x_new, -5, 5)
                f_new = eval_func(x_new)
                if eval_count >= budget:
                    return
                simplex.append(x_new)
                func_vals.append(f_new)

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(func_vals)
                simplex = [simplex[i] for i in order]
                func_vals = [func_vals[i] for i in order]

                x_worst = simplex[-1]
                f_worst = func_vals[-1]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = eval_func(x_ref)
                if eval_count >= budget:
                    break

                if f_ref < func_vals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = eval_func(x_exp)
                    if eval_count >= budget:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        func_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        func_vals[-1] = f_ref
                elif f_ref < func_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    func_vals[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = eval_func(x_con)
                    if eval_count >= budget:
                        break
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        func_vals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_shrink = eval_func(simplex[i])
                            if eval_count >= budget:
                                return
                            func_vals[i] = f_shrink

                # Check convergence
                if max(func_vals) - min(func_vals) < 1e-10:
                    break

        # Phase 1: Multi-start initialization
        rng = np.random.default_rng(42)
        
        # Generate initial candidates: Origin and random points in [-4, 4]
        init_points = [np.zeros(d)]
        for _ in range(9):
            init_points.append(rng.uniform(-4, 4, d))
        
        # Evaluate all initial points
        point_scores = []
        for p in init_points:
            if eval_count >= budget:
                break
            fp = eval_func(p)
            point_scores.append((fp, p))
        
        # Sort by function value
        point_scores.sort(key=lambda x: x[0])
        
        # Phase 2: Local refinement on the best starting points
        # Try to refine the top 2-3 points if budget allows
        top_k = min(3, len(point_scores))
        for i in range(top_k):
            if eval_count >= budget - 30:
                break
            nelder_mead(point_scores[i][1])

        return best_x, best_f
