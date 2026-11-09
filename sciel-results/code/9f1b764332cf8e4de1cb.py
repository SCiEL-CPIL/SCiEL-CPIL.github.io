import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        eval_count = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global exploration with (1+1)-ES using 1/5 success rule
        x = np.zeros(d)
        fx = evaluate(x)
        sigma = 2.0
        success_count = 0
        window_size = 10
        target_success_rate = 0.2

        # Run (1+1)-ES for the first 50% of the budget
        phase1_budget = budget // 2
        for i in range(phase1_budget - 1):
            if time.time() > deadline or eval_count >= budget - 10:
                break

            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = evaluate(y)

            if fy < fx:
                x, fx = y, fy
                success_count += 1
            else:
                pass

            # Adapt sigma every window_size steps
            if (i + 1) % window_size == 0:
                success_rate = success_count / window_size
                if success_rate > target_success_rate:
                    sigma *= 1.25
                elif success_rate < target_success_rate:
                    sigma *= 0.8
                success_count = 0

            if eval_count >= budget:
                break

        # Phase 2: Local refinement with Nelder-Mead
        # Initialize simplex around the best point found by (1+1)-ES
        center = best_x.copy()
        simplex = [(center.copy(), best_f)]

        # Add vertices with a step size appropriate for NM
        step = 0.1

        for i in range(d):
            if time.time() > deadline or eval_count >= budget:
                break
            pt = center.copy()
            pt[i] += step
            fval = evaluate(pt)
            simplex.append((pt.copy(), fval))

        # Sort simplex
        simplex.sort(key=lambda p: p[1])

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5

        # Run NM for the remaining budget
        nm_iterations = 0
        max_nm_iter = 200

        while nm_iterations < max_nm_iter:
            if time.time() > deadline or eval_count >= budget:
                break

            nm_iterations += 1

            # Get best, worst, second worst
            x_worst, f_worst = simplex[-1]
            x_second, f_second = simplex[-2]

            # Centroid of all but worst
            x_centroid = np.mean([p[0] for p in simplex[:-1]], axis=0)

            # Reflection
            if eval_count + 1 > budget:
                break
            x_ref = x_centroid + alpha * (x_centroid - x_worst)
            f_ref = evaluate(x_ref)

            if f_ref < simplex[0][1]:  # Better than best
                # Expansion
                if eval_count + 1 > budget:
                    break
                x_exp = x_centroid + gamma * (x_centroid - x_worst)
                f_exp = evaluate(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = (x_exp, f_exp)
                else:
                    simplex[-1] = (x_ref, f_ref)
            elif f_ref < f_second:  # Between best and second worst
                simplex[-1] = (x_ref, f_ref)
            else:  # Worse than second worst
                # Contraction
                if eval_count + 1 > budget:
                    break
                if f_ref < f_worst:
                    # Outer contraction
                    x_con = x_centroid + 0.5 * (x_ref - x_centroid)
                else:
                    # Inner contraction
                    x_con = x_centroid - 0.5 * (x_worst - x_centroid)

                f_con = evaluate(x_con)

                if f_con < min(f_ref, f_worst):
                    simplex[-1] = (x_con, f_con)
                else:
                    # Shrink
                    # Shrink requires d evaluations. Check if we have budget.
                    if eval_count + d > budget:
                        break
                    
                    x_best, f_best_cur = simplex[0]
                    new_simplex = [(x_best, f_best_cur)]
                    shrink_completed = True
                    for i, (x_i, f_i) in enumerate(simplex):
                        if i == 0:
                            continue
                        if eval_count >= budget:
                            shrink_completed = False
                            break
                        x_shrink = x_best + rho * (x_i - x_best)
                        f_shrink = evaluate(x_shrink)
                        new_simplex.append((x_shrink, f_shrink))
                    
                    if shrink_completed:
                        simplex = new_simplex
                    else:
                        break

            # Re-sort
            simplex.sort(key=lambda p: p[1])

            # Convergence check: if simplex is very small, break
            x_min = np.min([p[0] for p in simplex], axis=0)
            x_max = np.max([p[0] for p in simplex], axis=0)
            if np.max(x_max - x_min) < 1e-8:
                break

        return best_x, best_f
