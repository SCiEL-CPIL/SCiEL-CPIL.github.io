import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = float(fx)
                best_x = x.copy()
            return float(fx)

        def time_ok():
            return time.time() < deadline

        # ---- Phase 1: Diverse early sampling for fast initial progress ----
        # We want to cover the space well to find the basin of attraction quickly.
        samples = []
        # Broad random
        for _ in range(10):
            samples.append(rng.uniform(-5.0, 5.0, d))
        # Medium random
        for _ in range(10):
            samples.append(rng.uniform(-3.0, 3.0, d))
        # Local random
        for _ in range(8):
            samples.append(rng.uniform(-1.5, 1.5, d))
        # Very local
        for _ in range(4):
            samples.append(rng.uniform(-0.5, 0.5, d))
        # Structured points
        samples.append(np.zeros(d))
        samples.append(np.ones(d) * 0.5)
        samples.append(np.ones(d) * -0.5)
        # Gaussian
        samples.append(rng.normal(0, 2.0, d))
        samples.append(rng.normal(0, 1.0, d))
        samples.append(rng.normal(0, 0.5, d))
        
        # Limit initial samples to ensure we have budget left for optimization
        n_initial = min(len(samples), 30)
        for x in samples[:n_initial]:
            if eval_count >= 30 or not time_ok():
                break
            evaluate(x)

        # ---- Phase 2: Nelder-Mead Simplex Method ----
        def nelder_mead(x0, max_iter=150, init_step=1.0):
            nonlocal eval_count, best_x, best_f
            if not time_ok() or eval_count >= budget - 20:
                return x0

            # Build simplex around x0
            simplex = [x0.copy()]
            for i in range(d):
                v = x0.copy()
                # Ensure initial step is not too large to jump out of bounds immediately
                # but large enough to explore
                v[i] += init_step
                simplex.append(v)
            
            f_values = []
            for v in simplex:
                if not time_ok() or eval_count >= budget - 10:
                    return x0
                f_values.append(evaluate(v))

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for _ in range(max_iter):
                if not time_ok() or eval_count >= budget - 10:
                    break

                # Sort simplex by function values
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Convergence check
                if f_values[-1] - f_values[0] < 1e-12 * (1 + abs(f_values[0])):
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = f_values[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                if not time_ok() or eval_count >= budget - 10:
                    break
                f_r = evaluate(x_r)

                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    if not time_ok() or eval_count >= budget - 10:
                        break
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        if not time_ok() or eval_count >= budget - 10:
                            break
                        f_c = evaluate(x_c)
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            f_values[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                if not time_ok() or eval_count >= budget - 10:
                                    return simplex[0]
                                f_values[i] = evaluate(simplex[i])
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - worst)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        if not time_ok() or eval_count >= budget - 10:
                            break
                        f_c = evaluate(x_c)
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            f_values[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                if not time_ok() or eval_count >= budget - 10:
                                    return simplex[0]
                                f_values[i] = evaluate(simplex[i])

            # Return the best vertex
            idx = int(np.argmin(f_values))
            return simplex[idx]

        # Run Nelder-Mead from the current best
        if eval_count < budget - 50 and time_ok():
            x_nm1 = nelder_mead(best_x.copy(), max_iter=100, init_step=1.0)
            if x_nm1 is not None and np.all(np.isfinite(x_nm1)):
                best_x = np.clip(x_nm1, -5.0, 5.0)
                # Ensure we have the best value recorded
                if eval_count < budget - 5:
                    evaluate(best_x)

        # ---- Phase 3: Coordinate-wise refinement with adaptive steps ----
        # This helps if NM stopped slightly early or to polish the result.
        if eval_count < budget - 20 and time_ok():
            for i in range(d):
                if eval_count >= budget - 10 or not time_ok():
                    break
                # Try decreasing step sizes
                for step_size in [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                    if eval_count >= budget - 10 or not time_ok():
                        break
                    for direction in [1.0, -1.0]:
                        if eval_count >= budget - 10 or not time_ok():
                            break
                        test_x = best_x.copy()
                        test_x[i] = np.clip(test_x[i] + direction * step_size, -5.0, 5.0)
                        fx = evaluate(test_x)
                        if fx < best_f:
                            best_f = fx
                            best_x = test_x.copy()
                            # If we improved, we might want to try smaller steps immediately
                            # but the loop structure handles this by trying the next direction/size.
                            # A simple greedy approach: if we moved, re-evaluate the same direction with smaller step?
                            # The current loop just moves to the next step size/direction. 
                            # To be more aggressive, we could restart the step_size loop if improvement is found.
                            # However, to keep it simple and safe, we just continue.

        # ---- Phase 4: Second Nelder-Mead pass with smaller steps if budget allows ----
        if eval_count < budget - 20 and time_ok():
            x_nm2 = nelder_mead(best_x.copy(), max_iter=50, init_step=0.1)
            if x_nm2 is not None and np.all(np.isfinite(x_nm2)):
                best_x = np.clip(x_nm2, -5.0, 5.0)
                if eval_count < budget - 5:
                    evaluate(best_x)

        # ---- Phase 5: Final ultra-fine coordinate scan ----
        if eval_count < budget - 5 and time_ok():
            for i in range(d):
                if eval_count >= budget - 2 or not time_ok():
                    break
                for step_size in [0.001, 0.0005, 0.0002, 0.0001]:
                    if eval_count >= budget - 2 or not time_ok():
                        break
                    for direction in [1.0, -1.0]:
                        if eval_count >= budget - 2 or not time_ok():
                            break
                        test_x = best_x.copy()
                        test_x[i] = np.clip(test_x[i] + direction * step_size, -5.0, 5.0)
                        fx = evaluate(test_x)
                        if fx < best_f:
                            best_f = fx
                            best_x = test_x.copy()

        return best_x, best_f
