"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = np.inf
        eval_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            if eval_count >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start with CMA-ES-like exploration
        # Use a few diverse starting points and refine each with small CMA-ES
        n_starts = 5
        cma_per_start = min(40, budget // (n_starts + 2))
        lambda_cma = 10
        
        for start_idx in range(n_starts):
            if eval_count >= budget or not time_ok():
                break
            
            # Diverse initial point
            if start_idx == 0:
                x_mean = np.zeros(n)
                sigma = 2.0
            elif start_idx == 1:
                x_mean = np.array([3.0, -3.0, 3.0, -3.0, 3.0])
                sigma = 1.5
            elif start_idx == 2:
                x_mean = np.array([-3.0, 3.0, -3.0, 3.0, -3.0])
                sigma = 1.5
            elif start_idx == 3:
                x_mean = rng.uniform(-4, 4, n)
                sigma = 1.0
            else:
                x_mean = rng.uniform(-4, 4, n)
                sigma = 1.0
            
            # Small CMA-ES around this point
            for gen in range(cma_per_start // lambda_cma + 1):
                if eval_count >= budget or not time_ok():
                    break
                
                offspring = []
                for i in range(lambda_cma):
                    z = rng.standard_normal(n)
                    x_off = x_mean + sigma * z
                    x_off = np.clip(x_off, -5, 5)
                    offspring.append(x_off)
                
                f_offspring = []
                for x_off in offspring:
                    if eval_count >= budget or not time_ok():
                        break
                    fx = evaluate(x_off)
                    if fx is None:
                        break
                    f_offspring.append(fx)
                
                if len(f_offspring) < lambda_cma:
                    break
                
                order = np.argsort(f_offspring)
                sorted_offspring = [offspring[i] for i in order]
                
                # Update mean using top half
                mu_cma = lambda_cma // 2
                weights_cma = np.log(mu_cma + 0.5) - np.arange(mu_cma)
                weights_cma /= weights_cma.sum()
                
                new_mean = np.zeros(n)
                for i in range(mu_cma):
                    new_mean += weights_cma[i] * sorted_offspring[i]
                x_mean = new_mean
                
                # Adapt sigma
                f_spread = np.std(f_offspring[:mu_cma])
                if f_spread < 1e-6:
                    sigma *= 0.5
                else:
                    sigma = max(0.1, sigma * 0.95)
                
                if sigma < 0.05:
                    break
        
        # Phase 2: Nelder-Mead on best candidate
        if best_x is not None and eval_count < budget and time_ok():
            x0 = best_x.copy()
            f0 = best_f
            
            # Adaptive simplex size
            simplex_delta = 0.3
            
            simplex = [x0.copy()]
            for j in range(n):
                point = x0.copy()
                delta = simplex_delta if x0[j] + simplex_delta <= 5 else -simplex_delta
                point[j] += delta
                point = np.clip(point, -5, 5)
                simplex.append(point)

            fvals = [f0]
            for pt in simplex[1:]:
                if eval_count >= budget or not time_ok():
                    break
                fx = evaluate(pt)
                if fx is None:
                    break
                fvals.append(fx)

            if len(fvals) == len(simplex):
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_shrink = 0.5
                
                max_nm_iter = budget - eval_count
                for _ in range(max(0, max_nm_iter)):
                    if eval_count >= budget or not time_ok():
                        break

                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]

                    if abs(fvals[-1] - fvals[0]) < 1e-12:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                    if eval_count >= budget or not time_ok():
                        break
                    fr = evaluate(xr)
                    if fr is None:
                        break

                    if fr < fvals[0]:
                        xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                        if eval_count >= budget or not time_ok():
                            break
                        fe = evaluate(xe)
                        if fe is None:
                            simplex[-1], fvals[-1] = xr, fr
                            break
                        if fe < fr:
                            simplex[-1], fvals[-1] = xe, fe
                        else:
                            simplex[-1], fvals[-1] = xr, fr
                    elif fr < fvals[-2]:
                        simplex[-1], fvals[-1] = xr, fr
                    else:
                        if fr < fvals[-1]:
                            xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                        else:
                            xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                        if eval_count >= budget or not time_ok():
                            break
                        fc = evaluate(xc)
                        if fc is None:
                            break
                        if fc < min(fr, fvals[-1]):
                            simplex[-1], fvals[-1] = xc, fc
                        else:
                            for i in range(1, len(simplex)):
                                simplex[i] = np.clip(simplex[0] + sigma_shrink * (simplex[i] - simplex[0]), -5, 5)
                                if eval_count >= budget or not time_ok():
                                    break
                            if eval_count >= budget or not time_ok():
                                break
                            fvals_new = []
                            for pt in simplex:
                                if pt is simplex[0]:
                                    fvals_new.append(fvals[0])
                                else:
                                    fx = evaluate(pt)
                                    if fx is None:
                                        break
                                    fvals_new.append(fx)
                            if len(fvals_new) != len(simplex):
                                break
                            fvals = fvals_new

        # Phase 3: Final polish with adaptive random walk + NM restart
        if best_x is not None and eval_count < budget and time_ok():
            x = best_x.copy()
            f = best_f
            sigma = 0.1
            min_sigma = 1e-7
            stall_count = 0
            for _ in range(budget - eval_count):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = evaluate(y)
                if fy is None:
                    break
                if fy < f:
                    x, f = y, fy
                    sigma = min(sigma * 1.15, 0.5)
                    stall_count = 0
                else:
                    sigma *= 0.85
                    stall_count += 1
                    if sigma < min_sigma or stall_count > 30:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
