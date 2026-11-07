"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with random restarts for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_iter_per_restart = 100  # Limit iterations per simplex run to allow restarts

        # Initial point
        x0 = np.random.uniform(-4.0, 4.0, dim)
        
        while calls_made < budget and time.time() < deadline:
            # Initialize simplex
            simplex = [x0.copy()]
            f_vals = [evaluate(x0)]
            
            # Create initial simplex around x0
            for i in range(dim):
                x_new = x0.copy()
                # Step size for initial simplex
                if x_new[i] == 0:
                    x_new[i] += 0.5
                else:
                    x_new[i] += 0.5 * x_new[i]
                simplex.append(x_new)
                f_vals.append(evaluate(x_new))
                
                if calls_made >= budget or time.time() > deadline:
                    break

            if calls_made >= budget or time.time() > deadline:
                break

            # Nelder-Mead iterations
            for _ in range(max_iter_per_restart):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                x_best = simplex[0]
                f_best = f_vals[0]
                x_worst = simplex[-1]
                f_worst = f_vals[-1]
                
                # Check convergence
                if abs(f_worst - f_best) < 1e-8:
                    break

                # Centroid of all points except worst
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = evaluate(x_ref)
                
                if calls_made >= budget or time.time() > deadline:
                    break

                if f_ref < f_best:
                    # Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = evaluate(x_exp)
                    
                    if calls_made >= budget or time.time() > deadline:
                        break

                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                elif f_ref < f_worst:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_worst:
                        # Outside contraction
                        x_con = x_centroid + 0.5 * (x_ref - x_centroid)
                    else:
                        # Inside contraction
                        x_con = x_centroid - 0.5 * (x_worst - x_centroid)
                    
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = evaluate(x_con)
                    
                    if calls_made >= budget or time.time() > deadline:
                        break

                    if f_con < min(f_ref, f_worst):
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_vals[i] = evaluate(simplex[i])
                            
                            if calls_made >= budget or time.time() > deadline:
                                break
                        
                        if calls_made >= budget or time.time() > deadline:
                            break

            # If we finished a simplex run without running out of budget,
            # consider a restart from a new random point or perturbed best
            if calls_made < budget and time.time() < deadline:
                # 30% chance to restart from completely new random point
                # Otherwise, perturb the current best
                if np.random.random() < 0.3:
                    x0 = np.random.uniform(-4.0, 4.0, dim)
                else:
                    x0 = best_x + np.random.normal(0, 0.5, dim)
                    x0 = np.clip(x0, -4.0, 4.0)

        if best_x is None:
            # Fallback if no evaluations were made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
