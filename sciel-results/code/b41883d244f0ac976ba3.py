# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead Simplex Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            eval_count += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Strategy: Multiple restarts.
        # We divide the budget among restarts.
        # Let's try 5 restarts.
        num_restarts = 5
        
        for r in range(num_restarts):
            if eval_count >= budget:
                break
            
            # Determine starting point
            if r == 0:
                # First start: random in the box, maybe slightly centered
                x0 = rng.uniform(-4, 4, n)
                step_size = 1.0
            else:
                # Subsequent starts: 
                # Option 1: Random
                # Option 2: Perturb best
                # Let's mix: 50% random, 50% perturb best
                if rng.random() < 0.5 and best_x is not None:
                    # Perturb best with a large step to escape local minima
                    # Use a smaller step size for the simplex if we are close to best
                    x0 = best_x + rng.normal(0, 1.0, n)
                    x0 = np.clip(x0, -5, 5)
                    step_size = 0.5
                else:
                    x0 = rng.uniform(-5, 5, n)
                    step_size = 1.0
            
            # Initialize Simplex
            # x0 is the first vertex
            # Other vertices: x0 + step_size * e_i
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += step_size
                simplex.append(xi)
            
            # Evaluate initial simplex
            fvals = []
            for x in simplex:
                if eval_count >= budget:
                    break
                fvals.append(evaluate(x))
            
            if len(fvals) < n + 1:
                # Not enough budget to complete even one simplex
                break

            # Sort simplex by function value
            # We store as list of (x, f)
            simp = list(zip(simplex, fvals))
            simp.sort(key=lambda item: item[1])
            
            # Nelder-Mead Loop
            # Stop if simplex is small or no improvement for a while
            no_improve_count = 0
            max_nm_iterations = 100 # Cap per restart to ensure we restart
            
            for k in range(max_nm_iterations):
                if eval_count >= budget:
                    break
                
                # Current best, worst, second worst
                x_best_local, f_best_local = simp[0]
                x_worst, f_worst = simp[-1]
                x_second_worst, f_second_worst = simp[-2]
                
                # Centroid of all points except worst
                centroid = np.sum([s[0] for s in simp[:-1]], axis=0) / n
                
                # Reflection
                x_ref = centroid + alpha * (centroid - x_worst)
                f_ref = evaluate(x_ref)
                
                if f_ref < f_best_local:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - x_worst)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        # Replace worst with expanded
                        simp[-1] = (x_exp, f_exp)
                    else:
                        # Replace worst with reflected
                        simp[-1] = (x_ref, f_ref)
                    no_improve_count = 0
                elif f_ref < f_second_worst:
                    # Replace worst with reflected
                    simp[-1] = (x_ref, f_ref)
                    no_improve_count = 0
                else:
                    # Contraction
                    # Outside contraction if f_ref < f_worst, else inside
                    if f_ref < f_worst:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_worst - centroid)
                    
                    f_con = evaluate(x_con)
                    
                    if f_con < min(f_ref, f_worst):
                        simp[-1] = (x_con, f_con)
                        no_improve_count = 0
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            x_shrink = x_best_local + sigma * (simp[i][0] - x_best_local)
                            f_shrink = evaluate(x_shrink)
                            simp[i] = (x_shrink, f_shrink)
                        # Re-sort
                        simp.sort(key=lambda item: item[1])
                        no_improve_count += 1
                        if no_improve_count > 5:
                            break

                # Check convergence: simplex size
                x_min = np.min([s[0] for s in simp], axis=0)
                x_max = np.max([s[0] for s in simp], axis=0)
                if np.max(np.abs(x_max - x_min)) < 1e-4:
                    break

                # Re-sort after update
                simp.sort(key=lambda item: item[1])

            # End of Nelder-Mead for this restart
            # Update global best is handled in evaluate()

        return best_x, best_f
# EVOLVE-BLOCK-END
