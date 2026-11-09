# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Nelder-Mead Simplex Optimizer with Adaptive Restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = None
        best_f = np.inf

        def count_call(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # Phase 1: Random Sampling to find a good starting point
        # Use 10% of budget for initialization
        init_budget = max(10, self.budget // 10)
        
        # Start with center
        x0 = np.zeros(self.dim)
        f0 = count_call(x0)
        best_x = x0.copy()
        best_f = f0

        # Sample random points
        for _ in range(init_budget - 1):
            if calls_made >= self.budget:
                break
            xi = rng.uniform(-5, 5, self.dim)
            fi = count_call(xi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # Phase 2: Nelder-Mead with Restarts
        # We will run Nelder-Mead until budget is exhausted or stagnation occurs.
        # Stagnation triggers a restart with a new center.
        
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        max_iters_per_run = 50
        stagnation_limit = 10
        epsilon = 1e-8
        
        while calls_made < self.budget:
            # Initialize Simplex around best_x
            # Initial step size
            step_size = 0.5
            
            # Create simplex
            simplex = []
            f_vals = []
            
            # Vertex 0 is the best point
            simplex.append(best_x.copy())
            f_vals.append(best_f)
            
            # Other vertices
            for i in range(self.dim):
                v = best_x.copy()
                v[i] += step_size
                v = np.clip(v, -5, 5)
                fv = count_call(v)
                simplex.append(v)
                f_vals.append(fv)
                
                if fv < best_f:
                    best_f = fv
                    best_x = v.copy()

            # Sort simplex by fitness
            # We need to maintain the order for NM steps
            # Helper to sort and update best
            def sort_simplex():
                nonlocal best_f, best_x
                indices = np.argsort(f_vals)
                simplex = [simplex[i] for i in indices]
                f_vals = [f_vals[i] for i in indices]
                if f_vals[0] < best_f:
                    best_f = f_vals[0]
                    best_x = simplex[0].copy()
            
            sort_simplex()
            
            stagnant_count = 0
            prev_best = best_f
            
            for _ in range(max_iters_per_run):
                if calls_made >= self.budget:
                    break
                
                # Get worst and second worst
                worst_idx = self.dim
                second_worst_idx = self.dim - 1
                
                # Calculate centroid of all but worst
                centroid = np.mean(simplex[:self.dim], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[worst_idx])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = count_call(x_ref)
                
                if f_ref < f_vals[second_worst_idx]:
                    # Check for expansion
                    x_exp = centroid + gamma * (centroid - simplex[worst_idx])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = count_call(x_exp)
                    
                    if f_exp < f_ref:
                        simplex[worst_idx] = x_exp
                        f_vals[worst_idx] = f_exp
                    else:
                        simplex[worst_idx] = x_ref
                        f_vals[worst_idx] = f_ref
                elif f_ref < f_vals[worst_idx]:
                    # Contraction
                    x_con = centroid + rho * (simplex[worst_idx] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = count_call(x_con)
                    
                    if f_con < f_ref:
                        simplex[worst_idx] = x_con
                        f_vals[worst_idx] = f_con
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_vals[i] = count_call(simplex[i])
                else:
                    # Shrink
                    for i in range(1, self.dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_vals[i] = count_call(simplex[i])
                
                sort_simplex()
                
                # Check for stagnation
                if best_f > prev_best - epsilon:
                    stagnant_count += 1
                else:
                    stagnant_count = 0
                    prev_best = best_f
                
                if stagnant_count >= stagnation_limit:
                    break

            # If we exited due to stagnation and still have budget, restart
            if calls_made < self.budget:
                # Restart with a new random point or perturbed best
                # To escape local minima, jump to a new random point with high probability
                # or perturb the best point
                if rng.random() < 0.5:
                    new_center = rng.uniform(-5, 5, self.dim)
                else:
                    new_center = best_x + rng.normal(0, 1.0, self.dim)
                    new_center = np.clip(new_center, -5, 5)
                
                f_new = count_call(new_center)
                if f_new < best_f:
                    best_f = f_new
                    best_x = new_center.copy()
                else:
                    # Keep old best as center for next NM run?
                    # Or force center to be the new point?
                    # If new point is worse, maybe we are in a local min.
                    # Let's center the next NM on the best point found so far,
                    # but with a larger step size to escape.
                    pass # best_x remains the global best

        return best_x, best_f
# EVOLVE-BLOCK-END
