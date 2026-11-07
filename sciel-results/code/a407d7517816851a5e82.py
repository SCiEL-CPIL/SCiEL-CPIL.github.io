"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    Optimizer using Multistart Nelder-Mead.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # RNG for reproducibility within a run, but different instances will have different funcs
        # Using a fixed seed here might be bad if instances are correlated, but standard practice
        # for these tasks is often deterministic or random. Let's use random to explore different
        # regions if the function is multimodal (though Griewank-Rosenbrock is unimodal-ish with flat regions).
        # However, the task says "hidden seeded shift", implying the landscape shape is fixed but shifted.
        # A single good run of Nelder-Mead from a random start might suffice, but multistart is safer.
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Parameters for Nelder-Mead
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        initial_step = 1.0
        
        # Max evaluations per simplex run before restarting
        # With dim=5, simplex size is 6.
        # Let's allow ~40-50 evals per restart to encourage exploration.
        max_evals_per_run = 50
        
        # Total budget is 500.
        # We will run multiple simplexes.
        
        while calls_made < budget:
            # Initialize simplex
            # Start point
            center = np.random.uniform(-4, 4, dim) # Minimizer is in [-4, 4]^5
            
            # Create simplex
            simplex = [center.copy() for _ in range(dim + 1)]
            # Perturb each vertex
            for i in range(dim + 1):
                if i == 0:
                    continue
                # Perturb dimension i-1
                simplex[i][i-1] += initial_step
                # Also add a small random perturbation to all dims to avoid symmetry issues
                simplex[i] += np.random.uniform(-0.1, 0.1, dim)
                simplex[i] = np.clip(simplex[i], -5, 5)
            
            # Evaluate initial simplex
            fvals = []
            for i in range(dim + 1):
                if calls_made >= budget:
                    break
                f = func(simplex[i])
                fvals.append(f)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = simplex[i].copy()
            
            if calls_made >= budget:
                break
                
            # Sort simplex by function value
            # We need to keep simplex and fvals sorted
            sorted_indices = np.argsort(fvals)
            simplex = [simplex[i] for i in sorted_indices]
            fvals = [fvals[i] for i in sorted_indices]
            
            # Local best for this run
            local_best_f = fvals[0]
            local_best_x = simplex[0].copy()
            
            # Nelder-Mead iterations
            no_improvement_count = 0
            run_evals = 0
            
            while calls_made < budget and run_evals < max_evals_per_run:
                # Check for convergence or stall
                # If the range of fvals is very small, we might be stuck
                if len(fvals) > 1 and (fvals[-1] - fvals[0]) < 1e-10:
                    break
                    
                # Identify worst and second worst
                worst_idx = dim # Last element is worst
                second_worst_idx = dim - 1
                
                # Calculate centroid of all but worst
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[worst_idx])
                x_ref = np.clip(x_ref, -5, 5)
                
                if calls_made >= budget:
                    break
                
                f_ref = func(x_ref)
                calls_made += 1
                run_evals += 1
                
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
                    
                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    
                    if calls_made >= budget:
                        break
                        
                    f_exp = func(x_exp)
                    calls_made += 1
                    run_evals += 1
                    
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                        
                    if f_exp < f_ref:
                        # Replace worst with expanded point
                        simplex[worst_idx] = x_exp
                        fvals[worst_idx] = f_exp
                    else:
                        # Replace worst with reflected point
                        simplex[worst_idx] = x_ref
                        fvals[worst_idx] = f_ref
                        
                elif f_ref < fvals[second_worst_idx]:
                    # Replace worst with reflected point
                    simplex[worst_idx] = x_ref
                    fvals[worst_idx] = f_ref
                else:
                    # Contraction
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    
                    if calls_made >= budget:
                        break
                        
                    f_con = func(x_con)
                    calls_made += 1
                    run_evals += 1
                    
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                        
                    if f_con < f_ref:
                        # Replace worst with contracted point
                        simplex[worst_idx] = x_con
                        fvals[worst_idx] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            
                            if calls_made >= budget:
                                break
                                
                            f_shr = func(simplex[i])
                            calls_made += 1
                            run_evals += 1
                            
                            if f_shr < best_f:
                                best_f = f_shr
                                best_x = simplex[i].copy()
                                
                            fvals[i] = f_shr
                        
                        if calls_made >= budget:
                            break

                # Re-sort simplex
                sorted_indices = np.argsort(fvals)
                simplex = [simplex[i] for i in sorted_indices]
                fvals = [fvals[i] for i in sorted_indices]
                
                # Check if we made progress
                if fvals[0] < local_best_f - 1e-8:
                    local_best_f = fvals[0]
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    
                # If stuck for too many steps, break to restart
                if no_improvement_count > 10:
                    break
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
