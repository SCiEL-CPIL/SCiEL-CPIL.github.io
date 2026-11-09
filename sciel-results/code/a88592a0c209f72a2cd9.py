    import numpy as np
    from scipy.optimize import minimize
    
    class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj
        
        def __call__(self, problem, bounds):
            n = self.dim
            m = self.n_obj
            n_g = n - m
            
            # Phase 1: Optimize g-variables (first n_g) to minimize f3 (which is 2.25 + g when last m are 0.5)
            # Fix last m variables to 0.5
            x_fixed_tail = np.full(m, 0.5)
            
            def objective_g(x_g):
                x = np.concatenate([x_g, x_fixed_tail])
                f = problem(x)
                return f[2] # Minimize f3 to minimize g
            
            # Initial guess for x_g
            x0_g = np.full(n_g, 0.5)
            
            # Run L-BFGS-B on n_g dimensions
            # Budget allocation: 100 evals for this phase
            # maxiter needs to be small
            # 19D * 10 iter = 190 evals. Too much.
            # Use maxiter=3? 19*3 = 57 evals.
            res = minimize(objective_g, x0_g, method='L-BFGS-B', bounds=[(0,1) for _ in range(n_g)], options={'maxiter': 5})
            x_g_opt = res.x
            x_g_opt = np.clip(x_g_opt, 0, 1)
            
            # Phase 2: Sample Pareto front
            # Use remaining budget
            used_evals = res.nfev
            remaining = self.budget - used_evals
            
            # Generate LHS points for last m vars
            # Use scipy.stats.qmc.LatinHypercube
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=m, seed=42)
            n_samples = remaining
            # Ensure we don't exceed budget
            # Generate n_samples points
            x_p_samples = sampler.random(n=n_samples)
            
            # Evaluate all
            results = []
            for x_p in x_p_samples:
                x = np.concatenate([x_g_opt, x_p])
                f = problem(x)
                results.append(f)
            
            return np.array(results)
    