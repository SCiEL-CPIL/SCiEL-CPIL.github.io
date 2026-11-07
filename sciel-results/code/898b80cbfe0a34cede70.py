import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        D = self.dim
        budget = self.budget
        calls_made = 0
        
        # Nelder-Mead Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial Simplex
        # Start at 0 with a step size of 2.0 to cover a good portion of [-5, 5]^5
        step_size = 2.0
        x0 = np.zeros(D)
        f0 = func(x0)
        calls_made += 1
        
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        # Create other points in the simplex
        for i in range(D):
            xi = x0.copy()
            xi[i] = step_size
            fi = func(xi)
            calls_made += 1
            simplex.append(xi)
            simplex_f.append(fi)
        
        best_x = x0.copy()
        best_f = f0
        
        # Track progress for potential restarts
        last_best_f = best_f
        stagnant_iterations = 0
        max_stagnant = 10
        
        while calls_made < budget:
            # Sort simplex by fitness
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()
                stagnant_iterations = 0
            else:
                stagnant_iterations += 1
            
            # Check for convergence or stagnation
            # If simplex is too small or stagnant, restart
            simplex_size = np.linalg.norm(simplex[0] - simplex[-1])
            if simplex_size < 1e-8 or (stagnant_iterations > max_stagnant and calls_made > 50):
                # Restart around best_x with a larger step to explore further
                # Use a smaller step if we were stuck in a local min? 
                # For Schaffers, ill-conditioned, a larger step might help escape.
                # Let's try a step size of 1.0 for restart.
                step_size = 1.0
                x0 = best_x.copy()
                f0 = best_f # We already have this value
                calls_made += 0 # No new call for best_x, but we need to build new simplex
                
                simplex = [x0.copy()]
                simplex_f = [f0]
                
                for i in range(D):
                    xi = x0.copy()
                    xi[i] = step_size
                    # Check budget before calling
                    if calls_made >= budget:
                        break
                    fi = func(xi)
                    calls_made += 1
                    simplex.append(xi)
                    simplex_f.append(fi)
                
                stagnant_iterations = 0
                if calls_made >= budget:
                    break
            
            # Get centroid of all points except the worst
            centroid = np.sum(simplex[:-1], axis=0) / D
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            calls_made += 1
            
            if fr < simplex_f[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(xe)
                calls_made += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                # Outside contraction if reflection is worse than worst
                if fr < simplex_f[-1]:
                    xc = centroid + rho * (xr - centroid)
                    fc = func(xc)
                    calls_made += 1
                    
                    if fc <= fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = func(simplex[i])
                            calls_made += 1
                            if calls_made >= budget:
                                break
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                    fc = func(xc)
                    calls_made += 1
                    
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = func(simplex[i])
                            calls_made += 1
                            if calls_made >= budget:
                                break
                        if calls_made >= budget:
                            break
            
            if calls_made >= budget:
                break

        return best_x, best_f
