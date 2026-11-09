"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # 1. Global Coarse Grid Initialization
        # Use a grid to ensure good initial coverage of the domain [-5, 5]^5
        # For dim=5, 3^5 = 243 points. This is a reasonable fraction of 300.
        # We evaluate these first to find the best starting point.
        
        # Let's use a slightly coarser grid to save evaluations for local search
        # 3 points per dimension: -5, 0, 5. Total 3^5 = 243.
        # This consumes 243/300 = 81% of the budget. 
        # This might be too much if local search needs more.
        # Let's use 2^5 = 32 points: corners of the box.
        # Or maybe a mix. 
        # Actually, for BBOB functions like Sphere/Rastrigin, the minimum is often near the center or shifted.
        # A grid of 3 points per dim (243 evals) leaves only 57 for local search.
        # A grid of 2 points per dim (32 evals) leaves 268 for local search.
        # Let's try 32 initial points (corners) + center. Total 33.
        
        points = []
        # Corners
        for i in range(2**dim):
            p = np.array([5.0 if (i >> j) & 1 else -5.0 for j in range(dim)])
            points.append(p)
        # Center
        points.append(np.zeros(dim))
        
        # Shuffle points to randomize the order of local searches? 
        # No, let's just evaluate them all first to find the global best among them.
        
        evals = 0
        best_x = None
        best_f = float('inf')
        
        # Evaluate initial points
        for p in points:
            if time.time() > deadline:
                break
            fp = func(p)
            evals += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
        
        # 2. Local Refinement with Nelder-Mead
        # We will run Nelder-Mead starting from the best point found.
        # We can also run it from the top 2-3 points to hedge bets, but budget is tight.
        # Let's focus on the best one.
        
        # Nelder-Mead implementation
        def nelder_mead(func, x0, budget_remaining, dim, deadline):
            # Initialize simplex
            # x0 is the best point
            eps = 0.5  # Initial step size
            simplex = [x0.copy()]
            f0 = func(x0)
            evals_local = 1
            best_f_local = f0
            best_x_local = x0.copy()
            
            # Generate other vertices
            for i in range(dim):
                xi = x0.copy()
                # Choose step size based on distance to boundary
                if xi[i] >= 0:
                    xi[i] -= eps
                else:
                    xi[i] += eps
                # Clip to bounds
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                fi = func(xi)
                evals_local += 1
                if fi < best_f_local:
                    best_f_local = fi
                    best_x_local = xi.copy()
            
            # Sort simplex by function value
            simplex.sort(key=lambda x: func(x) if False else None) # We need to store f values
            # Let's use a list of tuples (f, x)
            simp = [(f0, x0.copy())]
            for i in range(1, dim+1):
                xi = simplex[i]
                fi = func(xi) if False else simp[i][0] # Wait, I already evaluated them above?
                # The code above evaluated them but didn't store in simp properly.
                # Let's restructure.
            
            # Restart NM logic properly
            simp = []
            x0 = best_x_local
            f0 = best_f_local
            simp.append((f0, x0.copy()))
            
            for i in range(dim):
                xi = x0.copy()
                if xi[i] >= 0:
                    xi[i] -= eps
                else:
                    xi[i] += eps
                xi = np.clip(xi, -5, 5)
                # If xi is same as x0 due to clipping/eps, adjust
                if np.allclose(xi, x0):
                     xi[i] += eps if xi[i] < 5 else -eps
                fi = func(xi)
                evals_local += 1
                simp.append((fi, xi.copy()))
                if fi < best_f_local:
                    best_f_local = fi
                    best_x_local = xi.copy()
            
            # Main loop
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            while evals_local < budget_remaining and time.time() < deadline:
                # Sort simplex
                simp.sort(key=lambda t: t[0])
                
                # Update global best
                if simp[0][0] < best_f_local:
                    best_f_local = simp[0][0]
                    best_x_local = simp[0][1].copy()
                
                # Check convergence (optional, can just run out of budget)
                # If range of f values is small, stop
                if simp[-1][0] - simp[0][0] < 1e-6:
                    break
                    
                x_worst = simp[-1][1]
                f_worst = simp[-1][0]
                
                # Calculate centroid of all but worst
                x_centroid = np.mean([s[1] for s in simp[:-1]], axis=0)
                
                # Reflection
                x_r = x_centroid + alpha * (x_centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                evals_local += 1
                
                if f_r < simp[0][0]:
                    # Expansion
                    x_e = x_centroid + gamma * (x_centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    evals_local += 1
                    if f_e < f_r:
                        simp[-1] = (f_e, x_e)
                    else:
                        simp[-1] = (f_r, x_r)
                elif f_r < simp[-2][0]:
                    simp[-1] = (f_r, x_r)
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = x_centroid + rho * (x_r - x_centroid)
                    else:
                        # Inside contraction
                        x_c = x_centroid - rho * (x_worst - x_centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    evals_local += 1
                    
                    if f_c <= min(f_r, f_worst):
                        simp[-1] = (f_c, x_c)
                    else:
                        # Shrink
                        for i in range(1, len(simp)):
                            simp[i][1] = simp[0][1] + sigma * (simp[i][1] - simp[0][1])
                            simp[i][1] = np.clip(simp[i][1], -5, 5)
                            fi = func(simp[i][1])
                            evals_local += 1
                            simp[i][0] = fi
                            
            return best_x_local, best_f_local, evals_local

        # Call Nelder-Mead with remaining budget
        remaining_budget = budget - evals
        if remaining_budget > 0:
            final_x, final_f, used_evals = nelder_mead(func, best_x, remaining_budget, dim, deadline)
            if final_f < best_f:
                best_f = final_f
                best_x = final_x
            
        return best_x, best_f
# EVOLVE-BLOCK-END
