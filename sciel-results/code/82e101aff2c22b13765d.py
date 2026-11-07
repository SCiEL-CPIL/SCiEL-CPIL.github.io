"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ7 using a structured grid + local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # DTLZ7 structure: first 21 variables are decision variables, last 1 is the G variable
        # DTLZ7: g(x) = 1 + 10*n_g*(sum(x_g) - n_g*sin(2*pi*x_g)) / n_g
        # where n_g = dim - M + 1 = 22 - 3 + 1 = 20
        # f1 = (1+g)*prod(cos(x_i*pi/2)) for i=1..M-2
        # f2 = (1+g)*cos(x_{M-1}*pi/2)*sin(x_1*pi/2)
        # f3 = (1+g)*sin(x_{M-1}*pi/2)
        
        n_g = dim - n_obj + 1  # 20
        n_m = n_obj - 1  # 2 variables for the angles
        
        # Strategy:
        # 1. Use a structured approach: sample the Pareto front structure directly
        # 2. For DTLZ7, the Pareto front is non-convex
        # 3. The optimal solution has g(x_g)=0, which means sum(x_g) = n_g*sin(2*pi*x_g)
        #    This is achieved when all x_g = 0 (or other specific patterns)
        # 4. The angles x_1, x_2 determine the position on the Pareto front
        
        # Best approach: set x_g to values that make g=0, then optimize angles
        # g(x_g) = 10*n_g*(sum(x_g) - n_g*sin(2*pi*x_g))/n_g = 10*(sum(x_g) - n_g*sin(2*pi*x_g))
        # For g=0: sum(x_g) = n_g*sin(2*pi*x_g)
        # If all x_g = 0: sum=0, n_g*sin(0)=0, so g=0. 
        # If all x_g = 1: sum=n_g, n_g*sin(2*pi)=0, so g=10*n_g which is bad.
        # So x_g=0 is optimal for the g part.
        
        # Let's verify: with x_g all zeros, g=0
        # Then f1 = prod(cos(x_i*pi/2)) for i=1,2 (since M=3, M-2=1, so just x_1)
        # Wait, let me re-read DTLZ7 definition.
        
        # DTLZ7 with M=3:
        # n_g = M - 1 = 2? No, standard DTLZ7: n_g = n - M + 1
        # For n=22, M=3: n_g = 20
        # g(x_g) = 1 + 10*n_g*(sum(x_g) + n_g*sin(2*pi*x_g))/n_g ... 
        # Actually the standard formula is:
        # g(x) = 1 + 10*n_g*(sum_{i=1}^{n_g}(x_i - sin(2*pi*x_i)))/n_g
        # Hmm, let me think about what makes g minimal.
        # g is minimized when each x_i - sin(2*pi*x_i) is minimized.
        # The function h(x) = x - sin(2*pi*x) has minimum at x=0 (h(0)=0) and x=0.5 (h(0.5)=0.5-sin(pi)=0.5)
        # Wait, h(0)=0, h(0.5)=0.5-0=0.5, h(1)=1-0=1
        # So minimum of h is at x=0 with value 0.
        # So g_min = 1 + 10*n_g*0/n_g = 1.
        
        # Actually I need to be more careful. The standard DTLZ7 g function:
        # g(x) = 1 + 10*n_g*(sum_{i=1}^{n_g} x_i - n_g*sin(2*pi*x_i))/n_g
        # No, that's not right either. Let me just use a practical approach.
        
        # Since problem(x) clips and evaluates, let me just do a smart search.
        
        # Key insight: for DTLZ7, the Pareto front is parameterized by angles.
        # The g-part should be minimized (set to its minimum value).
        # The angle variables determine the position on the front.
        
        # Let's use a two-phase approach:
        # Phase 1: Generate diverse points on the approximate Pareto front
        # Phase 2: Local refinement
        
        results = []
        used = 0
        
        def evaluate(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            if len(x) != dim:
                x = np.resize(x, dim)
            f = problem(x)
            used += 1
            results.append(f)
            return f
        
        # Phase 1: Structured sampling of the Pareto front
        # For DTLZ7 with M=3, the Pareto front is 2D
        # Parameterize by theta_1, theta_2 in [0, pi/2]
        # Set all x_g variables to 0 (or near 0) to minimize g
        # Set x_{n_g+1} = theta_1, x_{n_g+2} = theta_2
        
        # Grid for theta_1, theta_2
        n_grid = 12
        theta_vals = np.linspace(0, np.pi/2, n_grid)
        
        # Create a 2D grid of (theta_1, theta_2) pairs
        for t1 in theta_vals:
            for t2 in theta_vals:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                x = np.zeros(dim)
                # First n_g variables (indices 0 to n_g-1) set to 0 for minimal g
                # Last 2 variables (indices n_g and n_g+1) are the angles
                x[n_g] = t1
                x[n_g + 1] = t2
                f = evaluate(x)
            if time.time() > deadline or used >= budget:
                break
        
        if used >= budget or time.time() > deadline:
            return results[-1] if results else None
        
        # Phase 2: Local refinement around promising points
        # Sort by hypervolume contribution (approximate: sum of objectives, lower is better)
        if len(results) > 0:
            results_arr = np.array(results)
            # Filter out points that are worse than reference in any objective
            ref = np.array([1.1, 1.1, 25.0])
            valid_mask = np.all(results_arr < ref, axis=1)
            valid_results = results_arr[valid_mask]
            
            if len(valid_results) > 0:
                # Find non-dominated points
                def dominates(a, b):
                    return np.all(a <= b) and np.any(a < b)
                
                nd_idx = []
                for i in range(len(valid_results)):
                    is_nd = True
                    for j in range(len(valid_results)):
                        if i != j and dominates(valid_results[j], valid_results[i]):
                            is_nd = False
                            break
                    if is_nd:
                        nd_idx.append(i)
                
                # Refine around non-dominated points
                for idx in nd_idx[:10]:  # Refine up to 10 best points
                    if time.time() > deadline or used >= budget:
                        break
                    base = np.zeros(dim)
                    base[n_g] = theta_vals[int(idx % n_grid)]
                    base[n_g + 1] = theta_vals[int(idx // n_grid)]
                    
                    # Local search with small perturbations
                    for _ in range(3):
                        if time.time() > deadline or used >= budget:
                            break
                        perturbation = 0.05 * np.random.randn(dim)
                        x = np.clip(base + perturbation, lo, hi)
                        f = evaluate(x)
                    
                    # Also try varying the g variables slightly
                    for _ in range(2):
                        if time.time() > deadline or used >= budget:
                            break
                        x = base.copy()
                        # Small perturbation to g variables
                        x[:n_g] = np.clip(np.random.rand(n_g) * 0.1, 0, 1)
                        f = evaluate(x)
        
        # Phase 3: If budget remains, do more diverse sampling
        if used < budget and time.time() < deadline:
            remaining = budget - used
            n_extra = min(remaining, 50)
            for _ in range(n_extra):
                if time.time() > deadline or used >= budget:
                    break
                # Random sampling with bias towards low g
                x = np.zeros(dim)
                # g variables: small values
                x[:n_g] = np.random.rand(n_g) * 0.05
                # Angle variables: uniform in [0, pi/2]
                x[n_g] = np.random.rand() * np.pi / 2
                x[n_g + 1] = np.random.rand() * np.pi / 2
                f = evaluate(x)
        
        return results[-1] if results else None
# EVOLVE-BLOCK-END
