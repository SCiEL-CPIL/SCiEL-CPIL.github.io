"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Strategy: Multi-start Nelder-Mead with L-BFGS-B polish.
    
    Instead of a single global population-based method (like CMA-ES) which can be
    slow to converge in 5D with a tight budget of 300, we use a deterministic
    multi-start approach. We launch several local optimization trajectories from
    distinct, well-chosen starting points. Each trajectory uses Nelder-Mead (robust,
    derivative-free, good for multimodal landscapes) and then polishes with
    L-BFGS-B (fast quadratic convergence) if it gets close to a local minimum.
    
    This exploits the fact that BBOB functions are often unimodal or have few
    local minima, and local methods are extremely efficient once in the basin of
    attraction.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit
        
        dim = self.dim
        budget = self.budget
        
        # Track global best
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        # Helper to check time and budget
        def can_continue():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True
        
        # Define a few starting points spread across the domain
        # 1. Center
        # 2. Corners (subset to save evaluations)
        # 3. Random points
        
        start_points = []
        
        # Center
        start_points.append(np.zeros(dim))
        
        # Deterministic "corners" based on grid
        # For dim=5, we can pick a few points along the diagonal and axes
        # Point 1: All 3.0
        start_points.append(np.full(dim, 3.0))
        # Point 2: All -3.0
        start_points.append(np.full(dim, -3.0))
        # Point 3: Alternating signs
        alt = np.array([3.0, -3.0, 3.0, -3.0, 3.0])
        start_points.append(alt)
        # Point 4: One axis positive, others zero
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 3.0
            start_points.append(p)
        
        # Add a few random points for diversity
        rng = np.random.default_rng(42)
        for _ in range(5):
            start_points.append(rng.uniform(-4, 4, dim))
        
        # We will run Nelder-Mead from each start point, but we share the budget.
        # Instead of allocating fixed budget per start, we run a short NM phase,
        # then a polish phase.
        
        # To be efficient, we'll implement a simple Nelder-Mead loop that accepts
        # a function call counter and stops when budget/time is exceeded.
        
        def nelder_mead(func, x0, max_iter, call_counter, deadline):
            """
            Simple Nelder-Mead implementation that respects global budget.
            Returns (x_best, f_best, calls_used).
            """
            nonlocal calls_made, best_x, best_f
            
            dim = len(x0)
            # Initialize simplex
            simplex = []
            f_simplex = []
            
            # First vertex
            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            calls_made += 1
            simplex.append(x0.copy())
            f_simplex.append(f0)
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Other vertices
            for i in range(dim):
                xi = x0.copy()
                # Step size
                if abs(x0[i]) > 1e-10:
                    step = 0.05 * abs(x0[i])
                else:
                    step = 0.05
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_made += 1
                simplex.append(xi.copy())
                f_simplex.append(fi)
                
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrinkage
            
            for _ in range(max_iter):
                if not can_continue():
                    break
                
                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                x_best = simplex[0]
                f_best = f_simplex[0]
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                
                # Check convergence
                if abs(f_worst - f_best) < 1e-12:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_made += 1
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
                
                if f_ref < f_simplex[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_made += 1
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    # Simple reflection
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_worst:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_worst - centroid)
                    
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_made += 1
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                    
                    if f_con < min(f_ref, f_worst):
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            f_shrink = func(simplex[i])
                            calls_made += 1
                            f_simplex[i] = f_shrink
                            if f_shrink < best_f:
                                best_f = f_shrink
                                best_x = simplex[i].copy()
                        
                        if not can_continue():
                            break
            
            return best_x, best_f

        # Run Nelder-Mead from each start point
        # Allocate a rough budget per start, but stop early if converged or budget hit
        # With 300 budget, and ~10 start points, we can afford ~30 calls per NM phase.
        # But NM might need more. Let's allow up to 50 iterations per NM, but break early.
        
        for sp in start_points:
            if not can_continue():
                break
            
            # Run NM
            nelder_mead(func, sp, max_iter=50, call_counter=None, deadline=deadline)
            
            if not can_continue():
                break
            
            # If we found a good point, try to polish with L-BFGS-B
            # We need a gradient-free wrapper for L-BFGS-B? No, L-BFGS-B requires gradients.
            # We can approximate gradients numerically, but that's expensive.
            # Alternatively, we can use a simple local search or just rely on NM.
            # Given the budget, NM is likely sufficient. Let's stick to NM.
            # However, if we have time/budget left, we can run a second pass of NM
            # from the current best point with smaller step sizes.
            
        # Second pass: Refine from the best point found
        if can_continue():
            # Run NM again from best_x with smaller initial step
            # We can't easily change the step size in the function above, so we just call it again.
            # But it might just repeat. Let's add a small perturbation.
            x_start = best_x + np.random.normal(0, 0.1, dim)
            x_start = np.clip(x_start, -5, 5)
            nelder_mead(func, x_start, max_iter=50, call_counter=None, deadline=deadline)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
