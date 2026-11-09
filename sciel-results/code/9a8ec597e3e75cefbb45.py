"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart Lattice + Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        x_best = None
        f_best = np.inf

        # Helper to update best
        def update_best(x, f):
            nonlocal x_best, f_best
            if f < f_best:
                x_best = x.copy()
                f_best = f

        # Phase 1: Global Lattice Search (coarse to fine)
        # We allocate ~20% of the budget to this phase to ensure early progress.
        # 3 levels of lattice.
        # Level 0: 5x5x5x5x5 = 3125 points. Too many? 
        # 500 budget. 
        # Let's use a smaller lattice or fewer points.
        # 3^5 = 243 points. Good for coarse.
        # 4^5 = 1024. Too many.
        
        # Strategy: 
        # 1. 3^5 lattice (243 points) over [-5, 5].
        # 2. Take the best 5 points.
        # 3. Run Nelder-Mead from each.
        
        # But 243 points is almost half the budget. 
        # If we use 243 points, we have 257 left for 5 NM runs.
        # NM in 5D takes ~50-100 calls to converge.
        # 5 * 100 = 500. Too many.
        
        # Let's reduce lattice size.
        # 2^5 = 32 points. Very coarse.
        # 3^5 = 243 points.
        
        # Alternative: Random sampling + NM?
        # The seed (1+1)ES scored 1.0. It's simple.
        # NM is deterministic and efficient for smooth functions.
        # Griewank-Rosenbrock is smooth but has a narrow valley.
        
        # Let's try:
        # 1. 100 random points (uniform).
        # 2. Select best 10.
        # 3. Run NM from each with remaining budget / 10.
        
        # Actually, let's stick to a structured grid to ensure coverage.
        # 3^5 = 243 is a lot.
        # How about 2^5 = 32?
        # Then we have 468 left.
        # 468 / 5 restarts = 93 calls per NM.
        # NM in 5D usually converges in < 50 iterations if initialized well.
        
        # Let's do:
        # 1. 32-point lattice (corners of 2x2x2x2x2 grid in [-5,5]).
        #    Actually, 2 points per dim -> 32 points.
        # 2. Add 100 random points. Total 132.
        # 3. Best 5 points.
        # 4. NM from each.
        
        # Total calls: 132 + 5 * (budget - 132) / 5 = budget.
        
        # Let's refine the lattice.
        # 3 points per dim -> 243.
        # If we do 243, we have 257 left.
        # 257 / 5 = 51 calls per NM.
        # This is tight but feasible for NM.
        
        # Let's try 243 lattice points.
        # Grid points: -5, 0, 5.
        # This covers the box well.
        
        lattice_points = []
        for x1 in [-5, 0, 5]:
            for x2 in [-5, 0, 5]:
                for x3 in [-5, 0, 5]:
                    for x4 in [-5, 0, 5]:
                        for x5 in [-5, 0, 5]:
                            lattice_points.append([x1, x2, x3, x4, x5])
        
        lattice_points = np.array(lattice_points)
        
        # Evaluate lattice
        f_vals = []
        for p in lattice_points:
            if time.time() > deadline:
                break
            f_vals.append(func(p))
            
        if not f_vals:
             # Fallback
             x = rng.uniform(-5, 5, dim)
             return x, func(x)

        f_vals = np.array(f_vals)
        
        # Select top K candidates
        K = 5
        top_indices = np.argpartition(f_vals, K)[:K]
        top_points = lattice_points[top_indices]
        top_fvals = f_vals[top_indices]
        
        # Update best
        min_idx = np.argmin(top_fvals)
        update_best(top_points[min_idx], top_fvals[min_idx])
        
        # Phase 2: Nelder-Mead from top K
        # Budget remaining
        calls_made = len(f_vals)
        remaining_budget = budget - calls_made
        budget_per_nm = remaining_budget // K
        
        for i in range(K):
            if time.time() > deadline:
                break
            if budget_per_nm < 10:
                break
                
            x0 = top_points[i]
            f0 = top_fvals[i]
            
            # Nelder-Mead implementation
            # Initialize simplex
            simplex = [x0.copy() for _ in range(dim + 1)]
            f_simplex = [f0] * (dim + 1)
            
            # Perturb vertices
            for j in range(1, dim + 1):
                simplex[j] = x0.copy()
                simplex[j][j-1] += 0.1 * (simplex[j][j-1] if simplex[j][j-1] != 0 else 1.0)
                # Ensure within bounds
                simplex[j] = np.clip(simplex[j], -5, 5)
                f_simplex[j] = func(simplex[j])
                calls_made += 1
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
                
            # Sort simplex by f
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
            update_best(simplex[0], f_simplex[0])
            
            # NM Iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for _ in range(budget_per_nm):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                    
                # Centroid of best dim points
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            f_simplex[j] = func(simplex[j])
                            calls_made += 1
                            if time.time() > deadline:
                                break
                        
                        if time.time() > deadline:
                            break
                
                # Sort
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                update_best(simplex[0], f_simplex[0])

        return x_best, f_best
# EVOLVE-BLOCK-END
