"""Rastrigin-aware optimizer: Cuckoo Search for global basin hopping + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer tailored for BBOB f3 (Rastrigin).
    Strategy:
    1. Cuckoo Search (CS) for global exploration.
       - Lévy flights allow large jumps to escape the 10x10x... grid of local minima.
       - Survival of the fittest keeps the best candidates.
    2. Local Refinement:
       - Once a good basin is found, use a simplified Nelder-Mead or coordinate descent
         to polish the solution to high precision.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        calls_made = 0
        dim = self.dim
        budget = self.budget
        n_calls_limit = budget

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        best_x = np.zeros(dim)
        best_f = np.inf
        history = []

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= n_calls_limit:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            history.append(fx)
            return fx

        rng = np.random.default_rng(42)

        # --- Phase 1: Cuckoo Search Global Exploration ---
        # Cuckoo Search is effective for multimodal functions due to Lévy flights.
        # We use a small population to conserve budget for exploration.
        pop_size = 15
        # Initial population: mix of random and structured (origin + small perturbations)
        # Rastrigin has a global min at 0 (before shift), so starting near 0 is smart.
        # But shift is hidden. So we sample broadly but keep some near 0.
        
        # Initialize population
        population = []
        for i in range(pop_size):
            if i == 0:
                x = np.zeros(dim)
            elif i < 3:
                # Small perturbations around 0
                x = rng.normal(0, 0.5, dim)
            else:
                # Broad sampling in [-4, 4]
                x = rng.uniform(-4, 4, dim)
            x = clip(x)
            fx = counted_func(x)
            population.append((x, fx))
            
        # Sort by fitness
        population.sort(key=lambda t: t[1])
        
        # Lévy flight parameters
        alpha = 0.5
        beta = 1.5
        p = 0.05 # Discovery rate

        # Main CS Loop
        # We spend ~70% of budget on CS
        cs_budget = int(budget * 0.7)
        
        while calls_made < cs_budget and time.time() < deadline - 30:
            # 1. Generate new cuckoos (Lévy flight)
            # Pick a random solution
            idx = rng.integers(0, pop_size)
            x_parent = population[idx][0]
            
            # Lévy flight step
            # Mandelbrot-Barnley method for Lévy stable random variables
            s = rng.standard_normal()
            t = rng.standard_normal()
            # Simplified Lévy flight generation
            # Using a simpler approximation: heavy-tailed distribution
            # Or just use a scaled Gaussian with occasional large jumps
            # For Rastrigin, large jumps (period ~ 2*pi/2 ~ 3.14) are needed.
            # Let's use a mixture: 90% small steps, 10% large steps.
            if rng.random() < 0.1:
                step = rng.uniform(-3, 3, dim)
            else:
                step = rng.normal(0, 0.5, dim)
            
            x_new = x_parent + step
            x_new = clip(x_new)
            
            f_new = counted_func(x_new)
            
            # 2. Replace if better (or with probability p, displace a random one)
            # Standard CS: replace if f_new < f_old
            # We also do a random displacement with prob p to maintain diversity
            if f_new < population[idx][1]:
                population[idx] = (x_new, f_new)
            elif rng.random() < p:
                # Displace a random worse solution
                jdx = rng.integers(0, pop_size)
                if f_new < population[jdx][1]:
                    population[jdx] = (x_new, f_new)
            
            # Re-sort to keep best at top
            population.sort(key=lambda t: t[1])

        # --- Phase 2: Local Refinement ---
        # Use the best point found by CS to start a local search.
        # Nelder-Mead is robust and derivative-free.
        # We use a tight tolerance to squeeze out the last bits of precision.
        
        if calls_made < budget and time.time() < deadline - 5:
            # Try a few local refinements from the top 3 candidates
            for i in range(3):
                if calls_made >= budget or time.time() > deadline - 5:
                    break
                x_start = population[i][0].copy()
                
                # Custom Nelder-Mead implementation for tighter control and speed
                # Or use scipy if available and fast enough, but manual is safer for budget.
                # Let's use a simple coordinate descent + line search hybrid for speed.
                # Actually, Nelder-Mead is better for Rastrigin's wiggly local structure.
                
                # Simple Nelder-Mead
                def nm_minimize(func, x0, budget_left, deadline):
                    n = len(x0)
                    simplex = [x0.copy() for _ in range(n + 1)]
                    for i in range(n):
                        simplex[i+1] = x0.copy()
                        simplex[i+1][i] += 0.5 if x0[i] < 4.5 else -0.5
                        simplex[i+1] = clip(simplex[i+1])
                    
                    fvals = [func(x) for x in simplex]
                    
                    # Track local best
                    local_best_x = x0.copy()
                    local_best_f = np.inf
                    for x, f in zip(simplex, fvals):
                        if f < local_best_f:
                            local_best_f = f
                            local_best_x = x.copy()
                            
                    max_iter = 50
                    for _ in range(max_iter):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        # Sort simplex
                        idx = np.argsort(fvals)
                        simplex = [simplex[i] for i in idx]
                        fvals = [fvals[i] for i in idx]
                        
                        # Check convergence
                        if fvals[-1] - fvals[0] < 1e-8:
                            break
                            
                        # Centroid of best n points
                        centroid = np.mean(simplex[:-1], axis=0)
                        
                        # Reflection
                        alpha_r = 1.0
                        x_r = centroid + alpha_r * (centroid - simplex[-1])
                        x_r = clip(x_r)
                        f_r = func(x_r)
                        
                        if fvals[0] <= f_r < fvals[-2]:
                            simplex[-1] = x_r
                            fvals[-1] = f_r
                        elif f_r < fvals[0]:
                            # Expansion
                            gamma_e = 2.0
                            x_e = centroid + gamma_e * (x_r - centroid)
                            x_e = clip(x_e)
                            f_e = func(x_e)
                            if f_e < f_r:
                                simplex[-1] = x_e
                                fvals[-1] = f_e
                            else:
                                simplex[-1] = x_r
                                fvals[-1] = f_r
                        else:
                            # Contraction
                            gamma_c = 0.5
                            x_c = centroid + gamma_c * (simplex[-1] - centroid)
                            x_c = clip(x_c)
                            f_c = func(x_c)
                            if f_c < fvals[-1]:
                                simplex[-1] = x_c
                                fvals[-1] = f_c
                            else:
                                # Shrink
                                for i in range(1, n + 1):
                                    simplex[i] = (simplex[0] + simplex[i]) / 2.0
                                    simplex[i] = clip(simplex[i])
                                    fvals[i] = func(simplex[i])
                        
                        # Update local best
                        for x, f in zip(simplex, fvals):
                            if f < local_best_f:
                                local_best_f = f
                                local_best_x = x.copy()
                                
                    return local_best_x, local_best_f

                try:
                    x_opt, f_opt = nm_minimize(counted_func, x_start, budget - calls_made, deadline - 5)
                    if f_opt < best_f:
                        best_f = f_opt
                        best_x = x_opt
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
