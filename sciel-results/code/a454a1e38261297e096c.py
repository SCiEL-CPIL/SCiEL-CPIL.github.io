import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            return func(np.clip(x, -5.0, 5.0))

        def time_left():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Grid Sampling to Identify Promising Regions
        # Rastrigin has minima at integer coordinates.
        # We sample a grid of integer points to find the best one.
        # This bypasses the multimodal landscape by directly probing the "valleys".
        # Grid from -4 to 4 covers the likely range of the shifted minimum.
        # 9^5 = 59049 points is too many for 500 budget.
        # So we use a coarse grid first, then refine.
        
        # Coarse grid: step 2.0
        # Points: -4, -2, 0, 2, 4
        # 5^5 = 3125 points. Still too many.
        
        # Let's use a random subset of integer points or a very sparse grid.
        # Or, better: use the structure.
        # The function is sum of 1D Rastrigin terms.
        # f(x) = sum_i (x_i^2 - 10 cos(2 pi x_i)) + 10 * dim (shifted/rotated/offset)
        
        # Since we don't know the rotation, we can't just check integers in the original space.
        # However, the rotation is fixed for the instance.
        # The "integer" structure is in the rotated space.
        # This makes direct grid search on integers in the input space ineffective if rotation is non-trivial.
        
        # Wait, the prompt says: "Instances apply a hidden seeded shift, rotation and offset".
        # If there is a rotation, the minima are NOT at integer coordinates in the input space x.
        # They are at integer coordinates in the transformed space z = R(x - s).
        # So x = R^-1(z + s).
        # The minima are at x_k = R^-1(k + s) for integer vector k.
        
        # Since we don't know R or s, we can't directly jump to the minima.
        # However, we can use a global search to find a good basin, then refine.
        
        # Let's stick to a robust global search.
        # The PSO approach (Inspiration 1) scored 1.30.
        # The DE approach (Current) scored 1.10.
        # Let's try to combine PSO's exploration with a better refinement.
        
        # Strategy:
        # 1. Run PSO with a moderate population to explore the landscape.
        # 2. Use the best particle found to initialize a local search.
        # 3. Use a (1+1)-ES with adaptive step size for refinement.
        
        # To improve on PSO, we can add a "restart" mechanism or a "multi-start" element.
        # Or, we can use the "harmonic analysis" idea differently:
        # If we evaluate at points x_i, we can't easily extract the rotation.
        
        # Let's try a hybrid:
        # Phase 1: CMA-ES or DE for global search?
        # SciPy is available. CMA-ES is in scipy.optimize.
        # But we have a limited budget.
        
        # Let's try a "Multi-start (1+1)-ES" with larger initial sigma, similar to Inspiration 2 but with better parameters.
        # Inspiration 2 scored 1.09.
        
        # Let's go back to PSO but tune it.
        # The PSO in Inspiration 1 used:
        # n_particles = 20
        # w = 0.7, c1 = 1.5, c2 = 1.5
        # max_v = 2.0
        
        # Let's try to make the PSO more aggressive in exploration initially, then switch to exploitation.
        # Or, use a "hybrid" approach:
        # 1. Global search with a population-based method (DE or PSO).
        # 2. Local search with (1+1)-ES.
        
        # Let's use DE with a larger population for global search, then refine.
        # Current DE: pop_size=10, 5 restarts.
        # Let's try: pop_size=20, 1 restart, then refine.
        # Or, pop_size=15, 2 restarts.
        
        # Actually, the "harmonic analysis" hint suggests exploiting the periodicity.
        # Even with rotation, the function is periodic in the rotated coordinates.
        # The period is 1 in each rotated dimension.
        # This means the landscape has a "grid" of minima.
        # A global search should be able to find one of these minima.
        
        # Let's try a different global search: Simulated Annealing?
        # Or, just a very robust DE.
        
        # Let's combine:
        # 1. Global Search: DE with population size 20, run for ~300 calls.
        # 2. Local Search: (1+1)-ES with ~200 calls.
        
        # To make it better, we can use "adaptive" DE parameters.
        # Or, we can use "jDE" (self-adaptive DE).
        
        # Let's try a simpler but effective approach:
        # 1. Initialize a population of 30 random points.
        # 2. Evaluate them.
        # 3. Keep the best 10.
        # 4. Generate offspring using DE mutation/crossover.
        # 5. Select best from parents + offspring.
        # 6. Repeat until budget is nearly exhausted.
        # 7. Final (1+1)-ES refinement.
        
        # This is essentially DE.
        
        # Let's try to incorporate the "grid" idea by evaluating at integer points first?
        # If rotation is small, integer points are good.
        # If rotation is large, they are not.
        # But the rotation is seeded. It might be small?
        # "hidden seeded shift, rotation and offset".
        # Usually, these benchmarks use random rotations.
        
        # Let's stick to the PSO idea from Inspiration 1, which scored 1.30.
        # We will improve it by:
        # 1. Using a larger population (30) for better exploration.
        # 2. Adaptive inertia weight (start high, decrease).
        # 3. Better velocity clamping.
        # 4. Stronger (1+1)-ES refinement.
        
        n_particles = 30
        w_start = 0.9
        w_end = 0.4
        c1 = 2.0
        c2 = 2.0
        
        # Initialize particles
        X = rng.uniform(-5, 5, (n_particles, dim))
        V = rng.uniform(-1, 1, (n_particles, dim))
        
        # Evaluate initial population
        F = np.array([safe_func(p) for p in X])
        
        # Personal bests
        P_best = X.copy()
        P_best_f = F.copy()
        
        # Global best
        global_idx = np.argmin(F)
        G_best = X[global_idx].copy()
        G_best_f = F[global_idx]
        
        if G_best_f < best_f:
            best_f = G_best_f
            best_x = G_best.copy()

        # PSO Iterations
        max_iters = (budget - n_particles - 50) // n_particles  # Reserve 50 for refinement
        
        for it in range(max_iters):
            if not time_left() or call_count >= budget - n_particles - 10:
                break
            
            # Adaptive inertia
            w = w_start - (w_start - w_end) * (it / max_iters)
            
            # Update velocities
            r1 = rng.random((n_particles, dim))
            r2 = rng.random((n_particles, dim))
            
            V = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (G_best - X)
            
            # Limit velocity
            max_v = 1.5
            V = np.clip(V, -max_v, max_v)
            
            # Update positions
            X_new = X + V
            X_new = np.clip(X_new, -5, 5)
            
            # Evaluate new positions
            F_new = np.array([safe_func(x) for x in X_new])
            
            # Update personal bests
            improved = F_new < P_best_f
            P_best[improved] = X_new[improved]
            P_best_f[improved] = F_new[improved]
            
            # Update global best
            min_idx = np.argmin(F_new)
            if F_new[min_idx] < G_best_f:
                G_best_f = F_new[min_idx]
                G_best = X_new[min_idx].copy()
                
                if G_best_f < best_f:
                    best_f = G_best_f
                    best_x = G_best.copy()
            
            X = X_new
            F = F_new

        # Phase 2: Local Refinement with (1+1)-ES
        if best_x is not None and time_left() and call_count < budget - 1:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining = budget - 1 - call_count
            
            for _ in range(remaining):
                if not time_left() or call_count >= budget - 1:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.85
                
                sigma = np.clip(sigma, 0.001, 2.0)

        return best_x, best_f
