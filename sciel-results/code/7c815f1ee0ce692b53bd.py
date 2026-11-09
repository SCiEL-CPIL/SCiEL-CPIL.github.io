import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = 325.0  # Safety margin before 330s kill

        calls = 0
        best_x = None
        best_f = float('inf')
        
        # We will track the best found so far
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Strategy:
        # 1. Coarse Grid Search to find the basin of attraction.
        #    Rastrigin has many local minima. The global minimum is shifted.
        #    A grid search with reasonable resolution can locate the global basin.
        # 2. Nelder-Mead Simplex Method to refine the solution within that basin.
        #    NM is derivative-free and robust for small dimensions.
        
        # Phase 1: Grid Search
        # With budget 500, we can afford a decent grid.
        # 5 dimensions.
        # If we use 3 points per dimension: 3^5 = 243 evaluations.
        # This leaves 257 for local search.
        # 3 points: -5, 0, 5. This is very coarse.
        # Let's use 4 points per dimension: -5, -1.67, 1.67, 5. 4^5 = 1024 > 500. Too many.
        # Let's use a smarter grid or fewer points.
        
        # Alternative: Random Sampling + Best Point Selection
        # Random sampling with 200 points might find the basin.
        # Then NM for the rest.
        
        # But Rastrigin is rugged. Random might miss the global basin if it's small.
        # The shift is in [-4, 4]. The period is 2pi ~ 6.28.
        # So there are roughly 10/6.28 ~ 1.6 periods in the domain.
        # The global minimum is at a specific point (shifted).
        # Local minima are at integer multiples of 2pi (shifted).
        
        # A grid of 5 points per dimension: 5^5 = 3125. Too many.
        # A grid of 4 points per dimension: 4^5 = 1024. Too many.
        # A grid of 3 points per dimension: 3^5 = 243. Feasible.
        # Points: -5, 0, 5.
        # This might miss the basin if the shift is, say, 3.0.
        # The minima are at ... -2pi, 0, 2pi ... shifted by s.
        # If s=3, minima are at 3, 3+2pi=9.28 (out), 3-2pi=-3.28.
        # Grid points -5, 0, 5.
        # Closest to 3 is 5 (dist 2) or 0 (dist 3).
        # Closest to -3.28 is -5 (dist 1.72) or 0 (dist 3.28).
        # So -5 would be selected. NM from -5 might converge to -3.28.
        
        # Let's try a slightly denser grid for the first few points?
        # Or just use 200 random points.
        # Random points are better distributed in high dimensions than a grid.
        
        # Let's do 200 random points.
        # Then pick the best one.
        # Then run Nelder-Mead from there.
        
        # NM Parameters:
        # Initial simplex: 6 points (n+1).
        # Step size: 0.5?
        
        # Let's refine the random search.
        # Use 150 random points.
        # Then NM.
        
        # NM implementation details:
        # Standard Nelder-Mead.
        # Needs to handle box constraints?
        # We can clip the points.
        
        # Let's write a simple NM.
        
        def nelder_mead(func, x0, bounds=(-5.0, 5.0), max_iter=100):
            n = len(x0)
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i, i - 1] += 0.5  # Initial step size
            
            # Evaluate initial simplex
            fs = np.array([evaluate(simplex[i]) for i in range(n + 1)])
            
            # Main loop
            for _ in range(max_iter):
                if time.time() - start_time > deadline:
                    break
                if calls >= budget:
                    break
                
                # Sort simplex
                idx = np.argsort(fs)
                simplex = simplex[idx]
                fs = fs[idx]
                
                # Check convergence
                if np.std(fs) < 1e-10:
                    break
                
                # Centroid of all but worst
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_reflected = x_centroid + alpha * (x_centroid - simplex[-1])
                f_reflected = evaluate(x_reflected)
                
                if f_reflected < fs[0]:
                    # Expansion
                    gamma = 2.0
                    x_expanded = x_centroid + gamma * (x_centroid - simplex[-1])
                    f_expanded = evaluate(x_expanded)
                    if f_expanded < f_reflected:
                        simplex[-1] = x_expanded
                        fs[-1] = f_expanded
                    else:
                        simplex[-1] = x_reflected
                        fs[-1] = f_reflected
                elif f_reflected < fs[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflected
                    fs[-1] = f_reflected
                else:
                    # Contraction
                    beta = 0.5
                    if f_reflected < fs[-1]:
                        # Outside contraction
                        x_contracted = x_centroid + beta * (x_reflected - x_centroid)
                    else:
                        # Inside contraction
                        x_contracted = x_centroid - beta * (x_reflected - x_centroid)
                    
                    f_contracted = evaluate(x_contracted)
                    
                    if f_contracted < min(f_reflected, fs[-1]):
                        simplex[-1] = x_contracted
                        fs[-1] = f_contracted
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fs[i] = evaluate(simplex[i])
            
            # Return best
            idx_min = np.argmin(fs)
            return simplex[idx_min], fs[idx_min]

        # Phase 1: Random Search
        # Use 200 calls for random search.
        # This is a structural change from CMA-ES.
        # CMA-ES might struggle with the ruggedness if the population is small.
        # Random search ensures we sample the whole space.
        
        num_random = 200
        for i in range(num_random):
            if calls >= budget or time.time() - start_time > deadline:
                break
            x = np.random.uniform(-5, 5, n)
            evaluate(x)
        
        if best_x is None:
            best_x = np.zeros(n)
        
        # Phase 2: Nelder-Mead from the best point
        # Use remaining budget for NM.
        # NM uses ~6 calls per iteration initially, then 1-2.
        # With 300 calls left, we can do many iterations.
        
        x_opt, f_opt = nelder_mead(func, best_x)
        
        # If NM didn't improve, best_x is still valid.
        # evaluate() updates best_x globally.
        
        return best_x, best_f
