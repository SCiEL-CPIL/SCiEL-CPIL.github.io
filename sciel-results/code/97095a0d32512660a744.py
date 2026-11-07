"""Multi-start Nelder-Mead optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead simplex optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # 290s margin within 300s limit (evaluator kills at 330s, task says 300s)
        # Using 290s to be safe against overhead.
        deadline = start_time + 290 

        best_x = None
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Nelder-Mead parameters
        # Standard NM parameters, slightly tuned for robustness
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        # We will split the budget into several restarts.
        # For dim=5, NM is very fast. 
        # Let's use 20 restarts. 500 / 20 = 25 evaluations per restart.
        # This is tight for NM, but NM converges very quickly in low dimensions.
        # Alternatively, 10 restarts with 50 evals each.
        # Given the "early progress counts" metric, rapid initial descent is key.
        # NM from a good start descends very fast.
        
        n_restarts = 10
        budget_per_restart = budget // n_restarts
        
        # Ensure we don't exceed budget due to integer division
        # We will manage the total calls carefully.
        
        rng = np.random.default_rng(42)

        for _ in range(n_restarts):
            # Check time and budget
            if time.time() > deadline or total_calls >= budget:
                break
            
            # Initialize a diverse starting point
            # Uniform in [-4, 4]^5 to stay within bounds and near center
            # But we want diversity. 
            # Let's use a stratified approach or just random uniform in a slightly smaller box to avoid boundaries initially?
            # The problem says minimizer is in [-4,4]^5.
            # Starting in [-4,4] is safe.
            
            x_start = rng.uniform(-4.0, 4.0, size=dim)
            
            # Run Nelder-Mead from x_start
            # We implement a simple NM loop that respects the global budget and time.
            
            # Simplex initialization
            # Vertices: x0, and x0 + delta * e_i
            # delta should be small enough to be local but large enough to see gradient.
            # For ill-conditioned, small delta might be better to avoid jumping over narrow valleys?
            # Or large to find the valley? 
            # Standard NM uses delta=0.05 or 0.1 relative to scale.
            # Scale is 10 (from -5 to 5). So delta=0.5 or 1.0?
            # Let's try delta=0.5.
            
            delta = 0.5
            simplex = [x_start.copy()]
            for i in range(dim):
                v = x_start.copy()
                v[i] += delta
                simplex.append(v)
            
            # Evaluate initial simplex
            fvals = []
            for v in simplex:
                if total_calls >= budget or time.time() > deadline:
                    break
                fvals.append(safe_call(v))
            
            if len(fvals) < dim + 1:
                break

            # Sort simplex by fvals
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Main NM loop
            # Max iterations per restart to prevent getting stuck in one local min if it's slow
            # But we are limited by global budget anyway.
            
            while True:
                if total_calls >= budget or time.time() > deadline:
                    break
                
                # Check convergence: if range of fvals is small, stop
                if max(fvals) - min(fvals) < 1e-8:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = safe_call(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    f_e = safe_call(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - x_r)
                    
                    f_c = safe_call(x_c)
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        # Move all points except best towards best
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if total_calls >= budget or time.time() > deadline:
                                break
                            fvals[i] = safe_call(simplex[i])
                        
                        if total_calls >= budget or time.time() > deadline:
                            break
                        
                        # Re-sort after shrink
                        order = np.argsort(fvals)
                        simplex = [simplex[i] for i in order]
                        fvals = [fvals[i] for i in order]
                        continue

                # Re-sort after update
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

        return best_x, best_f
# EVOLVE-BLOCK-END
