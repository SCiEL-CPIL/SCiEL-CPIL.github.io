"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Grid + CMA-ES + Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            try:
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except:
                return float('inf')

        # Phase 1: Coarse Grid Search for early progress
        # We want to make good progress quickly to boost AOCC.
        # A grid of 3 points per dim in [-5, 5] gives 3^5 = 243 points.
        # This fits well within the budget (500) and provides a solid start.
        # Points: -5, 0, 5
        grid_points = np.array(np.meshgrid([-5, 0, 5] * dim)).T.reshape(-1, dim)
        
        # Shuffle grid points to mix in some randomness? No, deterministic grid is good for coverage.
        # But maybe prioritize center?
        # Let's just evaluate all grid points.
        for p in grid_points:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            safe_call(p)
        
        # If we ran out of budget or time, we are done.
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES from the best grid point
        # Use a population size suitable for dim=5.
        # lambda = 4 + floor(3 log(n)) = 4 + 3*1.6 = 8.8 -> 8
        # But we have limited budget left.
        # Let's use a smaller population to fit more iterations.
        # lambda = 6
        lam = 6
        sigma = 1.0
        x_mean = best_x.copy()
        
        # Initialize population around best_x
        # We need to be careful with the number of calls.
        # Each generation costs lam calls.
        # Remaining budget: budget - calls_made
        # Max generations: (budget - calls_made) // lam
        
        # CMA-ES state
        # For simplicity and robustness, we will implement a basic CMA-ES or use scipy's if available?
        # SciPy's CMA-ES is not in scipy.optimize.minimize directly (it's in cma library or differential_evolution).
        # scipy.optimize.differential_evolution is a good alternative for global search.
        # However, DE can be slow.
        # Let's try a simpler population-based approach: (mu+lambda)-ES with recombination.
        # Or just use Nelder-Mead if we are already close.
        
        # Given the "Griewank-Rosenbrock" nature, it's multimodal but has a global minimum.
        # Grid search likely found a good basin.
        # Let's use Nelder-Mead for local refinement, but maybe do a few DE iterations first to ensure we are in the right basin?
        
        # Let's try Differential Evolution via scipy for the remaining budget.
        # But DE requires a population.
        # If remaining budget is small, DE might not be effective.
        
        # Strategy:
        # If remaining budget > 100, run DE.
        # Else, run Nelder-Mead.
        
        rem_budget = budget - calls_made
        
        if rem_budget > 50:
            # Run a limited DE
            # We need to wrap func for DE.
            # DE in scipy uses 'func' directly.
            # We need to track calls to ensure we don't exceed budget.
            # This is tricky with scipy's black-box optimizers.
            # It's safer to implement a simple (mu+lambda)-ES or just use Nelder-Mead.
            
            # Let's stick to Nelder-Mead for reliability and speed, but maybe do a few random restarts of NM?
            # Or a simple coordinate descent?
            
            # Actually, a (1+1)-ES is already quite good for local.
            # The grid search is the main improvement here for early progress.
            # Let's use Nelder-Mead for the final polish.
            
            # Nelder-Mead can be unstable. Let's use a robust (1+1)-ES for the remaining budget.
            # It's simple and effective.
            
            x = best_x.copy()
            fx = best_f
            sigma = 0.5 # Start smaller as we are likely close
            
            while calls_made < budget and time.time() < deadline:
                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                    
                # If sigma becomes too small, break
                if sigma < 1e-6:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
