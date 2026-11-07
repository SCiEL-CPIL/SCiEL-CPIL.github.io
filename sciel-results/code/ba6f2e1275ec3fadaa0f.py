"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Rastrigin-Structure Exploiting Optimizer: Lattice-Primed Multi-Start Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # State
        calls_made = 0
        best_x = None
        best_f = np.inf
        history = []

        # Helper to safely call func and update state
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            try:
                f_val = func(x)
                calls_made += 1
                if np.isnan(f_val) or np.isinf(f_val):
                    return np.inf
                history.append((x.copy(), f_val))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return np.inf

        rng = np.random.default_rng(42)
        
        # Phase 1: Lattice-Primed Global Search
        # Rastrigin minima are at integers. Even with rotation/shift, 
        # sampling the integer lattice is a highly structured, high-probability 
        # way to hit near-minima compared to uniform random sampling.
        
        # Generate integer lattice points in [-4, 4] for each dimension
        # The minimizer is in [-4, 4]^5, so we focus there.
        lattice_vals = np.arange(-4, 5, 1)  # -4, -3, ..., 4
        # Create all combinations
        # 9^5 = 59049 points. We can store them or generate on the fly.
        # Storing 59049 * 5 * 8 bytes is ~2.3 MB, which is fine.
        lattice_points = np.array(np.meshgrid(*[lattice_vals]*n)).T.reshape(-1, n)
        
        # We cannot evaluate all 59049 points. We have 500 calls total.
        # Let's use 50 calls for global exploration.
        global_budget = 50
        
        # Shuffle lattice points and take the first 'global_budget'
        # This is equivalent to random sampling from the lattice.
        lattice_indices = rng.permutation(len(lattice_points))
        init_points = lattice_points[lattice_indices[:global_budget]]
        
        # Evaluate these lattice points
        for x in init_points:
            if calls_made >= global_budget or time.time() > deadline:
                break
            evaluate(x)
            
        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(n)
            best_f = 1000.0
            history.append((best_x, best_f))

        # Phase 2: Local Refinement
        # Select the top K distinct basins from the global search results.
        # Sort history by function value
        history.sort(key=lambda item: item[1])
        
        # Select top 5 starts, ensuring diversity (distance > 1.0)
        # Since lattice points are distance 1 apart, distinct basins are likely distance >= 1.
        selected_starts = []
        for x, f in history:
            if len(selected_starts) >= 5:
                break
            # Check if x is too close to already selected
            too_close = False
            for xs in selected_starts:
                if np.linalg.norm(x - xs) < 1.0:
                    too_close = True
                    break
            if not too_close:
                selected_starts.append(x)
                
        if not selected_starts:
            selected_starts = [best_x]

        # Remaining budget for local refinement
        remaining_budget = budget - calls_made
        budget_per_restart = remaining_budget // len(selected_starts)
        
        # Run Nelder-Mead from each selected start
        # Nelder-Mead is effective for the local bowl of Rastrigin.
        for start_x in selected_starts:
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Define a wrapper that stops if we exceed the per-restart budget
            # or if the global deadline is hit.
            # We use a simple counter for this specific restart.
            restart_calls = 0
            restart_limit = budget_per_restart
            
            def scipy_func(x):
                nonlocal restart_calls
                if restart_calls >= restart_limit:
                    return np.inf
                if time.time() > deadline:
                    return np.inf
                f_val = evaluate(x)
                restart_calls += 1
                return f_val
            
            try:
                # Nelder-Mead options
                # xatol and fatol help stop early if converged
                res = minimize(
                    scipy_func, 
                    start_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 50, # Cap iterations to avoid infinite loops
                        'xatol': 1e-4, 
                        'fatol': 1e-4,
                        'initial_simplex': None # Default is fine, or we could tune it
                    }
                )
                # The result is already in history via evaluate
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
