import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None
        self._deadline = None
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        self._start_time = time.time()
        # Set deadline to 325s to be safe (limit is 330s)
        self._deadline = self._start_time + 325
        self._call_count = 0
        self.budget_remaining = self.budget
        self._best_x = np.zeros(self.dim)
        self._best_f = np.inf
        self._rng = np.random.default_rng(42)

        def safe_call(x):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                return 1e10
            self._call_count += 1
            self.budget_remaining -= 1
            try:
                val = float(func(x))
            except Exception:
                return 1e10
            if not np.isfinite(val):
                return 1e10
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # 1. Global Exploration: Sobol Sequence
        n_sobol = 30
        try:
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            sobol_points = sampler.random(n_sobol)
            sobol_points = sobol_points * 10.0 - 5.0
        except Exception:
            sobol_points = self._rng.uniform(-5, 5, (n_sobol, self.dim))

        evaluated_points = []
        
        # Evaluate initial Sobol points
        for i in range(n_sobol):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            x = sobol_points[i]
            f = safe_call(x)
            evaluated_points.append((f, x))
            
        if not evaluated_points:
            return self._best_x, self._best_f

        # Sort by fitness
        evaluated_points.sort(key=lambda x: x[0])
        
        # 2. Coarse-to-Fine Grid Search with Restarts
        # Strategy:
        # - Start with the best point from Sobol.
        # - Perform a grid search with a certain radius R.
        # - If improvement is found, halve R and continue.
        # - If no improvement is found in a full grid level, restart from the next best Sobol point (or a perturbation) with a new R.
        
        # We will maintain a list of "centers" to try.
        # Initially, the top Sobol points.
        centers = [p[1].copy() for p in evaluated_points[:10]]
        
        # We'll iterate through these centers, but dynamically.
        # Actually, a simpler and robust approach:
        # Keep a "current center" and a "current radius".
        # If we stagnate, pick a new center from the untried Sobol points (or random).
        
        current_center = evaluated_points[0][1].copy()
        current_radius = 3.0 # Start large enough to cover most of the box
        
        # List of untried centers (Sobol points)
        untried_centers = [p[1].copy() for p in evaluated_points[1:]]
        
        # We will loop until budget runs out
        while self._call_count < self.budget and time.time() < self._deadline:
            # If radius is too small, we've converged locally.
            # Try to restart from a different basin.
            if current_radius < 1e-4:
                if untried_centers:
                    current_center = untried_centers.pop(0)
                    current_radius = 2.0
                else:
                    # No more untried centers, try a random perturbation or break
                    # For now, break to avoid wasting calls on tiny grids
                    break
            
            # Generate grid points: Corners of a hypercube centered at current_center
            # Number of points: 2^dim
            # For dim=5, 32 points.
            
            improved = False
            best_local_f = self._best_f
            best_local_x = self._best_x.copy()
            
            # Generate all 2^dim corner combinations
            # Using product of [-1, 1]
            signs = np.array(np.meshgrid(*[[-1, 1]] * self.dim)).T.reshape(-1, self.dim)
            
            for sign in signs:
                if self._call_count >= self.budget or time.time() >= self._deadline:
                    break
                
                x_new = current_center + sign * current_radius
                x_new = np.clip(x_new, -5, 5)
                
                f_new = safe_call(x_new)
                
                if f_new < best_local_f:
                    best_local_f = f_new
                    best_local_x = x_new.copy()
                    improved = True
            
            if improved:
                # Update global best (safe_call already did, but let's ensure center moves to best found)
                current_center = best_local_x.copy()
                # Halve the radius
                current_radius *= 0.5
            else:
                # Stagnation: No improvement in this grid level.
                # Move to next center.
                if untried_centers:
                    current_center = untried_centers.pop(0)
                    current_radius = 2.0
                else:
                    # If no untried centers, try a random restart
                    # or just break if we've exhausted good candidates
                    # Let's try a random point
                    x_rand = self._rng.uniform(-5, 5, self.dim)
                    f_rand = safe_call(x_rand)
                    if f_rand < self._best_f:
                        current_center = x_rand.copy()
                        current_radius = 2.0
                    else:
                        # If random didn't help, we might be stuck.
                        # Reduce radius anyway to finish off? Or break?
                        # Breaking is safer to save time if we are stuck in a local min.
                        # But we have budget. Let's try one more random restart.
                        if self._call_count < self.budget - 10:
                             x_rand2 = self._rng.uniform(-5, 5, self.dim)
                             safe_call(x_rand2)
                        break

        return self._best_x, self._best_f
