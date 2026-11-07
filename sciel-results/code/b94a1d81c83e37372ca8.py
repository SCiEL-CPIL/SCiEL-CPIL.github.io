"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive Radial Search with local refinement for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0
        
        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5, 5)
            try:
                f = func(x)
            except Exception:
                f = np.inf
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Coarse exploration with radial search
        # Start with a set of promising initial points
        # For BBOB f22 (Gallagher 21 peaks), the landscape is multimodal.
        # We use a grid of starting points and radial search from each.
        
        # Generate initial points:
        # 1. Origin
        # 2. A few random points
        # 3. Points along axes
        
        initial_points = []
        initial_points.append(np.zeros(dim))
        
        # Add some random points
        rng = np.random.default_rng(42)
        for _ in range(5):
            initial_points.append(rng.uniform(-4, 4, size=dim))
        
        # Add points along axes
        for i in range(dim):
            for val in [-3, 3]:
                p = np.zeros(dim)
                p[i] = val
                initial_points.append(p)
        
        # Evaluate initial points
        for p in initial_points:
            f = do_call(p)
            if f is None:
                break
        
        if call_count >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Adaptive radial search
        # Maintain a set of active search centers
        # Each center has a radius that adapts based on success/failure
        
        # Start with the best point found so far
        centers = [best_x.copy()]
        radii = [1.0]
        
        # Also add a few other good points as centers
        # Sort initial points by function value (we don't have stored values, so just use best)
        
        # We'll do iterative refinement:
        # 1. Pick a center
        # 2. Generate points on a sphere of radius r around it
        # 3. If a better point is found, update center and shrink radius
        # 4. If no improvement, expand radius or move to next center
        
        # Use a simple strategy:
        # - Start with radius 2.0
        # - Generate 10-20 points on sphere
        # - Keep best, reduce radius by 0.5x if improved, increase by 1.5x if not
        # - After some iterations, switch to local fine-tuning
        
        n_sphere = 15  # Points per sphere
        current_radius = 2.0
        min_radius = 1e-3
        max_radius = 4.0
        
        # Track consecutive failures to decide when to switch strategies
        consecutive_failures = 0
        max_failures = 5
        
        # Generate random directions
        def random_directions(n, dim):
            dirs = np.random.randn(n, dim)
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return dirs / norms
        
        # Main search loop
        while call_count + n_sphere <= budget and time.time() < deadline:
            # Pick a center (use best_x for simplicity, or cycle through centers)
            center = best_x.copy()
            
            # Generate points on sphere
            dirs = random_directions(n_sphere, dim)
            points = center[None, :] + current_radius * dirs
            
            improved = False
            for i in range(n_sphere):
                f = do_call(points[i])
                if f is None:
                    break
                if f < best_f:
                    improved = True
            
            if call_count >= budget or time.time() > deadline:
                break
            
            if improved:
                consecutive_failures = 0
                # Shrink radius
                current_radius *= 0.6
                if current_radius < min_radius:
                    current_radius = min_radius
            else:
                consecutive_failures += 1
                # Expand radius or try different center
                current_radius *= 1.5
                if current_radius > max_radius:
                    current_radius = max_radius
                
                # If too many failures, try a random restart
                if consecutive_failures >= max_failures:
                    consecutive_failures = 0
                    # Random restart
                    new_center = np.random.uniform(-4, 4, size=dim)
                    f = do_call(new_center)
                    if f is not None and f < best_f:
                        best_x = new_center.copy()
                        current_radius = 1.0
                    else:
                        # Stay at current best, try smaller radius
                        current_radius = 0.5
        
        # Phase 2: Local refinement using coordinate descent / Nelder-Mead style
        if call_count < budget and time.time() < deadline:
            # Use a simple local search: perturb each coordinate slightly
            local_radius = 0.1
            for coord in range(dim):
                while call_count < budget and time.time() < deadline and local_radius > 1e-4:
                    improved = False
                    for sign in [-1, 1]:
                        x_trial = best_x.copy()
                        x_trial[coord] += sign * local_radius
                        f = do_call(x_trial)
                        if f is None:
                            break
                        if f < best_f:
                            improved = True
                    if not improved:
                        local_radius *= 0.5
                    else:
                        # Keep going with same radius or slightly smaller
                        local_radius *= 0.8
        
        # Phase 3: Final fine-tuning with very small perturbations
        if call_count < budget and time.time() < deadline:
            # Try a few random fine perturbations
            n_fine = min(20, budget - call_count)
            for _ in range(n_fine):
                if call_count >= budget or time.time() > deadline:
                    break
                x_trial = best_x + np.random.randn(dim) * 0.01
                do_call(x_trial)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
