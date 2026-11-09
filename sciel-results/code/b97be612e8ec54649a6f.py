"""Rewritten program: coordinate-descent variant of Nelder-Mead for Rosenbrock valley."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Coordinate-descent Nelder-Mead with adaptive valley stretching for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Track best
        x_best = np.zeros(dim)
        fx_best = np.inf
        calls = 0

        def try_point(x):
            nonlocal calls, x_best, fx_best
            if calls >= budget:
                return None
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < fx_best:
                x_best = x_clipped.copy()
                fx_best = f
            return x_clipped, f

        # Phase 1: Sample a few diverse starting points to handle hidden shift/rotation
        candidates = [
            np.zeros(dim),
            np.ones(dim),
            -np.ones(dim),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        for c in candidates:
            if calls >= budget:
                break
            result = try_point(c)
            if result is None:
                break

        x = x_best.copy()
        fx = fx_best

        # Phase 2: Adaptive coordinate descent with valley-aware steps
        # For Rosenbrock, the valley runs along the direction where consecutive
        # coordinates are equal. We use correlated moves to follow the valley.
        
        # Step sizes: start moderately large
        steps = np.ones(dim) * 1.5
        min_step = 1e-7
        max_step = 10.0
        
        # Valley direction: for standard Rosenbrock, the valley is along
        # (1, 1, 1, ..., 1). But with rotation, we don't know the exact direction.
        # We'll adapt by using a mix of coordinate and diagonal moves.
        
        # Precompute some useful directions
        # Diagonal direction (for valley following)
        diag_dir = np.ones(dim) / np.sqrt(dim)
        # Anti-diagonal
        anti_diag_dir = np.ones(dim) * (-1.0) / np.sqrt(dim)
        
        # Alternating patterns
        alt_dir = np.array([1.0, -1.0, 1.0, -1.0, 1.0][:dim]) / np.sqrt(dim)
        
        # Coordinate descent with adaptive steps
        improved_count = 0
        no_improve_counter = 0
        
        while calls < budget:
            # Choose which direction to try: coordinate or valley-like
            # Use a schedule: early on, try valley directions more; later, coordinates
            progress = calls / budget
            use_valley = rng.random() < (1.0 - 0.5 * progress)
            
            if use_valley and calls < budget:
                # Try moves along valley-like directions
                for direction in [diag_dir, -diag_dir, alt_dir, -alt_dir]:
                    if calls >= budget:
                        break
                    
                    # Try a few step sizes
                    for step_scale in [steps[0], steps[0] * 0.5, steps[0] * 2.0]:
                        if calls >= budget:
                            break
                        
                        x_trial = x + step_scale * direction
                        x_trial = np.clip(x_trial, -5, 5)
                        
                        if np.allclose(x_trial, x):
                            continue
                        
                        f_trial = func(x_trial)
                        calls += 1
                        
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            if fx < fx_best:
                                x_best = x.copy()
                                fx_best = fx
                            improved_count += 1
                            no_improve_counter = 0
                            # Adapt steps
                            steps = np.clip(steps * 1.1, min_step, max_step)
                            break
                        else:
                            steps = np.clip(steps * 0.8, min_step, max_step)
            else:
                # Coordinate descent
                improved_this_pass = False
                for i in range(dim):
                    if calls >= budget:
                        break
                    
                    for sign in [1.0, -1.0]:
                        if calls >= budget:
                            break
                        
                        x_trial = x.copy()
                        x_trial[i] += sign * steps[i]
                        x_trial = np.clip(x_trial, -5, 5)
                        
                        if np.allclose(x_trial, x):
                            continue
                        
                        f_trial = func(x_trial)
                        calls += 1
                        
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            steps[i] = min(steps[i] * 1.2, max_step)
                            improved_this_pass = True
                            improved_count += 1
                            no_improve_counter = 0
                            if fx < fx_best:
                                x_best = x.copy()
                                fx_best = fx
                            break
                        else:
                            steps[i] = max(steps[i] * 0.7, min_step)
                
                if not improved_this_pass:
                    no_improve_counter += 1
                    # Shrink all steps
                    steps *= 0.6
                    if np.all(steps < min_step):
                        break
                    
                    # Try a random perturbation to escape
                    if no_improve_counter % 3 == 0 and calls < budget - 5:
                        x_perturb = x + rng.uniform(-0.5, 0.5, dim)
                        x_perturb = np.clip(x_perturb, -5, 5)
                        f_perturb = func(x_perturb)
                        calls += 1
                        if f_perturb < fx:
                            x = x_perturb
                            fx = f_perturb
                            if fx < fx_best:
                                x_best = x.copy()
                                fx_best = fx
                            improved_count += 1
                            no_improve_counter = 0
            if calls >= budget:
                break

        # Phase 3: Final fine-grained polish
        if calls < budget:
            # Use very small steps for precision
            polish_steps = np.ones(dim) * 0.05
            for _ in range(min(30, budget - calls)):
                if calls >= budget:
                    break
                
                improved = False
                for i in range(dim):
                    if calls >= budget:
                        break
                    
                    for sign in [1.0, -1.0]:
                        if calls >= budget:
                            break
                        
                        x_trial = x.copy()
                        x_trial[i] += sign * polish_steps[i]
                        x_trial = np.clip(x_trial, -5, 5)
                        
                        if np.allclose(x_trial, x):
                            continue
                        
                        f_trial = func(x_trial)
                        calls += 1
                        
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            improved = True
                            if fx < fx_best:
                                x_best = x.copy()
                                fx_best = fx
                            break
                
                if not improved:
                    polish_steps *= 0.5
                    if np.all(polish_steps < min_step):
                        break

        return x_best, fx_best
# EVOLVE-BLOCK-END
