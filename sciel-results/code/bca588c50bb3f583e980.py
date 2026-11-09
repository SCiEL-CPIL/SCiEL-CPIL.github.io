import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_scan(self, func, n, dim, deadline, calls_left):
        """Perform a Sobol scan to find a good starting point."""
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            u = sampler.random(n)
        except Exception:
            rng = np.random.default_rng(42)
            u = rng.random((n, dim))
        
        pts = -5.0 + 10.0 * u
        pts = np.clip(pts, -5, 5)
        
        best_val = float('inf')
        best_x = None
        
        for i in range(n):
            if time.time() >= deadline or calls_left[0] <= 0:
                break
            x = pts[i]
            fx = func(x)
            calls_left[0] -= 1
            if fx < best_val:
                best_val = fx
                best_x = x.copy()
        
        return best_x, best_val

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        budget = self.budget
        calls_left = [budget]

        def guarded(x):
            if calls_left[0] <= 0:
                return float('inf')
            calls_left[0] -= 1
            return float(func(x))

        rng = np.random.default_rng(12345)

        # Phase 1: Sobol Scan
        n_scan = min(50, budget // 5)
        if n_scan < 10:
            n_scan = 10
            
        best_x, best_fx = self._sobol_scan(guarded, n_scan, dim, deadline, calls_left)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = guarded(best_x)

        # Phase 2: Adaptive Coordinate Descent with momentum
        # For Bent Cigar, the function is f(x) = (x0 - 1)^2 + 100 * sum((xi - 1)^2 for i>0)
        # Actually standard bent cigar: f(x) = (x0 - 1)^2 + 100 * ||x_{1:} - 1||^2
        # The valley is along x0. We need to move fast along x0 and slow along others.
        
        x = best_x.copy()
        fx = best_fx
        
        # Estimate local curvature / scale for each dimension
        # Use a small step to probe
        probe_sigma = 0.1
        scales = np.ones(dim)
        
        for d in range(dim):
            if calls_left[0] <= 0 or time.time() >= deadline:
                break
            x_plus = x.copy()
            x_plus[d] += probe_sigma
            x_plus = np.clip(x_plus, -5, 5)
            fp = guarded(x_plus)
            
            x_minus = x.copy()
            x_minus[d] -= probe_sigma
            x_minus = np.clip(x_minus, -5, 5)
            fm = guarded(x_minus)
            
            # Approximate second derivative / curvature
            # If curvature is high, scale down step. If low, scale up.
            # f(x+e) - 2f(x) + f(x-e) approx f''(x) * e^2
            # But we don't know the sign of gradient easily without more info.
            # Let's just use the difference to estimate sensitivity.
            # If |fp - fm| is large, we are on a steep slope.
            # For coordinate descent, we want steps proportional to 1/sqrt(curvature).
            # Simple heuristic: if function changes a lot, reduce step.
            delta_f = abs(fp - fm)
            # Normalize delta_f
            # If delta_f is small, we can take larger steps.
            # If delta_f is large, we should take smaller steps.
            # Let's map delta_f to a scale factor.
            # Assume typical delta_f is around 1.
            if delta_f < 1e-6:
                scales[d] = 1.0
            else:
                # Inverse relationship
                scales[d] = np.clip(1.0 / np.sqrt(delta_f), 0.1, 10.0)
        
        # Initialize steps based on scales
        steps = 0.5 * scales
        momentum = np.zeros(dim)
        beta = 0.9
        
        while calls_left[0] > 0 and time.time() < deadline:
            improved = False
            
            # Coordinate descent cycle
            for d in range(dim):
                if calls_left[0] <= 0 or time.time() >= deadline:
                    break
                
                # Try positive step
                x_trial = x.copy()
                x_trial[d] += steps[d]
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = guarded(x_trial)
                
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    improved = True
                    # Update momentum
                    momentum[d] = beta * momentum[d] + (1 - beta) * steps[d]
                    # Increase step if we made progress
                    steps[d] *= 1.1
                else:
                    # Try negative step
                    x_trial = x.copy()
                    x_trial[d] -= steps[d]
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = guarded(x_trial)
                    
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        improved = True
                        # Update momentum
                        momentum[d] = beta * momentum[d] - (1 - beta) * steps[d]
                        # Increase step if we made progress
                        steps[d] *= 1.1
                    else:
                        # No improvement, shrink step
                        steps[d] *= 0.8
                        # Decay momentum
                        momentum[d] *= beta
        
            # If no improvement in a full cycle, try a random perturbation
            if not improved:
                # Add a small random noise to escape local minima (though BC is unimodal)
                # Or just reduce all steps
                steps *= 0.9
                if np.all(steps < 1e-6):
                    break

        return x, fx
