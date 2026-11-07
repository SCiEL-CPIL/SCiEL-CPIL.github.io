"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB Sphere (f1).
    Combines:
    1. Fast initial probe (center + axes) to get a good starting point quickly (AOCC).
    2. (1+1)-ES for robust global/local search.
    3. Powell's Method (coordinate-wise line search) for rapid convergence along
       hidden rotated axes, avoiding the geometric coupling of Nelder-Mead.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls_made = 0
        self._start_time = 0.0
        self._best_x = None
        self._best_f = np.inf

    def _evaluate(self, func, x):
        """Evaluate function with call counting and deadline guard."""
        if self._calls_made >= self.budget:
            return None
        if time.time() - self._start_time > 290:  # Safety margin
            return None
        self._calls_made += 1
        val = func(x)
        if np.isnan(val) or np.isinf(val):
            val = 1e10
        if val < self._best_f:
            self._best_f = val
            self._best_x = np.array(x, dtype=float)
        return val

    def _powell_line_search(self, func, x, direction, rng):
        """
        Custom quadratic line search along a direction.
        Uses 3 points to fit a quadratic and find the minimum.
        Returns new x and f(x).
        """
        x = np.array(x, dtype=float)
        fx = self._evaluate(func, x)
        if fx is None:
            return x, np.inf

        # Step 1: Determine step size
        # Try a small step to establish a direction of improvement or magnitude
        # We need to handle the case where the current point is already a local min along this direction
        # For Powell, we usually maintain a step size per direction.
        # Here we implement a simple adaptive step.
        
        # Initial step size estimate based on previous best or default
        # We don't have per-direction memory in this simple version, so we estimate.
        # A safe bet is to try a step proportional to the norm of the direction or a fixed small value.
        # Let's use a fixed small step for the first probe, then adapt.
        
        # Actually, standard Powell's method maintains step sizes. 
        # To keep it simple and robust within the call budget, we will use a 
        # "golden section" or "quadratic fit" approach that is self-contained per call.
        
        # However, doing a full line search from scratch every time is expensive in calls.
        # Let's use a simpler approach: 
        # 1. Take a step in direction.
        # 2. If better, take a larger step (golden section expansion).
        # 3. Fit quadratic to 3 best points found.
        
        # To save calls, we limit the line search to ~10-15 calls max.
        
        # Let's implement a simplified Powell step:
        # We maintain a step size for this direction in the caller? No, let's keep it local.
        
        # Strategy:
        # 1. x0 = x, f0 = fx
        # 2. x1 = x + h*direction, f1 = f(x1)
        # 3. If f1 > f0, reverse direction or shrink h.
        
        # Let's use a fixed small h initially, say 0.5 * ||direction||? No, direction is unit.
        # Let's try h = 1.0.
        
        h = 1.0
        x1 = np.clip(x + h * direction, -5, 5)
        f1 = self._evaluate(func, x1)
        if f1 is None:
            return x, np.inf
        
        if f1 < fx:
            # Improvement, try to go further
            x2 = np.clip(x + 2.0 * h * direction, -5, 5)
            f2 = self._evaluate(func, x2)
            if f2 is None:
                # Return best so far
                if f1 < fx:
                    return x1, f1
                return x, fx
            
            if f2 < f1:
                # Still improving, we have 3 points: x, x1, x2 (all improving)
                # Fit quadratic to (0, fx), (h, f1), (2h, f2)
                # t in [0, 2h]
                # f(t) = a*t^2 + b*t + c
                # c = fx
                # f1 = a*h^2 + b*h + fx
                # f2 = a*(2h)^2 + b*(2h) + fx = 4a*h^2 + 2b*h + fx
                
                # Solve for a, b
                # Let u = h
                # f1 - fx = a*u^2 + b*u
                # f2 - fx = 4a*u^2 + 2b*u
                # Multiply first by 2: 2(f1-fx) = 2a*u^2 + 2b*u
                # Subtract: (f2-fx) - 2(f1-fx) = 2a*u^2
                # a = (f2 - 2f1 + fx) / (2*u^2)
                
                denom = 2.0 * h * h
                if abs(denom) < 1e-12:
                    return x1, f1
                a = (f2 - 2.0 * f1 + fx) / denom
                b = (f1 - fx) / h - a * h
                
                # Vertex at t = -b / (2a)
                if a > 0:
                    t_min = -b / (2.0 * a)
                    # Clip t_min to [0, 2h]
                    t_min = np.clip(t_min, 0.0, 2.0 * h)
                    x_opt = np.clip(x + t_min * direction, -5, 5)
                    f_opt = self._evaluate(func, x_opt)
                    if f_opt is None:
                        # Return best known
                        if f2 < f1:
                            return x2, f2
                        return x1, f1
                    return x_opt, f_opt
                else:
                    # Parabola opens down or is flat, min is at boundary
                    if f2 < f1:
                        return x2, f2
                    return x1, f1
            else:
                # f2 >= f1, min is between x and x2, likely near x1
                # We have x, x1, x2. f1 is best.
                # Fit quadratic to x, x1, x2
                denom = 2.0 * h * h
                if abs(denom) < 1e-12:
                    return x1, f1
                a = (f2 - 2.0 * f1 + fx) / denom
                b = (f1 - fx) / h - a * h
                
                if a > 0:
                    t_min = -b / (2.0 * a)
                    t_min = np.clip(t_min, 0.0, 2.0 * h)
                    x_opt = np.clip(x + t_min * direction, -5, 5)
                    f_opt = self._evaluate(func, x_opt)
                    if f_opt is None:
                        return x1, f1
                    return x_opt, f_opt
                else:
                    return x1, f1
        else:
            # f1 >= fx, try smaller step or reverse?
            # If f1 > fx, the minimum might be behind us or we are at a local min.
            # Try half step
            h_half = 0.5 * h
            x_half = np.clip(x + h_half * direction, -5, 5)
            f_half = self._evaluate(func, x_half)
            if f_half is None:
                return x, fx
            
            if f_half < fx:
                # Improvement at half step
                # Fit quadratic to x, x_half, x1
                # Points: 0, 0.5h, h
                # Let's just use x_half if it's better, or try to refine
                # For simplicity, return x_half
                return x_half, f_half
            else:
                # No improvement, return original
                return x, fx

    def _powell_search(self, func, x0, max_calls, rng):
        """
        Powell's method implementation.
        """
        x = np.array(x0, dtype=float)
        fx = self._evaluate(func, x)
        if fx is None:
            return x, np.inf
        
        directions = [np.eye(self.dim)[i] for i in range(self.dim)]
        
        # Main loop
        while self._calls_made < self.budget - 10:
            if time.time() - self._start_time > 290:
                break
            
            old_x = x.copy()
            old_fx = fx
            min_fx = fx
            min_x = x.copy()
            min_dir_idx = 0
            
            # Pass 1: Line search along each direction
            for i in range(self.dim):
                if self._calls_made >= self.budget - 5:
                    break
                x, fx = self._powell_line_search(func, x, directions[i], rng)
                if fx < min_fx:
                    min_fx = fx
                    min_x = x.copy()
                    min_dir_idx = i
            
            # Check for convergence
            if np.linalg.norm(x - old_x) < 1e-8:
                break
            
            # Pass 2: Line search along the new direction (old_x to min_x)
            # New direction is min_x - old_x
            new_dir = min_x - old_x
            if np.linalg.norm(new_dir) > 1e-12:
                new_dir = new_dir / np.linalg.norm(new_dir)
                x2, fx2 = self._powell_line_search(func, min_x, new_dir, rng)
                
                if fx2 < min_fx:
                    # Replace the direction that gave the worst improvement in Pass 1
                    # In standard Powell, we replace the direction that had the largest increase in function value
                    # Here we simplify: replace the direction that gave the best improvement (min_dir_idx)
                    # Or better, replace the direction that was "least useful".
                    # Let's replace min_dir_idx with the new direction
                    directions[min_dir_idx] = new_dir
                    x = x2
                    fx = fx2
                    min_fx = fx
                    min_x = x.copy()
        
        return x, fx

    def _es_search(self, func, x0, sigma, max_calls, rng):
        """(1+1)-ES with 1/5 success rule."""
        x = np.array(x0, dtype=float)
        fx = self._evaluate(func, x)
        if fx is None:
            return x, np.inf

        for _ in range(max_calls - 1):
            if self._calls_made >= self.budget:
                break
            if time.time() - self._start_time > 290:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = self._evaluate(func, y)
            if fy is None:
                break

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.85

            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

        return x, fx

    def __call__(self, func):
        self._start_time = time.time()
        self._calls_made = 0
        self._best_x = None
        self._best_f = np.inf
        rng = np.random.default_rng(12345)

        # Phase 1: Fast Probe
        # Evaluate origin and axis points to get a good initial guess quickly
        # This helps AOCC by providing a reasonable b_i early on.
        probe_points = [np.zeros(self.dim)]
        for i in range(self.dim):
            for sign in [-1, 1]:
                pt = np.zeros(self.dim)
                pt[i] = sign * 3.0
                probe_points.append(pt)
        
        # Add a few random points
        for _ in range(5):
            probe_points.append(rng.uniform(-4, 4, self.dim))

        for pt in probe_points:
            if self._calls_made >= self.budget:
                break
            if time.time() - self._start_time > 290:
                break
            self._evaluate(func, pt)

        if self._best_x is None:
            # Fallback
            self._best_x = np.zeros(self.dim)
            self._best_f = self._evaluate(func, self._best_x)

        # Phase 2: (1+1)-ES for robust exploration
        # Use a portion of the budget for ES
        es_budget = self.budget // 2
        x_es, fx_es = self._es_search(func, self._best_x, 1.0, es_budget, rng)
        
        if fx_es < self._best_f:
            self._best_f = fx_es
            self._best_x = x_es

        # Phase 3: Powell's Method for rapid convergence
        # Use the remaining budget for Powell
        remaining = self.budget - self._calls_made
        if remaining > 10:
            x_powell, fx_powell = self._powell_search(func, self._best_x, remaining, rng)
            if fx_powell < self._best_f:
                self._best_f = fx_powell
                self._best_x = x_powell

        # Final check
        if self._best_x is None:
            self._best_x = np.zeros(self.dim)
            self._best_f = 0.0

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
