import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.time_limit = 300.0  # 330s kill, 30s safety margin

    def _check_time(self):
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > (self.time_limit - 5.0)

    def _evaluate(self, x):
        if self.calls >= self.budget:
            return None
        if self._check_time():
            return None
        self.calls += 1
        val = self.func(x)
        if val is not None and val < self.best_f:
            self.best_f = val
            self.best_x = x.copy()
        return val

    def _golden_section_1d(self, axis, x_center, search_radius):
        """
        Minimizes along a single axis using Golden Section Search.
        Updates self.best_x and self.best_f if better found.
        """
        if self.best_x is None:
            return

        gr = (np.sqrt(5) - 1) / 2  # 0.618...
        inv_gr = 1.0 / gr

        # Define search interval
        a = max(-5.0, x_center[axis] - search_radius)
        b = min(5.0, x_center[axis] + search_radius)

        if b - a < 1e-12:
            return

        # Initial points
        c = b - inv_gr * (b - a)
        d_point = a + inv_gr * (b - a)

        # Evaluate c
        x_c = x_center.copy()
        x_c[axis] = c
        f_c = self._evaluate(x_c)
        if f_c is None:
            return

        # Evaluate d
        x_d = x_center.copy()
        x_d[axis] = d_point
        f_d = self._evaluate(x_d)
        if f_d is None:
            return

        # Track local best
        local_best_val = min(f_c, f_d, self.best_f)
        local_best_pos = self.best_x[axis]
        if f_c < local_best_val:
            local_best_val = f_c
            local_best_pos = c
        if f_d < local_best_val:
            local_best_val = f_d
            local_best_pos = d_point

        # Iterative refinement
        for _ in range(30):  # Cap iterations to save budget
            if self.calls >= self.budget or self._check_time():
                break

            if f_c < f_d:
                b = d_point
                d_point = c
                f_d = f_c
                c = b - inv_gr * (b - a)
                x_c = x_center.copy()
                x_c[axis] = c
                f_c = self._evaluate(x_c)
                if f_c is None:
                    break
                if f_c < local_best_val:
                    local_best_val = f_c
                    local_best_pos = c
            else:
                a = c
                c = d_point
                f_c = f_d
                d_point = a + inv_gr * (b - a)
                x_d = x_center.copy()
                x_d[axis] = d_point
                f_d = self._evaluate(x_d)
                if f_d is None:
                    break
                if f_d < local_best_val:
                    local_best_val = f_d
                    local_best_pos = d_point

            if abs(b - a) < 1e-8:
                break

        # Update global best
        if local_best_val < self.best_f:
            self.best_f = local_best_val
            new_x = self.best_x.copy()
            new_x[axis] = local_best_pos
            self.best_x = new_x

    def _coordinate_descent_pass(self, search_radius):
        """
        Perform one full pass of coordinate descent.
        """
        if self.best_x is None:
            return

        for i in range(self.dim):
            if self.calls >= self.budget or self._check_time():
                return
            self._golden_section_1d(i, self.best_x, search_radius)

    def _random_restart(self):
        """
        Generate a random point within [-5, 5]^d.
        """
        return np.random.uniform(-5, 5, size=self.dim)

    def __call__(self, func):
        self.start_time = time.time()
        self.func = func
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = None

        d = self.dim

        # 1. Initial evaluation at origin
        x0 = np.zeros(d)
        f0 = self._evaluate(x0)
        if f0 is None:
            return np.zeros(d), 0.0

        # 2. Coordinate Descent Refinement
        # Strategy:
        # - Start with a large radius to ensure we are in the valley (or close).
        # - Shrink radius in subsequent passes for precision.
        # - Bent Cigar has a narrow valley. Coordinate descent is very effective
        #   for this because it decouples the axes. Once we are in the valley,
        #   optimizing one axis at a time is fast.
        
        # Pass 1: Wide radius to find the valley direction/center.
        # We don't know the shift, so a radius of 5.0 covers the whole box.
        # But since we start at 0, and min is in [-4,4], 5.0 is safe.
        # However, if the min is far, we might bounce. 
        # Let's try a few passes with decreasing radius.
        
        radii = [5.0, 2.5, 1.0, 0.5, 0.1, 0.01]
        
        for r in radii:
            if self.calls >= self.budget or self._check_time():
                break
            self._coordinate_descent_pass(r)
            
            # If we are stuck (no improvement in a pass), maybe we should restart?
            # But CD is deterministic. If it doesn't improve, it won't.
            # The risk is getting stuck in a local minima if the function is non-convex,
            # but Bent Cigar is unimodal. The only "trap" is if the valley is very thin
            # and we miss it in the first pass.
            
            # To mitigate missing the valley, if we don't improve significantly 
            # after the first wide pass, we could try a random restart.
            # However, with 500 calls, we have plenty of budget for multiple passes.
            
            # Let's add a simple check: if the best value hasn't improved much, 
            # and we have budget, try a random restart and refine.
            # But for Bent Cigar, the valley is long and thin. 
            # Starting at 0 is usually okay if the shift is not too large.
            # The shift is in [-4, 4].
            
            # If we are at 0, and min is at (4,4,4,4,4), the function value is high.
            # CD will move towards the min.
            
            # Let's just run the passes.
            # If we run out of budget, we stop.
            
            # Optional: If we have budget left and are not converging, do a random restart.
            # But convergence is key for AOCC. Early progress counts.
            # CD gives fast early progress.
            
            pass

        # 3. If we still have budget, do a final polish with very small radius
        # or random perturbations around the best point to ensure we are at the bottom.
        
        if self.calls < self.budget and not self._check_time():
            # Final fine-tuning
            for _ in range(3):
                if self.calls >= self.budget or self._check_time():
                    break
                self._coordinate_descent_pass(0.001)

        # Ensure we return a valid point
        if self.best_x is None:
            self.best_x = x0
            self.best_f = f0

        return self.best_x, self.best_f
