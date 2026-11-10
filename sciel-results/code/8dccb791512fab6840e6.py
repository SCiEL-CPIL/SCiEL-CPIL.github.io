import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = 0
        self.best_x = None
        self.best_f = np.inf
        self.func = None

    def __call__(self, func):
        self.start_time = time.time()
        self.func = func
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

        # 1. Initialization: Find a good start
        # Use 10 random points to handle hidden shift
        rng = np.random.default_rng(42)
        for _ in range(10):
            x = rng.uniform(-5, 5, self.dim)
            f = self._eval(x)
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
        
        x = self.best_x
        f = self.best_f
        
        # 2. Coordinate Descent with Line Search
        while self.calls < self.budget - 10:
            if time.time() - self.start_time > 280:
                break
            
            x_start_cycle = x.copy()
            f_start_cycle = f
            
            for i in range(self.dim):
                if self.calls >= self.budget - 10:
                    break
                if time.time() - self.start_time > 280:
                    break
                
                # Line search along dimension i
                direction = np.zeros(self.dim)
                direction[i] = 1.0
                
                x, f = self._line_search(x, f, direction)
                
            # Check for convergence
            if abs(f_start_cycle - f) < 1e-12:
                break

        return self.best_x, self.best_f

    def _eval(self, x):
        self.calls += 1
        return self.func(x)

    def _line_search(self, x, f_x, direction):
        # Explore phase
        best_alpha = 0.0
        best_f = f_x
        
        # Try a few steps
        for alpha in [1.0, 2.0, 4.0, -1.0, -2.0, -4.0]:
            if self.calls >= self.budget - 5:
                break
            x_try = x + alpha * direction
            # Clip to box? The func does it, but we can avoid out-of-bounds if we want?
            # The task says func clips. So we can pass any x.
            f_try = self._eval(x_try)
            if f_try < best_f:
                best_f = f_try
                best_alpha = alpha
        
        if best_alpha == 0.0:
            return x, f_x
        
        # Refine phase: Golden Section
        if best_alpha > 0:
            a = 0.0
            b = best_alpha * 2.0
            f_a = f_x
            f_b = self._eval(x + b * direction)
        else:
            a = best_alpha * 2.0
            b = 0.0
            f_a = self._eval(x + a * direction)
            f_b = f_x
            
        # Ensure a < b
        if a > b:
            a, b = b, a
            f_a, f_b = f_b, f_a
            
        phi = (np.sqrt(5) - 1) / 2
        
        # 5 iterations of Golden Section
        for _ in range(5):
            if self.calls >= self.budget - 2:
                break
            c = b - phi * (b - a)
            d = a + phi * (b - a)
            
            f_c = self._eval(x + c * direction)
            f_d = self._eval(x + d * direction)
            
            if f_c < f_d:
                b = d
                f_b = f_d
            else:
                a = c
                f_a = f_c
                
        mid = (a + b) / 2
        x_best = x + mid * direction
        f_best = self._eval(x_best)
        
        if f_best < self.best_f:
            self.best_f = f_best
            self.best_x = x_best.copy()
            
        return x_best, f_best
