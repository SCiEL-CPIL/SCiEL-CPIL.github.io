import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, leave 10s margin
        
        # Wrapper to count calls and track best
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration
            self.calls += 1
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # 1. Broad exploration with Halton sequence for AOCC
        def halton(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result

        # Generate Halton points for early coverage
        num_halton = 50
        halton_points = []
        for i in range(1, num_halton + 1):
            point = np.array([halton(i, base) for base in range(2, 2 + self.dim)])
            # Map [0,1]^d to [-5,5]^d
            point = point * 10.0 - 5.0
            halton_points.append(point)
            
        # Add some deterministic points
        init_points = [
            np.zeros(self.dim),
            np.ones(self.dim) * 2,
            np.ones(self.dim) * -2,
            np.ones(self.dim) * 1,
            np.ones(self.dim) * -1,
            np.ones(self.dim) * 3,
            np.ones(self.dim) * -3
        ]
        
        all_init_points = init_points + halton_points
        best_init_x = None
        best_init_f = float('inf')
        
        for p in all_init_points:
            if self.calls >= self.budget or time.time() > deadline:
                break
            try:
                fp = eval_func(p)
                if fp < best_init_f:
                    best_init_f = fp
                    best_init_x = p.copy()
            except StopIteration:
                break
                
        if best_init_x is None:
            return np.zeros(self.dim), float('inf')

        # 2. Adaptive (1+1)-ES with 1/5 success rule
        x = best_init_x.copy()
        f_x = best_init_f
        sigma = 1.0  # Initial step size
        
        # 1/5 rule parameters
        success_threshold = 0.2  # 1/5 = 0.2
        window_size = 10
        successes = []
        
        while self.calls < self.budget and time.time() < deadline:
            # Generate candidate
            direction = np.random.randn(self.dim)
            direction /= np.linalg.norm(direction)
            x_trial = x + sigma * direction
            
            try:
                f_trial = eval_func(x_trial)
            except StopIteration:
                break
            
            # Track success
            if f_trial < f_x:
                successes.append(1)
                x = x_trial
                f_x = f_trial
            else:
                successes.append(0)
            
            # Apply 1/5 rule every window_size steps
            if len(successes) >= window_size:
                recent_successes = successes[-window_size:]
                success_rate = np.mean(recent_successes)
                
                if success_rate > success_threshold:
                    sigma *= 1.2  # Increase step size
                elif success_rate < success_threshold:
                    sigma *= 0.8  # Decrease step size
                
                successes = []
            
            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 0.01, 5.0)

        return self.best_x, self.best_f
