import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = 0.0

    def _run_11_es(self, func, budget, start_x, start_sigma):
        """
        Runs a (1+1)-ES with momentum for a specific budget.
        """
        dim = self.dim
        x = start_x.copy()
        f_x = func(x)
        self.calls += 1
        run_best_x = x.copy()
        run_best_f = f_x
        if f_x < self.best_f:
            self.best_f = f_x
            self.best_x = x.copy()

        sigma = start_sigma
        momentum = np.zeros(dim)
        momentum_beta = 0.5 # Momentum decay
        step_size_factor = 1.0
        
        # Simple (1+1)-ES parameters
        # Success probability target for sigma adaptation
        # We will use a simple rule: if step is accepted, increase sigma slightly, else decrease.
        # Or use a more standard adaptation.
        # Let's use the standard (1+1)-ES sigma update: 
        # sigma_{t+1} = sigma_t * exp(c_sigma * (1{success} - 1/p))
        # For (1+1), p is often 1/5 or similar, but simple binary is easier.
        # Let's try: if accepted, sigma *= 1.1, else sigma /= 1.1.
        
        while self.calls < budget and time.time() < self.deadline:
            # Generate candidate
            z = np.random.randn(dim)
            # Apply momentum
            # v = beta * v + (1-beta) * z
            # x_new = x + sigma * (v / norm(v) * norm(z) ?) 
            # Standard momentum: v = beta * v + (1-beta) * step_dir
            # Here we can just use z as the step direction and scale it.
            # Let's use: step = sigma * (momentum + z)
            # But we need to normalize to keep step size consistent.
            
            # Alternative: 
            # step_dir = momentum + z
            # x_new = x + sigma * step_dir / np.linalg.norm(step_dir) * np.linalg.norm(z)
            # This keeps the expected step size proportional to sigma.
            
            step_dir = momentum + z
            norm_dir = np.linalg.norm(step_dir)
            if norm_dir > 1e-10:
                step = sigma * (step_dir / norm_dir) * np.linalg.norm(z)
            else:
                step = sigma * z
                
            # Update momentum
            momentum = momentum_beta * momentum + (1 - momentum_beta) * (step / sigma)
            
            x_new = x + step
            
            # Handle boundaries
            # Clip to [-5, 5]
            x_new_clipped = np.clip(x_new, -5, 5)
            
            f_new = func(x_new_clipped)
            self.calls += 1
            
            if f_new < f_x:
                # Success
                x = x_new_clipped
                f_x = f_new
                # Adapt sigma
                sigma *= 1.1
                # Update best
                if f_x < self.best_f:
                    self.best_f = f_x
                    self.best_x = x.copy()
                if f_x < run_best_f:
                    run_best_f = f_x
                    run_best_x = x.copy()
            else:
                # Failure
                sigma /= 1.1
                # Keep x same
                # Optional: if we hit boundary and failed, maybe reflect?
                # For now, just keep x.
                
            # Prevent sigma from becoming too small or too large
            if sigma < 1e-6:
                sigma = 1e-6
            if sigma > 10.0:
                sigma = 10.0

        return run_best_x, run_best_f

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 280
        
        # Strategy: Multiple (1+1)-ES runs with momentum
        # The Lunacek function is bi-modal. 
        # We will use several restarts to increase the chance of finding the global minimum.
        
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts
        
        # Initial points
        # We can use a few strategic points and random ones.
        # The global minimum of Lunacek is at the origin (in original coords), 
        # but it's shifted/rotated.
        # Random uniform sampling is robust.
        
        for i in range(n_restarts):
            if self.calls >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            # Random start
            mean = np.random.uniform(-4, 4, self.dim)
            sigma = 2.0
            
            # Run (1+1)-ES
            self._run_11_es(func, budget_per_restart, mean, sigma)
            
        # If we have budget left, do a final polish with the best point found
        if self.calls < self.budget and time.time() < self.deadline:
            remaining = self.budget - self.calls
            # Use a smaller sigma for polishing
            self._run_11_es(func, remaining, self.best_x.copy(), 0.5)

        return self.best_x, self.best_f
