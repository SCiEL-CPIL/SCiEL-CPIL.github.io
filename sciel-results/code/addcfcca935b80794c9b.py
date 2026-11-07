import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Time limit safety margin: Evaluator kills at 330s. 
        # We stop at 320s to be safe.
        time_limit = 320.0
        
        # State
        self.best_x = np.zeros(n)
        self.best_f = np.inf
        self.calls_made = 0
        
        def call_func(x):
            if self.calls_made >= budget:
                return np.inf
            if time.time() - self.start_time > time_limit:
                return np.inf
            fx = func(x)
            self.calls_made += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Global Exploration using Latin Hypercube Sampling (LHS)
        # This provides better coverage of the [-5, 5]^n space than a simple grid or random sampling.
        # We use 50 points for LHS to get a good initial spread.
        num_lhs = 50
        lhs_points = []
        
        # Generate LHS points
        # For each dimension, we divide the range [-5, 5] into num_lhs bins.
        # We pick one random point from each bin in each dimension.
        # This ensures that we sample across the entire range in every dimension.
        
        # Simple LHS generation
        # 1. Create a permutation matrix of size num_lhs x n
        # 2. For each dimension j, permute the indices 0..num_lhs-1
        # 3. For each sample i, the j-th coordinate is in bin perm[j][i]
        
        perms = [np.arange(num_lhs) for _ in range(n)]
        for j in range(n):
            rng.shuffle(perms[j])
            
        for i in range(num_lhs):
            x = np.zeros(n)
            for j in range(n):
                # Bin index for sample i in dimension j
                bin_idx = perms[j][i]
                # Random point within that bin
                # Bin width is 10.0 / num_lhs
                # Bin range is [-5 + bin_idx * w, -5 + (bin_idx + 1) * w)
                w = 10.0 / num_lhs
                x[j] = -5.0 + bin_idx * w + rng.uniform(0, w)
            lhs_points.append(x)
            
        # Evaluate LHS points
        for x in lhs_points:
            if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                break
            call_func(x)
            
        # Phase 2: Multi-Start (1+8) Evolution Strategy
        # We will run a (1+8) ES with adaptive step size.
        # The strategy maintains a "best" point and generates 8 offspring per iteration.
        # If any offspring is better, it becomes the new best.
        # Step size is adapted based on the success rate (target 20%).
        
        # Initialize ES from the best point found in Phase 1
        current_x = self.best_x.copy()
        current_f = self.best_f
        
        # Initial step size
        # A reasonable initial step size for a 10-unit box is around 1.0 or 2.0
        # Let's start with 1.0
        sigma = 1.0
        
        # Success rate target
        target_success_rate = 0.2
        # Adaptation rate
        adapt_rate = 1.1
        
        # Population size for offspring
        pop_size = 8
        
        # Main ES loop
        while self.calls_made < budget and time.time() - self.start_time < time_limit:
            # Generate offspring
            best_offspring_x = None
            best_offspring_f = np.inf
            num_success = 0
            
            for _ in range(pop_size):
                # Generate random direction
                direction = rng.standard_normal(n)
                direction /= np.linalg.norm(direction)
                
                # Generate offspring
                y = current_x + sigma * direction
                y = np.clip(y, -5.0, 5.0)
                
                # Evaluate offspring
                fy = call_func(y)
                
                if fy < best_offspring_f:
                    best_offspring_f = fy
                    best_offspring_x = y.copy()
                
                if fy < current_f:
                    num_success += 1
                    
                # Check time/budget
                if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                    break
            
            # Update current best if an offspring was better
            if best_offspring_f < current_f:
                current_x = best_offspring_x
                current_f = best_offspring_f
                # Update global best
                if current_f < self.best_f:
                    self.best_f = current_f
                    self.best_x = current_x.copy()
            
            # Adapt step size
            success_rate = num_success / pop_size
            if success_rate > target_success_rate:
                sigma *= adapt_rate
            elif success_rate < target_success_rate:
                sigma /= adapt_rate
            
            # Safety check: if sigma becomes too small or too large, reset or adjust
            if sigma < 1e-6:
                break
            if sigma > 10.0:
                sigma = 10.0

        return self.best_x, self.best_f
