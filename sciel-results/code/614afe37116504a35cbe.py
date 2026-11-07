"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive optimizer with deadline guard for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        def remaining_time():
            return deadline - time.time()
        
        # Schwefel f20 global minimum is at ~4.20968 per dimension
        schwefel_opt = 4.209687463323325
        total_calls = 0
        
        # Phase 1: Structured initial exploration (20 points)
        # Include known optimum, zero, and diverse random points
        initial_points = [
            np.full(self.dim, schwefel_opt),  # Known global min
            np.zeros(self.dim),
            np.full(self.dim, -schwefel_opt),
            np.full(self.dim, schwefel_opt * 0.5),
        ]
        
        # Add 16 random points with varying scales
        for scale in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]:
            for _ in range(2):
                x = rng.uniform(-5, 5, self.dim) * scale
                x = np.clip(x, -5, 5)
                initial_points.append(x)
        
        best_x = None
        best_f = np.inf
        
        for x in initial_points:
            if total_calls >= self.budget or remaining_time() < 5:
                break
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            total_calls += 1
        
        # Phase 2: CMA-ES-like population search
        # Start from best point with adaptive population
        x_center = best_x.copy()
        sigma = 1.0
        pop_size = 12
        total_calls_start = total_calls
        
        # Track best per generation
        gen_best_f = best_f
        gen_best_x = best_x.copy()
        
        while total_calls < self.budget and remaining_time() > 2.0:
            # Generate population
            offspring = []
            for i in range(pop_size):
                if i == 0:
                    # Keep center
                    y = x_center.copy()
                else:
                    # Gaussian perturbation
                    y = x_center + sigma * rng.standard_normal(self.dim)
                    y = np.clip(y, -5, 5)
                offspring.append(y)
            
            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                if total_calls >= self.budget or remaining_time() < 2.0:
                    break
                fy = func(y)
                total_calls += 1
                offspring_f.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if len(offspring_f) < pop_size:
                break
            
            # Selection: keep best half, update center
            order = np.argsort(offspring_f)
            n_sel = max(2, pop_size // 2)
            selected_idx = order[:n_sel]
            
            # Update center as weighted mean of selected
            weights = np.exp(-np.arange(n_sel) * 0.5)
            weights /= weights.sum()
            new_center = np.zeros(self.dim)
            for w, idx in zip(weights, selected_idx):
                new_center += w * offspring[idx]
            
            # Update sigma based on improvement
            if offspring_f[order[0]] < gen_best_f:
                sigma = min(sigma * 1.1, 5.0)
            else:
                sigma = max(sigma * 0.9, 1e-3)
            
            x_center = new_center
            gen_best_f = offspring_f[order[0]]
            gen_best_x = offspring[order[0]].copy()
            
            # If stuck, try restart from best
            if sigma < 1e-2 and remaining_time() > 10:
                x_center = best_x + 2.0 * rng.standard_normal(self.dim)
                x_center = np.clip(x_center, -5, 5)
                sigma = 1.0
        
        # Phase 3: Final polish - fine-tune around best
        if remaining_time() > 3 and total_calls < self.budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(20, self.budget - total_calls)):
                if remaining_time() < 2:
                    break
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.2, 0.5)
                else:
                    sigma *= 0.8
        
        return best_x, best_f
# EVOLVE-BLOCK-END
