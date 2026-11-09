"""Simulated Annealing with adaptive cooling for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Simulated Annealing with adaptive cooling schedule, initialized from a broad uniform scout."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial broad uniform scout
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive parameters
        T = 5.0
        T_min = 1e-10
        step_size = 1.5
        
        # Tracking for adaptation
        stall_counter = 0
        stall_limit = 30
        recent_accepts = 0
        recent_total = 0
        window_len = 15

        # Track best improvement magnitude for scaling
        best_improve = 0.0

        for i in range(budget - 1):
            # Generate candidate with occasional larger jumps
            if rng.random() < 0.1:
                # Larger exploration step
                direction = rng.standard_normal(dim)
                y = np.clip(x + step_size * 2.0 * direction, -5, 5)
            else:
                direction = rng.standard_normal(dim)
                y = np.clip(x + step_size * direction, -5, 5)
            
            fy = func(y)
            
            accepted = False
            if fy < fx:
                accepted = True
            else:
                delta_f = fy - fx
                # Use a threshold to avoid unnecessary exp calls for huge deltas
                if delta_f < T * 5:
                    prob = np.exp(-delta_f / T)
                    if rng.random() < prob:
                        accepted = True
            
            if accepted:
                x = y
                fx = fy
                recent_accepts += 1
                if fx < best_fx:
                    improvement = best_fx - fx
                    best_improve = max(best_improve, improvement)
                    best_x = x.copy()
                    best_fx = fx
                    stall_counter = 0
                    # Cool down faster on improvement
                    T *= 0.97
                else:
                    T *= 0.995
            else:
                stall_counter += 1
                T *= 0.992

            recent_total += 1

            # Adaptive Step Size
            if recent_total >= window_len:
                accept_ratio = recent_accepts / recent_total
                if accept_ratio > 0.55:
                    step_size *= 1.12
                elif accept_ratio < 0.35:
                    step_size *= 0.88
                # Reset window
                recent_accepts = 0
                recent_total = 0
                # Clamp step size
                step_size = np.clip(step_size, 0.05, 4.0)

            # Restart mechanism for stagnation
            if stall_counter > stall_limit:
                x = best_x.copy()
                fx = best_fx
                T = max(T, 3.0) # Boost temperature to escape
                step_size = max(step_size, 0.8) # Ensure we can move
                stall_counter = 0

            if T < T_min:
                T = T_min

        return best_x, best_fx
# EVOLVE-BLOCK-END
