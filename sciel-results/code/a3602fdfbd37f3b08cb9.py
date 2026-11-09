"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start optimizer with adaptive (1+lambda)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Stratified multi-start exploration
        # Use quasi-random (Halton-like) sequence for better coverage than pure random
        n_init = min(30, budget // 8)
        
        # Generate stratified samples using deterministic low-discrepancy approach
        # Mix of scales to cover the domain
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            
            # Use different ranges to ensure coverage
            if i < n_init // 4:
                # Very broad: corners and edges
                x = rng.uniform(-4.8, 4.8, dim)
            elif i < n_init // 2:
                # Medium: most of the domain
                x = rng.uniform(-3.5, 3.5, dim)
            elif i < 3 * n_init // 4:
                # Tighter: center region where minima likely are
                x = rng.uniform(-2.0, 2.0, dim)
            else:
                # Very tight: around origin (common for shifted functions)
                x = rng.uniform(-0.5, 0.5, dim)
            
            evaluate(x)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Population-based local refinement with improved adaptation
        x = best_x.copy() if best_x is not None else rng.uniform(-1, 1, dim)
        fx = best_f

        sigma = 0.8  # Start with moderate step size
        pop_size = 8
        successes = []
        window_size = 15
        stagnation = 0
        max_stagnation = 40
        
        # Track sigma history for smoother adaptation
        sigma_history = [sigma]
        
        while calls < budget and time.time() < deadline:
            # Generate population of offspring
            improvements = 0
            for j in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break

                # Use slightly correlated perturbations for better exploration
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    improvements += 1
                    x, fx = y, fy
                    successes.append(1)
                    stagnation = 0
                else:
                    successes.append(0)
                    stagnation += 1

            # Trim success window
            if len(successes) > window_size:
                successes = successes[-window_size:]

            # Improved sigma adaptation using success rate with smoothing
            if len(successes) >= window_size:
                rate = sum(successes) / window_size
                # More aggressive adaptation
                if rate > 0.30:
                    sigma *= 1.20
                elif rate > 0.15:
                    sigma *= 1.05
                elif rate < 0.05:
                    sigma *= 0.75
                elif rate < 0.15:
                    sigma *= 0.90
                else:
                    sigma *= 0.98

            # Clamp sigma with wider range
            sigma = np.clip(sigma, 0.003, 5.0)
            sigma_history.append(sigma)
            if len(sigma_history) > 20:
                sigma_history = sigma_history[-20:]

            # Enhanced restart mechanism
            if stagnation > max_stagnation:
                if sigma < 0.10:
                    # Very small step: likely in a sharp local min, restart with larger sigma
                    x = best_x.copy()
                    fx = best_f
                    sigma = 2.5
                    stagnation = 0
                    successes = []
                    sigma_history = [sigma]
                else:
                    # Moderate step: try several larger jumps to escape plateau
                    found_improvement = False
                    for _ in range(min(8, budget - calls)):
                        if calls >= budget or time.time() > deadline:
                            break
                        # Try different jump scales
                        jump_scale = sigma * rng.uniform(1.5, 3.0)
                        y = x + jump_scale * rng.standard_normal(dim)
                        y = np.clip(y, -5, 5)
                        fy = evaluate(y)
                        if fy < fx:
                            x, fx = y, fy
                            stagnation = 0
                            found_improvement = True
                            break
                    
                    if not found_improvement:
                        # No improvement from jumps, try even larger jump or reset
                        if sigma > 1.0:
                            # Large sigma but no improvement: reset to best
                            x = best_x.copy()
                            fx = best_f
                            sigma = 3.0
                        else:
                            sigma *= 0.6
                        stagnation = 0
                    successes = []
                    sigma_history = [sigma]

        # Phase 3: Fine-tuning with adaptive small steps
        if calls < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            fine_sigma = 0.03
            fine_stagnation = 0
            max_fine_stagnation = 20

            while calls < budget and time.time() < deadline:
                y = x + fine_sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    fine_sigma *= 1.15
                    fine_stagnation = 0
                else:
                    fine_sigma *= 0.85
                    fine_stagnation += 1
                
                if fine_sigma < 0.0005 or fine_stagnation > max_fine_stagnation:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
