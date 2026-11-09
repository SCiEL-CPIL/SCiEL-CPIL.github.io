"""Simulated Annealing for BBOB f7 (Step ellipsoidal) with plateau-aware adaptation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with plateau-aware step adaptation for step ellipsoidal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_left():
            return deadline - time.time()

        # Phase 1: Broad exploration with random sampling
        n_initial = min(20, budget // 5)
        for i in range(n_initial):
            if calls_made >= budget or time_left() < 10.0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Simulated Annealing with multiple restarts
        # The step ellipsoidal function has flat plateaus and sharp drops.
        # SA with appropriate temperature can escape plateaus via uphill moves.
        
        n_sa_runs = 8
        calls_per_run = max(20, (budget - calls_made) // n_sa_runs)
        
        for run in range(n_sa_runs):
            if calls_made >= budget or time_left() < 15.0:
                break
            
            # Choose starting point
            if run == 0 and best_x is not None:
                x = best_x.copy()
                fx = best_f
            elif run % 2 == 0 and best_x is not None:
                # Start near best with moderate perturbation
                x = best_x + 2.0 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Fresh random start
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # SA parameters
            T_init = 5.0  # Initial temperature (function value scale)
            T_min = 0.01
            cooling_rate = 0.985
            T = T_init
            
            # Adaptive step size
            sigma = 1.5
            accept_count = 0
            total_count = 0
            plateau_steps = 0
            max_plateau = 40
            
            for step in range(calls_per_run):
                if calls_made >= budget or time_left() < 5.0:
                    break
                
                # Generate trial point with adaptive step size
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                total_count += 1
                
                delta_f = fy - fx
                
                if delta_f < 0:
                    # Improvement: always accept
                    x = y
                    fx = fy
                    accept_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    plateau_steps = 0
                    # Slightly increase step on success
                    sigma *= 1.05
                else:
                    # Uphill move: accept with probability exp(-delta_f / T)
                    if T > T_min:
                        prob = np.exp(-delta_f / T)
                        if rng.random() < prob:
                            x = y
                            fx = fy
                            accept_count += 1
                            plateau_steps = 0
                        else:
                            plateau_steps += 1
                    else:
                        plateau_steps += 1
                
                # Cool down
                T *= cooling_rate
                
                # Adapt step size based on acceptance rate
                if total_count > 10:
                    accept_rate = accept_count / total_count
                    if accept_rate > 0.6:
                        sigma *= 1.1  # Too easy, increase step
                    elif accept_rate < 0.1:
                        sigma *= 0.85  # Too hard, decrease step
                    # Reset counters periodically
                    if total_count % 20 == 0:
                        accept_count = 0
                        total_count = 0
                
                # Handle plateaus: if stuck, try a larger jump
                if plateau_steps > max_plateau:
                    # Large random jump to escape plateau
                    y = x + 3.0 * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        plateau_steps = 0
                        sigma = max(sigma, 1.5)
                    else:
                        # Try even larger jump
                        y = x + 5.0 * rng.standard_normal(dim)
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_made += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        plateau_steps = 0
                        sigma = max(sigma, 2.0)
                
                sigma = np.clip(sigma, 0.05, 8.0)

        # Phase 3: Fine-grained local search around best point
        if best_x is not None and calls_made < budget and time_left() > 10.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.2
            polish_steps = min(80, budget - calls_made)
            for _ in range(polish_steps):
                if calls_made >= budget or time_left() < 3.0:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.92
                sigma = np.clip(sigma, 0.005, 1.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
