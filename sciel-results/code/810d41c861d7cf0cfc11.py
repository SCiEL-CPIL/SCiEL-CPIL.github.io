"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma (1/5 rule), multi-start initialization, and local escape."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 0: Multi-start initialization (small budget)
        # Try origin and a few random points to find a good basin
        init_candidates = [np.zeros(d)]
        for _ in range(4):
            init_candidates.append(rng.uniform(-4.5, 4.5, d))
        
        for x0 in init_candidates:
            if check_deadline():
                break
            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            total_calls += 1
            update_best(x0, f0)
        
        # Start ES from the best initial point
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        
        # Phase 1: Main (1+1)-ES with adaptive sigma and local escape
        # 1/5 rule: track success rate over last 5 steps
        success_window = []  # Last 5 success flags
        no_improve = 0
        max_no_improve = 40  # Stall threshold for local escape
        escape_count = 0
        max_escapes = 8
        
        while total_calls < budget - 20 and not check_deadline():
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            total_calls += 1
            
            success = fy < fx
            success_window.append(success)
            if len(success_window) > 5:
                success_window.pop(0)
            
            if success:
                x, fx = y, fy
                no_improve = 0
                update_best(x, fx)
            else:
                no_improve += 1
            
            # Adaptive sigma via 1/5 rule
            if len(success_window) == 5:
                success_rate = sum(success_window) / 5.0
                if success_rate > 0.5:
                    sigma *= 1.3
                elif success_rate < 0.1:
                    sigma *= 0.75
                # else: keep sigma unchanged
            
            # Local escape: if stalled, try a kick from the best point
            if no_improve >= max_no_improve and escape_count < max_escapes:
                escape_count += 1
                # Kick: larger step from current best in a random direction
                kick_dir = rng.standard_normal(d)
                kick_dir /= (np.linalg.norm(kick_dir) + 1e-12)
                kick_size = max(sigma * 3, 0.5)
                y_kick = np.clip(best_x + kick_size * kick_dir, -5, 5)
                fy_kick = func(y_kick)
                total_calls += 1
                
                if fy_kick < fx:
                    x, fx = y_kick, fy_kick
                    update_best(x, fx)
                    sigma = max(sigma * 0.8, 0.3)
                else:
                    # Keep current x, just reset stall counter
                    sigma = max(sigma * 0.7, 0.2)
                
                no_improve = 0
                success_window.clear()
        
        # Phase 2: Final polish around the best point
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.2
            
            remaining_calls = budget - total_calls
            polish_steps = min(20, remaining_calls)
            
            for _ in range(polish_steps):
                if check_deadline():
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1
                else:
                    sigma *= 0.85
                
                if sigma < 1e-9:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
