"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= self.budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            calls_used += 1
            return val

        # Phase 1: Broad exploration with several starting points
        # Use a grid-like spread plus random points for initial coverage
        n_starts = min(10, self.budget // 50)  # At most 10 starts, each gets ~50 calls
        
        # Generate diverse starting points
        start_points = []
        # Center
        start_points.append(np.zeros(self.dim))
        # Axis-aligned points
        for i in range(self.dim):
            for sign in [-1, 1]:
                p = np.zeros(self.dim)
                p[i] = sign * 3.0
                start_points.append(p)
        # Random points
        while len(start_points) < n_starts:
            start_points.append(rng.uniform(-4.5, 4.5, self.dim))
        start_points = start_points[:n_starts]

        # Evaluate all starting points
        for sp in start_points:
            if calls_used >= self.budget:
                break
            fx = safe_call(sp)
            if fx is not None and fx < best_f:
                best_f = fx
                best_x = sp.copy()

        # Phase 2: Local refinement from the best starting point using (1+1)-ES
        if best_x is not None and calls_used < self.budget:
            x = best_x.copy()
            fx = best_f
            sigma = 1.5
            success_count = 0
            total_count = 0
            
            while calls_used < self.budget:
                # Time check
                if time.time() > deadline:
                    break
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(self.dim)
                fy = safe_call(y)
                if fy is None:
                    break
                
                total_count += 1
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    success_count += 1
                    # Increase step size on success
                    sigma *= 1.3
                    # Update global best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Decrease step size on failure
                    sigma *= 0.85
                
                # Adaptive sigma control: if too many successes, increase; if too many failures, decrease
                if total_count >= 20:
                    rate = success_count / total_count
                    if rate > 0.4:
                        sigma *= 1.1
                    elif rate < 0.2:
                        sigma *= 0.9
                    success_count = 0
                    total_count = 0
                
                # If sigma becomes very small, try a restart from a nearby point
                if sigma < 0.01:
                    # Try a few random perturbations
                    for _ in range(3):
                        if calls_used >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                        pert = x + 0.5 * rng.standard_normal(self.dim)
                        fp = safe_call(pert)
                        if fp is None:
                            break
                        if fp < fx:
                            x = pert.copy()
                            fx = fp
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            sigma = 0.5
                            break
                    else:
                        break  # No improvement, exit local search
        
        # Phase 3: If budget remains, do additional random restarts from best_x neighborhood
        if best_x is not None and calls_used < self.budget:
            for _ in range(min(5, (self.budget - calls_used) // 10)):
                if time.time() > deadline:
                    break
                # Random perturbation around best
                scale = 0.3 + 0.5 * rng.random()
                candidate = best_x + scale * rng.standard_normal(self.dim)
                fc = safe_call(candidate)
                if fc is not None and fc < best_f:
                    best_f = fc
                    best_x = candidate.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
