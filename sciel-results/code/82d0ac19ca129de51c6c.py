"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and strict wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Counting wrapper
        call_count = 0

        def counted_func(x):
            nonlocal call_count
            call_count += 1
            return func(np.clip(x, -5.0, 5.0))

        def remaining_time():
            return deadline - time.time()

        # Phase 1: Diverse initial evaluation
        # Use a set of well-spread points including center, vertices, and random
        starts = []
        starts.append(np.zeros(dim))
        
        # Vertices of the box (mapped to [-4, 4])
        for i in range(2 ** dim):
            v = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            v = v * 2.0 - 1.0  # map to [-1, 1]
            v = v * 4.0  # map to [-4, 4]
            starts.append(v)
        
        # Random points
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        best_x = None
        best_f = np.inf
        evaluated = []

        for x0 in starts:
            if call_count >= budget or remaining_time() < 10.0:
                break
            f0 = counted_func(x0)
            evaluated.append((x0.copy(), f0))
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = counted_func(best_x)
            evaluated.append((best_x.copy(), best_f))

        # Phase 2: Local refinement using (1+1)-ES from the best starting points
        # Sort evaluated points by fitness and try to refine the top few
        evaluated.sort(key=lambda t: t[1])
        
        tried_starts = set()
        for x0, f0 in evaluated:
            if call_count >= budget or remaining_time() < 15.0:
                break
            
            # Skip if we've already tried a nearby point
            key = tuple(np.round(x0, 2))
            if key in tried_starts:
                continue
            tried_starts.add(key)
            
            # Run (1+1)-ES refinement from this point
            x = x0.copy()
            fx = f0
            sigma = 0.5
            min_sigma = 1e-12
            success_ratio = 0.5  # target success rate
            
            # Adaptive sigma based on success rate
            while call_count < budget and remaining_time() > 5.0:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = counted_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.05
                    success_ratio = 0.9 * success_ratio + 0.1
                else:
                    sigma *= 0.95
                    success_ratio = 0.9 * success_ratio + 0.0
                    if sigma < min_sigma:
                        break
        
        # Phase 3: If budget remains, do a (1+1)-ES refinement from the best point
        if call_count < budget and remaining_time() > 10.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            min_sigma = 1e-12
            success_ratio = 0.5  # target success rate
            
            # Adaptive sigma based on success rate
            while call_count < budget and remaining_time() > 5.0:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = counted_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.05
                    success_ratio = 0.9 * success_ratio + 0.1
                else:
                    sigma *= 0.95
                    success_ratio = 0.9 * success_ratio + 0.0
                    if sigma < min_sigma:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
