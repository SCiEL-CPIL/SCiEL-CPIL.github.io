"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Multi-start (1+1)-ES + Rastrigin-aware refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s kill

        # Track best
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start (1+1)-ES
        # We use 4 restarts. First one at center, others random.
        # Budget allocation: ~250 calls for this phase.
        n_restarts = 4
        calls_per_restart = 60  # 4 * 60 = 240 calls + initial evals
        
        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            # Start point
            if r == 0:
                x = np.zeros(dim)
            else:
                # Sample from a wide range to catch different basins
                x = rng.uniform(-4.5, 4.5, dim)

            fx = eval_func(x)
            
            # (1+1)-ES parameters
            sigma = 1.5
            window = 10
            recent = [False] * window
            idx = 0

            for _ in range(calls_per_restart):
                if calls >= budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)

                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                else:
                    recent[idx % window] = False
                idx += 1

                # Adapt sigma
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma *= 1.1
                    elif sr < 0.2:
                        sigma *= 0.9
                    else:
                        sigma *= 1.0
                    sigma = np.clip(sigma, 0.05, 5.0)

        # Phase 2: Rastrigin-aware refinement
        # Rastrigin minima are often near integer coordinates in the rotated space.
        # Since we don't know the rotation, we try to snap the best point found
        # to nearby integer-like structures and perform fine local search.
        
        if best_x is not None and calls < budget and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            
            # Step 2a: Snap to integer grid candidates
            # Generate candidates by rounding coordinates to nearest integer and +/- 1
            candidates = []
            for d in range(dim):
                base = x_local.copy()
                for offset in [-1, 0, 1]:
                    y = base.copy()
                    y[d] = round(y[d]) + offset
                    y = np.clip(y, -5, 5)
                    candidates.append(y)
            
            # Also add the fully rounded point
            candidates.append(np.clip(np.round(x_local), -5, 5))
            # And the original best point
            candidates.append(x_local)

            seen = set()
            for y in candidates:
                if calls >= budget or time.time() > deadline:
                    break
                key = tuple(np.round(y, 6))
                if key not in seen:
                    seen.add(key)
                    fy = eval_func(y)
                    if fy < f_local:
                        x_local, f_local = y, fy

            # Step 2b: Fine-grained coordinate descent
            # Use remaining budget for precise local search
            step = 0.2
            # We want to use about 100-150 calls here
            max_refine_calls = min(150, budget - calls)
            
            for _ in range(max_refine_calls):
                if calls >= budget or time.time() > deadline:
                    break
                
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        y = x_local.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = eval_func(y)
                        if fy < f_local:
                            x_local, f_local = y, fy
                            improved = True
                            # If we found an improvement, we might want to continue
                            # in this direction or restart the dimension loop.
                            # For efficiency, break out of sign loop and continue to next dim
                            break
                    if improved:
                        # If we improved in this dimension, keep step size or reduce slightly?
                        # For Rastrigin, reducing step helps converge to the peak of the cosine wave.
                        step *= 0.8
                if not improved:
                    step *= 0.5
                    if step < 0.005:
                        break

            # Update global best
            if f_local < best_f:
                best_f = f_local
                best_x = x_local

        return best_x, best_f
# EVOLVE-BLOCK-END
