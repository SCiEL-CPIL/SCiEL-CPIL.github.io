"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with large-jump restarts for Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _es_with_jumps(self, func, x0, sigma0, max_calls, rng, deadline, d):
        """
        Run a (1+1)-ES that performs large random jumps when stuck.
        This helps escape local minima in multimodal landscapes like Rastrigin.
        """
        x = x0.copy()
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx
        sigma = sigma0
        no_improve = 0
        max_no_improve = 5  # Jump if stuck for 5 steps

        while calls < max_calls:
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(d)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Accept
                x, fx = y, fy
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                no_improve = 0
                sigma *= 1.1  # Slight increase to explore around new best
            else:
                # Reject
                no_improve += 1
                sigma *= 0.9  # Decrease step size to refine

            # Escape local minimum if stuck
            if no_improve >= max_no_improve:
                # Perform a large random jump to a new region
                # Step size for jump is significantly larger than current sigma
                jump_sigma = 2.0
                x = x + jump_sigma * rng.standard_normal(d)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                
                # Reset step size for the new region
                sigma = 1.0
                no_improve = 0

            # Ensure sigma doesn't get too small or too large
            sigma = np.clip(sigma, 0.01, 5.0)

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        # Phase 1: Exploration with diverse starts and jump-enabled ES
        # We allocate 60% of the budget to exploration
        exploration_budget = int(budget * 0.6)
        
        # Generate diverse starting points
        # 1. Origin
        starts = [np.zeros(d)]
        # 2. Random uniform
        for _ in range(15):
            starts.append(rng.uniform(-4.5, 4.5, d))
        # 3. Lattice-like points (integers) - Rastrigin minima are at integers in unrotated space
        # Even with rotation, sampling near integers might be beneficial due to periodicity
        for _ in range(10):
            # Random integers in [-4, 4]
            pts = rng.integers(-4, 5, d).astype(float)
            # Add small noise to avoid exact singularities if any
            pts += rng.uniform(-0.1, 0.1, d)
            starts.append(pts)

        # Shuffle starts to randomize exploration order
        order = rng.permutation(len(starts))
        starts = [starts[i] for i in order]

        # Allocate budget per start
        # We have ~26 starts. 300 calls / 26 ~ 11 calls per start.
        # Let's use 15 calls per start for the first few, then less.
        calls_per_start = 15
        
        for idx, x0 in enumerate(starts):
            if check_deadline():
                break
            
            remaining_exploration = exploration_budget - total_calls
            if remaining_exploration < 10:
                break

            alloc = min(calls_per_start, remaining_exploration)
            
            bx, bf, calls_used = self._es_with_jumps(
                func, x0, 1.5, alloc, rng, deadline, d
            )
            total_calls += calls_used

            if bf < best_f:
                best_f, best_x = bf, bx.copy()

        # Phase 2: Polishing the best point found
        if best_x is not None and not check_deadline():
            polish_budget = budget - total_calls
            if polish_budget > 10:
                # Use a smaller initial step size for polishing
                # We assume we are in a good basin now
                bx, bf, calls_used = self._es_with_jumps(
                    func, best_x, 0.5, polish_budget, rng, deadline, d
                )
                total_calls += calls_used
                if bf < best_f:
                    best_f, best_x = bf, bx.copy()

        # Phase 3: Final ultra-fine local search
        if best_x is not None and not check_deadline():
            # Use a very small step size to converge to the minimum
            # We can reuse the ES function but with a very small sigma and no jumps
            # Or just do a simple Nelder-Mead if available, but let's stick to ES
            # Let's do a final (1+1)-ES with sigma=0.05 and no jumps (max_no_improve=100)
            
            # We need a version without jumps for the final polish
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            remaining = budget - total_calls
            
            for _ in range(remaining):
                if check_deadline():
                    break
                y = x + sigma * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    
                sigma = np.clip(sigma, 1e-6, 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
