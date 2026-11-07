"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with diverse exploration and aggressive polishing."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _lattice_walk(self, func, x0, max_calls, rng, deadline, d):
        """Lattice-based random walk exploiting Rastrigin's periodic structure."""
        x = x0.copy()
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx
        
        # Grid step size: Rastrigin has minima at integers, so step by 0.5 or 1.0
        # Use a mix of large jumps (to find new basins) and small steps (to refine)
        step_sizes = [1.0, 0.5, 0.25, 0.1]
        
        while calls < max_calls:
            if time.time() > deadline:
                break
            
            # Choose a step size: bias towards larger steps initially, smaller later
            progress = calls / max_calls
            if progress < 0.3:
                idx = rng.choice(4, p=[0.4, 0.3, 0.2, 0.1])
            elif progress < 0.7:
                idx = rng.choice(4, p=[0.2, 0.3, 0.3, 0.2])
            else:
                idx = rng.choice(4, p=[0.05, 0.15, 0.3, 0.5])
            
            step = step_sizes[idx]
            
            # Generate a lattice move: change 1-2 dimensions by +/- step
            num_dims = 1 if rng.random() < 0.7 else 2
            dims_to_change = rng.choice(d, size=num_dims, replace=False)
            
            y = x.copy()
            for dim_idx in dims_to_change:
                direction = 1 if rng.random() < 0.5 else -1
                y[dim_idx] += direction * step
            
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
            # No explicit rejection; just keep x if not improved (random walk)
            
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        # Strategy: Multiple diverse starts with ES, then polish
        # Phase 1: Multi-start exploration
        # Use ~60% of budget for exploration, 40% for polishing

        # Generate diverse starting points
        starts = []
        # Origin
        starts.append(np.zeros(d))
        # Random uniform points
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, d))
        # Points biased toward corners of the search space
        for _ in range(5):
            signs = rng.choice([-1, 1], d)
            starts.append(4.0 * signs * rng.uniform(0.5, 1.0, d))
        # Points with mixed magnitudes
        for _ in range(5):
            mags = rng.uniform(0.5, 4.5, d)
            signs = rng.choice([-1, 1], d)
            starts.append(mags * signs)

        # Shuffle to randomize order
        order = rng.permutation(len(starts))
        starts = [starts[i] for i in order]

        # Allocate budget: ~50 calls per start for 20 starts
        exploration_budget = int(budget * 0.6)
        calls_per_start = max(20, exploration_budget // len(starts))

        for idx, x0 in enumerate(starts):
            if check_deadline():
                break

            remaining_exploration = exploration_budget - total_calls
            if remaining_exploration < 10:
                break

            alloc = min(calls_per_start, remaining_exploration)
            bx, bf, calls_used = self._lattice_walk(
                func, x0, alloc, rng, deadline, d
            )
            total_calls += calls_used

            if bf < best_f:
                best_f, best_x = bf, bx.copy()

        # Phase 2: Polish around the best point found
        if best_x is not None and not check_deadline():
            polish_budget = budget - total_calls
            if polish_budget > 10:
                # Run a fine-grained lattice walk from the best point
                bx, bf, calls_used = self._lattice_walk(
                    func, best_x, polish_budget, rng, deadline, d
                )
                total_calls += calls_used
                if bf < best_f:
                    best_f, best_x = bf, bx.copy()

        # Phase 3: Final local search with very small steps
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            remaining = budget - total_calls
            
            # Use very fine lattice steps for final polish
            fine_steps = [0.1, 0.05, 0.02, 0.01]
            
            for _ in range(min(remaining, 50)):
                if check_deadline():
                    break
                step = fine_steps[rng.choice(4)]
                num_dims = 1
                dim_idx = rng.integers(0, d)
                direction = 1 if rng.random() < 0.5 else -1
                
                y = x.copy()
                y[dim_idx] += direction * step
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
