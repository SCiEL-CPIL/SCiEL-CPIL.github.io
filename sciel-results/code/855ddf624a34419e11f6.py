"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with aggressive sigma adaptation and lightweight restarts."""

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
            nonlocal best_f, best_x
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Discrete lattice-based random walk exploiting Rastrigin's periodicity
        # Rastrigin has local minima at integer coordinates (after affine transform)
        # Use discrete moves: step sizes that align with the function's periodicity

        # Phase 1: Broad exploration with discrete jumps
        # Start with larger discrete steps to find basin neighborhoods
        phase1_budget = int(budget * 0.5)
        phase2_budget = int(budget * 0.3)
        phase3_budget = budget - phase1_budget - phase2_budget

        # Initialize at origin
        x = np.zeros(d)
        fx = func(x)
        total_calls += 1
        update_best(x, fx)

        # Discrete step sizes for exploration (aligned with Rastrigin period ~2)
        # Try moves of size 1.0 and 2.0 in random directions
        no_improve = 0
        max_no_improve = 40

        for _ in range(phase1_budget):
            if check_deadline():
                break

            # Choose a random discrete move
            # 50% chance: unit step in one random coordinate
            # 30% chance: double step in one random coordinate  
            # 20% chance: unit step in two random coordinates
            r = rng.random()
            if r < 0.5:
                # Unit step in one coordinate
                idx = rng.integers(d)
                sign = 1.0 if rng.random() < 0.5 else -1.0
                y = x.copy()
                y[idx] += sign * 1.0
            elif r < 0.8:
                # Double step in one coordinate
                idx = rng.integers(d)
                sign = 1.0 if rng.random() < 0.5 else -1.0
                y = x.copy()
                y[idx] += sign * 2.0
            else:
                # Unit step in two coordinates
                idxs = rng.choice(d, size=2, replace=False)
                signs = rng.choice([-1.0, 1.0], size=2)
                y = x.copy()
                y[idxs[0]] += signs[0]
                y[idxs[1]] += signs[1]

            y = np.clip(y, -5, 5)
            fy = func(y)
            total_calls += 1

            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                update_best(x, fx)
            else:
                no_improve += 1

            # Restart if stalled
            if no_improve >= max_no_improve:
                x = rng.uniform(-4.0, 4.0, d)
                fx = func(x)
                total_calls += 1
                update_best(x, fx)
                no_improve = 0
                if check_deadline():
                    break

        # Phase 2: Medium-resolution refinement around best point
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            no_improve = 0
            max_no_improve = 30

            for _ in range(phase2_budget):
                if check_deadline():
                    break

                # Smaller discrete steps: 0.5 and 0.25
                r = rng.random()
                if r < 0.5:
                    idx = rng.integers(d)
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    y = x.copy()
                    y[idx] += sign * 0.5
                elif r < 0.8:
                    idx = rng.integers(d)
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    y = x.copy()
                    y[idx] += sign * 0.25
                else:
                    idxs = rng.choice(d, size=2, replace=False)
                    signs = rng.choice([-1.0, 1.0], size=2)
                    y = x.copy()
                    y[idxs[0]] += signs[0] * 0.25
                    y[idxs[1]] += signs[1] * 0.25

                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    update_best(x, fx)
                else:
                    no_improve += 1

                if no_improve >= max_no_improve:
                    # Small perturbation around current best
                    x = best_x + rng.standard_normal(d) * 0.3
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    total_calls += 1
                    update_best(x, fx)
                    no_improve = 0
                    if check_deadline():
                        break

        # Phase 3: Fine polish with very small steps
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            no_improve = 0
            max_no_improve = 20

            for _ in range(phase3_budget):
                if check_deadline():
                    break

                # Very fine steps: 0.1 and 0.05
                r = rng.random()
                if r < 0.5:
                    idx = rng.integers(d)
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    y = x.copy()
                    y[idx] += sign * 0.1
                else:
                    idx = rng.integers(d)
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    y = x.copy()
                    y[idx] += sign * 0.05

                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    update_best(x, fx)
                else:
                    no_improve += 1

                if no_improve >= max_no_improve:
                    x = best_x + rng.standard_normal(d) * 0.1
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    total_calls += 1
                    update_best(x, fx)
                    no_improve = 0
                    if check_deadline():
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
