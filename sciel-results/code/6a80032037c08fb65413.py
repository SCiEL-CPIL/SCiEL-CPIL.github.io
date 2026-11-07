# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Anisotropic coordinate descent with per-dimension adaptive steps,
    exploiting the Bent Cigar function's structure (one easy dim, many hard)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # --- Phase 1: coarse random restarts to get near the basin ---
        rng = np.random.default_rng(0)
        x = np.zeros(dim)
        fx = func(x)
        calls += 1
        best_x = x.copy()
        best_f = fx

        # Random restarts: evaluate a few random points, keep the best
        for _ in range(8):
            if calls >= budget or time.time() > deadline:
                break
            cand = rng.uniform(-5.0, 5.0, dim)
            fcand = func(cand)
            calls += 1
            if fcand < best_f:
                best_f = fcand
                best_x = cand.copy()
        x = best_x.copy()
        fx = best_f

        # --- Phase 2: coordinate descent with per-dim adaptive steps ---
        # Bent Cigar: f = (x1-1)^2 + sum_{i=2..n} 100*(x_i-1)^2
        # Dimension 0 is easy (width ~1), others are hard (width ~0.1)
        # Start with small steps for hard dims, larger for easy dim
        steps = np.ones(dim)
        steps[0] = 1.0
        for i in range(1, dim):
            steps[i] = 0.3

        # Per-dimension step sizes, adapt based on success
        # Use a local search around current point
        dim_order = list(range(dim))
        # Try to learn which dimension is the "easy" one by probing

        # Probe each dimension to estimate local curvature / step scale
        # For each dim, try a few step sizes and see which gives improvement
        for probe_iter in range(3):
            if calls >= budget or time.time() > deadline:
                break
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                # Try a forward step
                trial = x.copy()
                trial[i] += steps[i]
                trial = clip(trial)
                ftrial = func(trial)
                calls += 1
                if ftrial < fx:
                    x = trial
                    fx = ftrial
                    steps[i] *= 1.5
                else:
                    # Try backward
                    trial = x.copy()
                    trial[i] -= steps[i]
                    trial = clip(trial)
                    ftrial = func(trial)
                    calls += 1
                    if ftrial < fx:
                        x = trial
                        fx = ftrial
                        steps[i] *= 1.5
                    else:
                        steps[i] *= 0.5

        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        # --- Phase 3: fine coordinate descent with shrinking steps ---
        # Alternate between forward/backward steps, adapt step size
        # For Bent Cigar, the hard dimensions need very small steps
        for i in range(dim):
            steps[i] = max(steps[i], 0.05)

        # Main loop: iterate over dimensions, try to improve
        # Use a "walk" strategy: for each dim, do a local 1D search
        # Golden-section-like refinement for each dimension

        # Re-learn step scales: for each dim, find a step that gives ~improvement
        # by trying a range of step sizes
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            # Binary-search-like: find step size where function decreases
            # Try decreasing step sizes until we find improvement
            s = steps[i]
            improved = False
            for _ in range(5):
                if calls >= budget or time.time() > deadline:
                    break
                # Try forward
                trial = x.copy()
                trial[i] += s
                trial = clip(trial)
                ftrial = func(trial)
                calls += 1
                if ftrial < fx:
                    x = trial
                    fx = ftrial
                    steps[i] = s
                    improved = True
                    break
                # Try backward
                trial = x.copy()
                trial[i] -= s
                trial = clip(trial)
                ftrial = func(trial)
                calls += 1
                if ftrial < fx:
                    x = trial
                    fx = ftrial
                    steps[i] = s
                    improved = True
                    break
                s *= 0.5
            if not improved:
                steps[i] = max(s * 2, 1e-6)

        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        # --- Phase 4: main optimization loop ---
        # Coordinate descent with adaptive steps, cycling through dimensions
        # Occasionally do a random perturbation to escape local minima
        stagnation = 0
        best_f_local = fx
        last_improve_iter = 0
        total_iters = 0

        while calls < budget and time.time() < deadline:
            # Cycle through dimensions in a fixed order
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                # Try to improve along dimension i
                # Use a small local search: try +/- step, and if improved, continue
                s = steps[i]
                # Forward
                trial = x.copy()
                trial[i] += s
                trial = clip(trial)
                ftrial = func(trial)
                calls += 1
                if ftrial < fx:
                    x = trial
                    fx = ftrial
                    steps[i] = min(s * 1.3, 5.0)
                    if fx < best_f_local:
                        best_f_local = fx
                        last_improve_iter = total_iters
                        stagnation = 0
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                else:
                    # Backward
                    trial = x.copy()
                    trial[i] -= s
                    trial = clip(trial)
                    ftrial = func(trial)
                    calls += 1
                    if ftrial < fx:
                        x = trial
                        fx = ftrial
                        steps[i] = min(s * 1.3, 5.0)
                        if fx < best_f_local:
                            best_f_local = fx
                            last_improve_iter = total_iters
                            stagnation = 0
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                    else:
                        # No improvement; shrink step
                        steps[i] *= 0.7
                        stagnation += 1

            total_iters += 1

            # Periodic: if stagnating, try a random perturbation
            if stagnation > 20 and calls + dim < budget:
                # Try a small random kick
                kick = rng.standard_normal(dim) * 0.1
                trial = clip(x + kick)
                ftrial = func(trial)
                calls += 1
                if ftrial < fx:
                    x = trial
                    fx = ftrial
                    stagnation = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Revert and try a larger step reset for hard dims
                    for i in range(dim):
                        if steps[i] < 0.01:
                            steps[i] = 0.05

        # Final: ensure best is recorded
        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
