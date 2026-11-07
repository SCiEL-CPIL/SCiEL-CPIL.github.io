"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based (mu+lambda)-ES with adaptive step, multi-restart,
    and axis-wise Rastrigin-periodic refinement for f15 (rotated Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill

        # Global best tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            call_count += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Rastrigin-aware 1D refinement along one axis ---
        def refine_axis(x, axis):
            """Try to snap x[axis] to the nearest Rastrigin basin minimum.
            Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
            Minima at x_i = 0 mod 2*pi (i.e., x_i = 2*pi*k).
            But since the function is rotated, we work in the rotated frame.
            Here we just do a local 1D search around the current value."""
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return
            x0 = x[axis]
            # Try a few offsets around x0 to find better
            offsets = [0.0]
            # Try snapping to nearest 2*pi multiple
            nearest = 2.0 * np.pi * np.round(x0 / (2.0 * np.pi))
            offsets.append(nearest - x0)
            offsets.append(nearest - x0 + 2.0 * np.pi)
            offsets.append(nearest - x0 - 2.0 * np.pi)
            # Also try small perturbations
            for d in [0.1, -0.1, 0.3, -0.3]:
                offsets.append(d)

            x_f = float(func(x))
            call_count += 1

            for off in offsets:
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                y = x.copy()
                y[axis] = np.clip(x0 + off, -5, 5)
                fy = eval_func(y)
                if fy < x_f:
                    x[axis] = y[axis]
                    x_f = fy

        # --- Local refinement: coordinate descent with Rastrigin snap ---
        def local_refine(x, fx, max_iter=20):
            nonlocal call_count, best_x, best_f
            if call_count >= budget or time.time() > deadline:
                return x, fx
            for _ in range(max_iter):
                if call_count >= budget or time.time() > deadline:
                    break
                improved = False
                for axis in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    x0 = x[axis]
                    # Try snapping to nearest 2*pi multiple
                    nearest = 2.0 * np.pi * np.round(x0 / (2.0 * np.pi))
                    candidates = [x0, nearest, nearest + 2*np.pi, nearest - 2*np.pi]
                    # Add small perturbations
                    for d in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2]:
                        candidates.append(x0 + d)
                    best_cand = x0
                    best_cand_f = fx
                    for c in candidates:
                        if call_count >= budget:
                            break
                        c = float(np.clip(c, -5, 5))
                        if abs(c - x0) < 1e-10:
                            continue
                        y = x.copy()
                        y[axis] = c
                        fy = eval_func(y)
                        if fy < best_cand_f:
                            best_cand_f = fy
                            best_cand = c
                    if best_cand != x0:
                        x[axis] = best_cand
                        fx = best_cand_f
                        improved = True
                if not improved:
                    break
            return x, fx

        # --- Main search: (mu+lambda)-ES with periodic restarts ---
        # Phase 1: Broad exploration with population
        mu = 4
        lam = 12
        sigma = 2.5
        # Initial population
        pop = np.array([np.clip(rng.uniform(-5, 5, dim), -5, 5) for _ in range(mu)])
        pop_f = np.array([eval_func(p) for p in pop])
        # Sort
        idx = np.argsort(pop_f)
        pop = pop[idx]
        pop_f = pop_f[idx]

        generation = 0
        no_improve_count = 0
        best_f_prev = best_f

        while call_count < budget and time.time() < deadline:
            generation += 1

            # Adaptive sigma based on success rate (1/5 rule)
            # Count improvements from parent to offspring
            # For (mu+lambda), we'll just use a simple rule
            if generation % 10 == 0:
                # Check if we've been improving
                if best_f < best_f_prev - 1e-6:
                    sigma = min(sigma * 1.3, 8.0)
                    best_f_prev = best_f
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= 3:
                        sigma = max(sigma * 0.7, 0.1)
                        no_improve_count = 0

            # Generate offspring
            offspring = []
            offspring_f = []
            for i in range(lam):
                if call_count >= budget or time.time() > deadline:
                    break
                # Select parent (tournament or uniform from best mu)
                parent_idx = rng.integers(0, mu)
                parent = pop[parent_idx]
                child = np.clip(parent + sigma * rng.standard_normal(dim), -5, 5)
                child_f = eval_func(child)
                offspring.append(child)
                offspring_f.append(child_f)

            if not offspring:
                break

            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)

            # Combine parents and offspring, select best mu
            all_x = np.vstack([pop, offspring])
            all_f = np.concatenate([pop_f, offspring_f])
            idx = np.argsort(all_f)[:mu]
            pop = all_x[idx]
            pop_f = all_f[idx]

            # Periodic local refinement on the best point
            if generation % 15 == 0:
                x_ref, f_ref = local_refine(pop[0].copy(), pop_f[0], max_iter=10)
                if f_ref < pop_f[0]:
                    pop[0] = x_ref
                    pop_f[0] = f_ref

            # Restart if stuck: if sigma is very small and no improvement
            if sigma < 0.15 and generation % 30 == 0:
                # Restart with a few new random points
                for _ in range(2):
                    if call_count >= budget or time.time() > deadline:
                        break
                    new_x = np.clip(rng.uniform(-5, 5, dim), -5, 5)
                    new_f = eval_func(new_x)
                    # Replace worst if better
                    worst_idx = np.argmax(pop_f)
                    if new_f < pop_f[worst_idx]:
                        pop[worst_idx] = new_x
                        pop_f[worst_idx] = new_f
                sigma = 2.0
                no_improve_count = 0

        # Final phase: intensive local refinement on best point
        if best_x is not None and call_count < budget and time.time() < deadline:
            # Multiple rounds of coordinate descent
            x_final = best_x.copy()
            f_final = best_f
            for round in range(3):
                if call_count >= budget or time.time() > deadline:
                    break
                x_final, f_final = local_refine(x_final, f_final, max_iter=15)
                if f_final >= best_f:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
