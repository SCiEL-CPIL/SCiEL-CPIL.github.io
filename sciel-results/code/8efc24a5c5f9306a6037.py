"""Multi-start (1+1)-ES with 1/5-rule step-size adaptation and island-based parallel search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-start (1+1)-ES with:
    - 1/5 rule for step-size adaptation
    - Island-based parallel search (multiple independent trails)
    - Occasional migration between islands
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s limit

        n = self.dim
        call_count = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def make_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            fx = func(x)
            call_count += 1
            return fx

        def update_best(x, f):
            nonlocal best_x, best_f
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()

        def es_11_search(x0, sigma0, max_calls, deadline):
            """
            Run (1+1)-ES from x0 with initial step size sigma0.
            Uses 1/5 rule for step-size adaptation.
            Returns (best_x_local, best_f_local, calls_made)
            """
            nonlocal best_x, best_f, call_count

            x = x0.copy()
            sigma = sigma0
            local_best_x = x.copy()
            local_best_f = np.inf
            calls_in_run = 0
            success_count = 0
            total_trials = 0
            stagnation = 0

            # Initial evaluation
            fx = make_call(x)
            if fx is None:
                return local_best_x, local_best_f, 1
            calls_in_run += 1
            if np.isfinite(fx):
                local_best_f = fx
                local_best_x = x.copy()
                update_best(x, fx)

            while calls_in_run < max_calls:
                if call_count >= self.budget:
                    break
                if time.time() > deadline - 3.0:
                    break

                # Generate trial point
                direction = rng.standard_normal(n)
                direction /= (np.linalg.norm(direction) + 1e-12)
                x_trial = x + sigma * direction
                x_trial = np.clip(x_trial, -5, 5)

                f_trial = make_call(x_trial)
                if f_trial is None:
                    break
                calls_in_run += 1

                if not np.isfinite(f_trial):
                    # Bad point, shrink step size
                    sigma *= 0.5
                    continue

                # Accept if better
                if f_trial < local_best_f:
                    x = x_trial
                    local_best_f = f_trial
                    local_best_x = x.copy()
                    update_best(x, f_trial)
                    success_count += 1
                    stagnation = 0
                else:
                    stagnation += 1

                total_trials += 1

                # 1/5 rule: adapt step size every 1/n trials
                if total_trials >= n:
                    success_rate = success_count / total_trials
                    if success_rate > 0.2:
                        sigma *= 1.05
                    elif success_rate < 0.2:
                        sigma *= 0.95
                    success_count = 0
                    total_trials = 0

                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 10.0)

                # If stagnating for too long, break
                if stagnation > 50:
                    break

            return local_best_x, local_best_f, calls_in_run

        # Strategy:
        # 1. Initialize multiple islands with diverse starting points
        # 2. Run (1+1)-ES on each island in parallel (round-robin)
        # 3. Occasionally migrate the best solution between islands
        # 4. Track global best

        num_islands = 4
        islands = []
        for i in range(num_islands):
            if i == 0:
                x_start = np.zeros(n)
            else:
                x_start = rng.uniform(-3, 3, n)
            sigma_start = 1.5 if i == 0 else 1.0
            islands.append({
                'x': x_start,
                'sigma': sigma_start,
                'best_x': x_start.copy(),
                'best_f': np.inf,
                'active': True
            })

        # Initial evaluations for all islands
        for island in islands:
            if call_count >= self.budget:
                break
            fx = make_call(island['x'])
            if fx is not None and np.isfinite(fx):
                island['best_f'] = fx
                island['best_x'] = island['x'].copy()
                update_best(island['x'], fx)

        # Main loop: round-robin across islands
        calls_per_island_budget = 100
        total_calls_made = 0
        migration_interval = 30
        migration_counter = 0

        while call_count < self.budget - 10:
            if time.time() > deadline - 5.0:
                break

            # Find active islands
            active_islands = [isl for isl in islands if isl['active']]
            if not active_islands:
                break

            # Pick an island to work on (round-robin)
            island_idx = total_calls_made % len(active_islands)
            island = active_islands[island_idx]

            # Run a short burst of (1+1)-ES on this island
            burst_calls = min(calls_per_island_budget, self.budget - call_count - 10)
            if burst_calls <= 0:
                break

            x_new, f_new, calls_made = es_11_search(
                island['x'], island['sigma'], burst_calls, deadline
            )

            # Update island state
            island['x'] = x_new
            island['best_x'] = f_new if f_new < island['best_f'] else island['best_x']
            if f_new < island['best_f']:
                island['best_f'] = f_new
                island['best_x'] = x_new.copy()

            total_calls_made += calls_made
            migration_counter += calls_made

            # Migration: occasionally copy the best solution to a worse island
            if migration_counter >= migration_interval:
                migration_counter = 0
                # Find the best and worst island
                best_island = min(islands, key=lambda isl: isl['best_f'])
                worst_island = max(islands, key=lambda isl: isl['best_f'])

                # Migrate if there's a significant gap
                if best_island['best_f'] < worst_island['best_f'] - 0.1:
                    # Perturb the best solution and assign to worst island
                    perturbation = rng.standard_normal(n) * 0.5
                    worst_island['x'] = best_island['best_x'] + perturbation
                    worst_island['x'] = np.clip(worst_island['x'], -5, 5)
                    worst_island['sigma'] = max(worst_island['sigma'] * 0.5, 0.1)

            # Deactivate islands that have stagnated badly
            for isl in islands:
                if not np.isfinite(isl['best_f']) or isl['best_f'] > best_f + 50:
                    if call_count > self.budget * 0.5:
                        isl['active'] = False

        # Final fallback
        if best_x is None:
            x = np.zeros(n)
            fx = make_call(x)
            if fx is not None and np.isfinite(fx):
                best_x = x.copy()
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
