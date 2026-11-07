"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Archive-based (1+1)-ES with restarts for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        max_time = 295.0

        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        # Archive of diverse good solutions
        archive = []  # list of (f, x)

        def add_to_archive(f, x):
            nonlocal archive
            # Keep at most 20 archive entries
            if len(archive) < 20:
                archive.append((f, x.copy()))
            else:
                # Only add if it's better than the worst in archive
                if f < archive[-1][0]:
                    archive[-1] = (f, x.copy())
            # Sort archive by fitness
            archive.sort(key=lambda t: t[0])
            # Keep only top 20
            archive = archive[:20]

        # Phase 1: Global search with large steps
        # Try several random starting points to find the main basin
        n_starts = 5
        for s in range(n_starts):
            if calls >= budget or time.time() - start_time > max_time:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            add_to_archive(fx, x)

        # Phase 2: Local refinement with adaptive (1+1)-ES
        # Use multiple restarts from archive points
        # Each restart runs a (1+1)-ES with sigma adaptation

        # Adaptive (1+1)-ES parameters
        sigma_init = 1.5
        sigma_min = 1e-8
        sigma_max = 5.0
        success_ratio = 0.2  # target success rate
        stagnation_limit = 30  # max iterations without improvement before restart

        def run_local_es(x0, f0, max_calls):
            """Run a (1+1)-ES starting from x0, f0."""
            nonlocal best_f, best_x, calls
            x = x0.copy()
            fx = f0
            sigma = sigma_init
            iters_no_improve = 0
            local_best_f = fx
            local_best_x = x.copy()

            for _ in range(max_calls):
                if calls >= budget or time.time() - start_time > max_time:
                    break

                # Generate candidate
                direction = rng.standard_normal(dim)
                candidate = x + sigma * direction
                candidate = np.clip(candidate, -5, 5)

                fc = func(candidate)
                calls += 1

                if fc < fx:
                    # Success
                    x = candidate
                    fx = fc
                    iters_no_improve = 0
                    # Increase sigma slightly on success
                    sigma = min(sigma * 1.05, sigma_max)

                    if fx < local_best_f:
                        local_best_f = fx
                        local_best_x = x.copy()

                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                    add_to_archive(fx, x)
                else:
                    # Failure
                    iters_no_improve += 1
                    # Decrease sigma on failure
                    sigma = max(sigma * 0.95, sigma_min)

                # Stagnation check
                if iters_no_improve >= stagnation_limit:
                    break

            return local_best_f, local_best_x

        # Run multiple local ES from different archive points
        # Sort archive by fitness and try the best few
        n_restarts = min(len(archive), 8)
        # Shuffle archive order for diversity
        archive_indices = list(range(n_restarts))
        rng.shuffle(archive_indices)

        for idx in archive_indices:
            if calls >= budget or time.time() - start_time > max_time:
                break

            f_start, x_start = archive[idx]
            # Allocate remaining budget proportionally
            remaining = budget - calls
            max_calls = max(10, remaining // (n_restarts - idx))

            local_f, local_x = run_local_es(x_start, f_start, max_calls)

            if local_f < best_f:
                best_f = local_f
                best_x = local_x.copy()
            add_to_archive(local_f, local_x)

        # Phase 3: Fine-tuning around the best solution
        if calls < budget and time.time() - start_time < max_time:
            sigma = 0.1
            x = best_x.copy()
            fx = best_f
            for _ in range(50):
                if calls >= budget or time.time() - start_time > max_time:
                    break
                direction = rng.standard_normal(dim)
                candidate = x + sigma * direction
                candidate = np.clip(candidate, -5, 5)
                fc = func(candidate)
                calls += 1
                if fc < fx:
                    x = candidate
                    fx = fc
                    sigma = min(sigma * 1.02, 0.5)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.98, 1e-6)

        return best_x, best_f
# EVOLVE-BLOCK-END
