"""Multi-fidelity population-based ES with diverse archive and coarse-to-fine adaptive sigma."""
import numpy as np


class Optimizer:
    """
    Population-based evolutionary strategy with:
    - Diverse archive to prevent premature convergence
    - Coarse-to-fine sigma schedule
    - Sigma dynamically scaled by improvement rate
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Population and archive sizes
        pop_size = 15
        archive_size = 10
        total_pool = pop_size + archive_size

        # Initialize diverse population
        pop = np.array([rng.uniform(-5, 5, dim) for _ in range(pop_size)])
        pop_f = np.array([func(x) for x in pop])

        # Initialize archive with diverse points (spread across the box)
        archive = np.array([rng.uniform(-5, 5, dim) for _ in range(archive_size)])
        archive_f = np.array([func(x) for x in archive])

        calls_made = pop_size + archive_size
        remaining = budget - calls_made

        # Global best tracking
        all_x = np.vstack([pop, archive])
        all_f = np.concatenate([pop_f, archive_f])
        best_idx = np.argmin(all_f)
        best_x = all_x[best_idx].copy()
        best_f = all_f[best_idx]

        # Adaptive sigma parameters
        sigma = 4.0  # Start large for global exploration
        sigma_min = 1e-6
        sigma_max = 8.0
        sigma_history = []  # Track recent sigmas for adaptation

        # Improvement tracking
        best_f_history = [best_f]
        improvement_momentum = 1.0  # 1.0 = actively improving, 0.0 = stuck

        # Coarse-to-fine schedule: sigma decays over budget
        # Phase boundaries
        phase1_end = 0.35  # 35% budget: global exploration
        phase2_end = 0.70  # 70% budget: transition
        # After 70%: fine exploitation

        while remaining > 0:
            # Compute progress (0 to 1)
            progress = 1.0 - remaining / max(budget, 1)

            # Determine target sigma from coarse-to-fine schedule
            if progress < phase1_end:
                # Phase 1: Large steps, explore globally
                target_sigma = 4.0
            elif progress < phase2_end:
                # Phase 2: Medium steps
                target_sigma = 1.5
            else:
                # Phase 3: Fine exploitation
                target_sigma = 0.3 * max(1.0, 1.0 - (progress - phase2_end) / (1.0 - phase2_end))

            # Adapt sigma based on improvement momentum
            # If improving well, maintain or slightly increase sigma
            # If stuck, reduce sigma to exploit locally
            if improvement_momentum > 0.3:
                # Actively improving: move toward target, allow some increase
                sigma = sigma * 0.7 + target_sigma * 0.3
                # Allow slight increase if momentum is strong
                if improvement_momentum > 0.7:
                    sigma = min(sigma * 1.1, sigma_max)
            elif improvement_momentum > 0.1:
                # Slow improvement: move toward target
                sigma = sigma * 0.8 + target_sigma * 0.2
            else:
                # Stuck: reduce sigma to exploit around best
                sigma = max(sigma * 0.6, target_sigma * 0.5, sigma_min)

            sigma = np.clip(sigma, sigma_min, sigma_max)
            sigma_history.append(sigma)

            # Generate offspring
            n_offspring = min(pop_size, remaining)
            if n_offspring <= 0:
                break
            remaining -= n_offspring

            # Parent selection: mix of best population members and archive diversity
            # Sort population
            pop_sorted_idx = np.argsort(pop_f)
            # Top half of population + a few random archive members for diversity
            elite_count = max(1, pop_size // 3)
            parent_pool = np.concatenate([
                pop[pop_sorted_idx[:elite_count]],
                archive[rng.choice(archive_size, size=max(1, archive_size // 3), replace=False)]
            ])

            offspring = np.zeros((n_offspring, dim))
            offspring_f = np.zeros(n_offspring)

            for i in range(n_offspring):
                # Pick parent from pool
                pi = rng.integers(len(parent_pool))
                parent = parent_pool[pi]

                # Multi-scale perturbation: mix of large and small steps
                # 70% chance: standard Gaussian with current sigma
                # 30% chance: smaller step for local refinement
                if rng.random() < 0.7:
                    step = sigma * rng.standard_normal(dim)
                else:
                    step = sigma * 0.2 * rng.standard_normal(dim)

                mutant = parent + step
                mutant = np.clip(mutant, -5, 5)
                offspring[i] = mutant
                offspring_f[i] = func(mutant)

            # Update population: keep best pop_size from combined pool
            # Combine population and offspring
            combined_x = np.vstack([pop, offspring])
            combined_f = np.concatenate([pop_f, offspring_f])

            keep_idx = np.argsort(combined_f)[:pop_size]
            pop = combined_x[keep_idx].copy()
            pop_f = combined_f[keep_idx].copy()

            # Update archive: maintain diversity
            # Add best new points to archive, but keep diversity
            # Sort offspring by fitness
            off_sorted = np.argsort(offspring_f)
            # Take top few offspring that are sufficiently different from archive
            for oi in off_sorted[:3]:
                new_point = offspring[oi]
                # Check distance to existing archive
                dists = np.min(np.linalg.norm(archive - new_point, axis=1))
                if dists > 0.5:  # Sufficiently diverse
                    # Replace worst archive member
                    worst_archive_idx = np.argmax(archive_f)
                    archive[worst_archive_idx] = new_point
                    archive_f[worst_archive_idx] = offspring_f[oi]

            # Update global best
            all_x = np.vstack([pop, archive])
            all_f = np.concatenate([pop_f, archive_f])
            new_best_idx = np.argmin(all_f)
            new_best_f = all_f[new_best_idx]

            if new_best_f < best_f:
                improvement = best_f - new_best_f
                # Update improvement momentum
                if abs(best_f) > 1e-6:
                    rel_improvement = improvement / abs(best_f)
                else:
                    rel_improvement = 1.0 if improvement > 1e-8 else 0.0
                improvement_momentum = min(1.0, 0.5 * improvement_momentum + 0.5 * min(1.0, rel_improvement * 10))
                best_f = new_best_f
                best_x = all_x[new_best_idx].copy()
            else:
                # No improvement: decay momentum
                improvement_momentum *= 0.85

            best_f_history.append(best_f)

            # In late exploitation phase, add extra local refinement around best
            if progress > phase2_end and remaining > 0:
                local_sigma = max(sigma * 0.1, 0.05)
                n_local = min(3, remaining)
                for _ in range(n_local):
                    local_mutant = best_x + local_sigma * rng.standard_normal(dim)
                    local_mutant = np.clip(local_mutant, -5, 5)
                    local_f = func(local_mutant)
                    remaining -= 1
                    if local_f < best_f:
                        best_f = local_f
                        best_x = local_mutant.copy()
                        improvement_momentum = 0.5

        return best_x, best_f
