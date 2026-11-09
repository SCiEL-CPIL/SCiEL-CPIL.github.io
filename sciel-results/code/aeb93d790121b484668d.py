"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Multi-Start Local Search with global archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 325.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([1705.0, 12.0, 0.3])

        rng = np.random.default_rng(0)

        # Global archive
        archive_X = []
        archive_F = []
        
        # Helper to check if a point is dominated by any in archive
        def is_dominated(f_new, F_archive):
            if len(F_archive) == 0:
                return False
            # Check if any point in archive dominates f_new
            # f_new is dominated if there exists f_a such that f_a <= f_new and f_a != f_new
            # For efficiency, we can check vectorized
            # But since archive grows, we do a simple loop for clarity and safety
            for f_a in F_archive:
                if np.all(f_a <= f_new) and np.any(f_a < f_new):
                    return True
            return False

        def update_archive(x_new, f_new):
            nonlocal archive_X, archive_F
            # Check if dominated
            if is_dominated(f_new, archive_F):
                return False
            
            # Remove points dominated by f_new
            new_archive_X = []
            new_archive_F = []
            for x_a, f_a in zip(archive_X, archive_F):
                # If f_new dominates f_a, discard f_a
                if np.all(f_new <= f_a) and np.any(f_new < f_a):
                    continue
                new_archive_X.append(x_a)
                new_archive_F.append(f_a)
            
            archive_X = new_archive_X
            archive_F = new_archive_F
            archive_X.append(x_new)
            archive_F.append(f_new)
            return True

        # Initial evaluations: diverse starting points
        # Use a mix of random and structured points (corners, midpoints)
        init_points = []
        # Random points
        n_random_init = 10
        init_points.extend(rng.random((n_random_init, dim)))
        
        # Corners
        for i in range(2 ** dim):
            point = np.zeros(dim)
            for j in range(dim):
                if (i >> j) & 1:
                    point[j] = 1.0
            init_points.append(point)
            
        # Midpoints
        mid = np.full(dim, 0.5)
        init_points.append(mid)

        init_points = np.array(init_points)
        # Scale to bounds
        init_points = lo + (hi - lo) * init_points
        
        # Ensure we don't exceed budget
        n_init = min(len(init_points), budget)
        init_points = init_points[:n_init]
        
        for x in init_points:
            if time.time() > deadline - 5:
                break
            f = problem(x)
            update_archive(x, f)
        
        used = len(init_points)
        
        # Multi-start local search
        # We have remaining budget. Let's divide it into runs.
        # Each run starts from a random point in the current archive (or random if empty)
        # and performs local search steps.
        
        # To ensure diversity, we can also start some runs from random points
        # but prioritize archive points for exploitation.
        
        remaining_budget = budget - used
        
        # Define local search step
        def local_search_step(x_current, f_current, rng):
            # Differential Evolution style mutation
            # Pick two random points from archive (if available) or random
            if len(archive_X) >= 3:
                # Pick 3 distinct random indices from archive
                idxs = rng.choice(len(archive_X), size=3, replace=False)
                x1 = np.array(archive_X[idxs[0]])
                x2 = np.array(archive_X[idxs[1]])
                x3 = np.array(archive_X[idxs[2]])
            else:
                # Fallback to random
                x1 = lo + (hi - lo) * rng.random(dim)
                x2 = lo + (hi - lo) * rng.random(dim)
                x3 = lo + (hi - lo) * rng.random(dim)
            
            # Mutation: x1 + 0.5 * (x2 - x3)
            F = 0.5
            mutant = x1 + F * (x2 - x3)
            
            # Crossover: Binomial crossover with probability 0.5
            cross_mask = rng.random(dim) < 0.5
            # Ensure at least one gene is from mutant
            if not np.any(cross_mask):
                j = rng.integers(dim)
                cross_mask[j] = True
            
            trial_x = np.where(cross_mask, mutant, x_current)
            trial_x = np.clip(trial_x, lo, hi)
            
            return trial_x

        # Run local searches
        # We will perform local search iterations until budget is exhausted or deadline reached.
        # Each "run" is a sequence of local search steps starting from a seed.
        # To balance exploration and exploitation, we alternate between:
        # 1. Starting from a random archive point (exploitation)
        # 2. Starting from a random point (exploration)
        
        # Let's just do a loop that picks a starting point and does a few steps
        # Or better, a continuous loop where we pick a random point from archive as seed
        # and perform one local search step. If it improves (non-dominated), it enters archive.
        
        # To avoid getting stuck, we can also inject random points periodically.
        
        while used < budget:
            if time.time() > deadline - 10:
                break
            
            # Pick a seed
            # 80% chance from archive, 20% chance random
            if len(archive_X) > 0 and rng.random() < 0.8:
                idx = rng.integers(len(archive_X))
                x_seed = np.array(archive_X[idx])
            else:
                x_seed = lo + (hi - lo) * rng.random(dim)
            
            # Perform a local search step
            x_trial = local_search_step(x_seed, None, rng)
            
            # Evaluate
            f_trial = problem(x_trial)
            used += 1
            
            # Update archive
            update_archive(x_trial, f_trial)
            
            # Optional: If the trial point is not dominated, we could continue local search from it
            # but for simplicity and to spread budget, we just restart from a new seed each time.
            # This is essentially a "hill-climbing" approach with random restarts.
            
            # To improve convergence, if the trial point is added to archive, 
            # we might want to do a few more steps from it.
            # Let's try a short chain: if added, do 2 more steps from it.
            
            if is_dominated(f_trial, archive_F[:-1]) == False and len(archive_F) > 1:
                # It was added (not dominated by others)
                # Do a couple more steps from this point
                for _ in range(2):
                    if used >= budget or time.time() > deadline - 5:
                        break
                    x_next = local_search_step(x_trial, f_trial, rng)
                    f_next = problem(x_next)
                    used += 1
                    # Check if it improves
                    if not is_dominated(f_next, archive_F):
                        update_archive(x_next, f_next)
                        x_trial = x_next
                        f_trial = f_next
                    else:
                        # If dominated, break the chain
                        break

        return None
# EVOLVE-BLOCK-END
