import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        limit_time = 325  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Archive for non-dominated solutions
        archive_X = []
        archive_F = []
        
        # Helper to add to archive and return if dominated
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if x is dominated by any existing solution
            if len(archive_F) > 0:
                existing_F = np.array(archive_F)
                # f is dominated by existing if existing <= f and existing < f
                dominated_mask = np.all(existing_F <= f, axis=1) & np.any(existing_F < f, axis=1)
                if np.any(dominated_mask):
                    return False
            
            # Remove existing solutions dominated by new solution
            if len(archive_F) > 0:
                existing_F = np.array(archive_F)
                # existing is dominated by f if f <= existing and f < existing
                dominated_by_new_mask = np.all(f <= existing_F, axis=1) & np.any(f < existing_F, axis=1)
                if np.any(dominated_by_new_mask):
                    indices_to_remove = np.where(dominated_by_new_mask)[0]
                    # Remove from both lists
                    # Since lists are small, filtering is fine
                    new_archive_X = [archive_X[i] for i in range(len(archive_X)) if i not in indices_to_remove]
                    new_archive_F = [archive_F[i] for i in range(len(archive_F)) if i not in indices_to_remove]
                    archive_X = new_archive_X
                    archive_F = new_archive_F
            
            archive_X.append(x)
            archive_F.append(f)
            return True

        # Helper to evaluate and add
        def eval_and_add(x):
            x = np.clip(x, lo, hi)
            f = problem(x)
            return add_to_archive(x, f), x, f

        # Phase 1: Global Exploration with Stratified Sampling
        # We want to cover the space well. Let's use 50 initial points.
        N_init = 50
        # Latin Hypercube Sampling approximation
        # For each dimension, split range into N_init intervals, pick one random point in each, then shuffle.
        # This is hard to do perfectly in parallel for all dims simultaneously without a library, 
        # so we do a simpler stratified random: divide each dim into bins, pick one per bin? 
        # Actually, simple random is often fine if N is large enough. 
        # But let's try a structured approach: 
        # Generate N_init random points.
        
        X_init = lo + np.random.rand(N_init, self.dim) * span
        for x in X_init:
            eval_and_add(x)
        
        # Phase 2: Evolutionary / Local Search Hybrid
        # We have budget - N_init left.
        # We will maintain a population of size N_pop.
        N_pop = 30
        # Initialize population from archive if possible, otherwise random
        if len(archive_X) >= N_pop:
            # Sample N_pop from archive
            indices = np.random.choice(len(archive_X), N_pop, replace=False)
            X_pop = np.array([archive_X[i] for i in indices])
        else:
            # Fill with random
            extra = N_pop - len(archive_X)
            X_extra = lo + np.random.rand(extra, self.dim) * span
            X_pop = np.vstack([np.array(archive_X), X_extra]) if archive_X else X_extra
            for x in X_pop:
                eval_and_add(x) # Re-evaluate to ensure they are in archive if not dominated? 
                # Actually, we should just keep X_pop as is and evaluate offspring.
        
        used = N_init + len(X_pop) # Roughly, some might have been added in init loop
        # Let's track used more carefully. 
        # We added N_init in loop. 
        # Then we created X_pop. If we sampled from archive, they were already evaluated. 
        # If we created extra, we need to evaluate them.
        
        # Let's reset used count to be precise.
        # The evaluator counts calls to problem(x).
        # We called problem N_init times.
        # In the block above:
        # If len(archive_X) >= N_pop: we did NOT call problem again for those.
        # Else: we called problem for `extra` points.
        
        if len(archive_X) >= N_pop:
            used = N_init
        else:
            used = N_init + (N_pop - len(archive_X))
            
        # Now enter the loop
        while used < self.budget:
            if time.time() > limit_time:
                break
            
            # Generate offspring
            # Strategy: For each individual in population, generate a mutant.
            # If it is non-dominated, it goes to archive.
            # Then we select the next population.
            
            offspring_X = []
            offspring_F = []
            
            for i in range(N_pop):
                if used >= self.budget:
                    break
                
                parent = X_pop[i]
                
                # Random mutation: 
                # Mix of global jump (small probability) and local refinement (high probability)
                if np.random.rand() < 0.1:
                    # Global jump
                    child = lo + np.random.rand(self.dim) * span
                else:
                    # Local refinement
                    # Gaussian mutation scaled by distance to bounds? 
                    # Simple: add noise
                    noise = np.random.randn(self.dim) * 0.1
                    # Scale noise by span to be relative
                    # Better: use a fixed small step in normalized space
                    child = parent + noise * span * 0.05
                    # Or, more standard: SBX-like or Polynomial mutation.
                    # Let's use a simple bounded mutation:
                    # Pick a subset of dimensions to mutate
                    n_mutate = max(1, int(self.dim * 0.1))
                    dims_to_mutate = np.random.choice(self.dim, n_mutate, replace=False)
                    for d in dims_to_mutate:
                        # Uniform mutation in a window around parent
                        window = 0.2 * span[d]
                        child[d] = parent[d] + np.random.uniform(-window, window)
                
                child = np.clip(child, lo, hi)
                f = problem(child)
                used += 1
                
                # Check if non-dominated
                is_new = add_to_archive(child, f)
                if is_new:
                    offspring_X.append(child)
                    offspring_F.append(f)
            
            if used >= self.budget:
                break
            
            # Update Population
            # We want the next population to be a diverse subset of the archive.
            # To ensure diversity, we can use a simple greedy selection:
            # Pick the most spread out points.
            # Or, simpler: Keep the best N_pop points from the archive based on "crowding" or just random sampling to allow exploration.
            # Random sampling from archive is often surprisingly good for small budgets as it prevents premature convergence.
            # However, we should bias towards "better" points (lower objectives).
            
            if len(archive_F) > 0:
                archive_F_arr = np.array(archive_F)
                # Normalize objectives for distance calculation? 
                # Let's just pick N_pop random points from the archive.
                # This is an "Island" style approach where the population is always a random sample of the global best.
                if len(archive_F) > N_pop:
                    indices = np.random.choice(len(archive_F), N_pop, replace=False)
                else:
                    indices = range(len(archive_F))
                    # If archive is smaller, we need to fill pop with new randoms to keep N_pop constant?
                    # Or just reduce N_pop? Reducing N_pop slows down convergence but we have budget.
                    # Let's keep N_pop constant by adding randoms if archive is small.
                
                X_new = np.array([archive_X[i] for i in indices])
                
                # If we have fewer than N_pop, add randoms
                if len(X_new) < N_pop:
                    extra_needed = N_pop - len(X_new)
                    X_extra = lo + np.random.rand(extra_needed, self.dim) * span
                    X_new = np.vstack([X_new, X_extra])
                    # We need to evaluate these extra randoms?
                    # No, we just add them to population. They will be evaluated in the next iteration's offspring generation?
                    # Wait, the population members themselves are not "used" budget if they are from archive.
                    # But if we add *new* randoms to the population, do we count them?
                    # The problem says "More than 200 evaluations".
                    # Creating a random vector and adding it to X_pop does NOT call problem().
                    # So it is free.
                    # However, in the next loop, we will mutate them.
                    # So this is a valid strategy.
                
                X_pop = X_new
            
            # Small chance to inject a completely new random individual into population
            if np.random.rand() < 0.05:
                x_new = lo + np.random.rand(self.dim) * span
                # Replace worst individual? Or just replace random one?
                idx = np.random.randint(0, len(X_pop))
                X_pop[idx] = x_new

        return archive_X
