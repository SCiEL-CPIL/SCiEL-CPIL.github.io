"""Parallel Multi-Scale Differential Evolution Optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Parallel Multi-Scale Differential Evolution (PMDE) Optimizer.
    
    Maintains three populations of different sizes simultaneously:
    1. Large population (Global): Explores the full space, low crossover rate.
    2. Medium population (Basin): Focuses on promising regions, medium crossover.
    3. Small population (Fine): Exploits the current best neighborhood, high crossover.
    
    Uses a shared archive of the best points found so far to guide mutation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295  # Safety margin for 300s limit

        x_best = np.zeros(dim)
        f_best = float('inf')
        calls = 0

        # Archive of top K points to guide mutation
        ARCHIVE_SIZE = 10
        archive = []  # List of (f_val, x_val)

        def safe_call(x):
            nonlocal calls, f_best, x_best
            if time.time() > deadline or calls >= budget:
                return None
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls += 1
            
            if val < f_best:
                f_best = val
                x_best = x_clipped.copy()
                
            # Update archive
            archive.append((val, x_clipped.copy()))
            archive.sort(key=lambda item: item[0])
            if len(archive) > ARCHIVE_SIZE:
                archive.pop()
            
            return val

        def update_archive_from_population(pop, evals):
            """Add evaluated points to the global archive."""
            for x, f_val in zip(pop, evals):
                if f_val is not None:
                    archive.append((f_val, x.copy()))
            archive.sort(key=lambda item: item[0])
            if len(archive) > ARCHIVE_SIZE:
                archive = archive[:ARCHIVE_SIZE]

        # Define population sizes
        # Total budget is 300.
        # We will run these populations in a loop, allocating a chunk of budget to each "round".
        # Let's define the populations:
        n_large = 20
        n_med = 10
        n_small = 5
        total_pop = n_large + n_med + n_small

        # Initialize populations
        # Large: Random uniform
        # Med: Random around best few points from previous round (or center initially)
        # Small: Random around current best

        def init_populations():
            pop_large = np.random.uniform(-5, 5, (n_large, dim))
            # For med, start with some random, some around 0
            pop_med = np.random.uniform(-2, 2, (n_med, dim))
            # For small, start around 0
            pop_small = np.random.uniform(-1, 1, (n_small, dim))
            return pop_large, pop_med, pop_small

        pop_large, pop_med, pop_small = init_populations()
        
        # We need to evaluate the initial populations.
        # To save calls, we evaluate all at once? No, we must track calls carefully.
        # Let's evaluate the initial sets.
        
        # Helper to evaluate a population
        def evaluate_population(pop):
            evals = []
            for x in pop:
                if time.time() > deadline or calls >= budget:
                    break
                val = safe_call(x)
                evals.append(val)
            return evals

        evals_large = evaluate_population(pop_large)
        evals_med = evaluate_population(pop_med)
        evals_small = evaluate_population(pop_small)
        
        # Update archive with initial evaluations
        update_archive_from_population(pop_large, evals_large)
        update_archive_from_population(pop_med, evals_med)
        update_archive_from_population(pop_small, evals_small)

        # Main Loop: Iterate until budget is exhausted
        # Each iteration performs one generation for each population.
        # We allocate calls dynamically.
        
        while time.time() < deadline and calls < budget:
            # Check if we have enough budget for a full round
            # A full round uses ~ total_pop calls.
            # If budget is low, we might just do local search or stop.
            if budget - calls < total_pop:
                # Use remaining budget for a final local polish or random search
                # Let's just do random search around best
                while time.time() < deadline and calls < budget:
                    x_try = x_best + np.random.normal(0, 0.1, dim)
                    safe_call(x_try)
                break

            # 1. Large Population Update (Global Exploration)
            # DE/rand/1
            # Crossover rate: Low (0.3) to maintain diversity
            # Mutation: F * (x_r1 - x_r2)
            new_large = np.zeros_like(pop_large)
            for i in range(n_large):
                if time.time() > deadline or calls >= budget:
                    break
                # Choose 3 distinct random individuals
                r1, r2, r3 = np.random.choice(n_large, 3, replace=False)
                # Ensure they are not i
                while r1 == i: r1 = np.random.choice(n_large)
                while r2 == i: r2 = np.random.choice(n_large)
                while r3 == i: r3 = np.random.choice(n_large)
                
                F = 0.8 # Mutation factor
                CR = 0.3 # Crossover rate
                
                # Mutation vector
                v = pop_large[r1] + F * (pop_large[r2] - pop_large[r3])
                
                # Crossover
                j_rand = np.random.randint(0, dim)
                u = np.random.rand(dim)
                trial = np.where(u < CR, v, pop_large[i])
                trial[j_rand] = v[j_rand] # Ensure at least one gene from mutant
                
                new_large[i] = trial
            
            # Evaluate new large population
            evals_large_new = evaluate_population(new_large)
            # Selection: Keep the better of old and new
            for i in range(n_large):
                if evals_large_new[i] is not None and (evals_large[i] is None or evals_large_new[i] < evals_large[i]):
                    pop_large[i] = new_large[i]
                    evals_large[i] = evals_large_new[i]
            
            update_archive_from_population(new_large, evals_large_new)

            # 2. Medium Population Update (Basin Identification)
            # DE/best/1 or DE/rand-to-best/1
            # Use archive best to guide
            # Crossover rate: Medium (0.6)
            if time.time() < deadline and calls < budget:
                new_med = np.zeros_like(pop_med)
                # Get best from archive
                if archive:
                    x_guide = archive[0][1]
                else:
                    x_guide = x_best
                    
                for i in range(n_med):
                    if time.time() > deadline or calls >= budget:
                        break
                    r1, r2 = np.random.choice(n_med, 2, replace=False)
                    while r1 == i: r1 = np.random.choice(n_med)
                    while r2 == i: r2 = np.random.choice(n_med)
                    
                    F = 0.5
                    CR = 0.6
                    
                    # Mutation: towards best, plus random difference
                    v = x_guide + F * (pop_med[r1] - pop_med[r2])
                    
                    j_rand = np.random.randint(0, dim)
                    u = np.random.rand(dim)
                    trial = np.where(u < CR, v, pop_med[i])
                    trial[j_rand] = v[j_rand]
                    
                    new_med[i] = trial
                
                evals_med_new = evaluate_population(new_med)
                for i in range(n_med):
                    if evals_med_new[i] is not None and (evals_med[i] is None or evals_med_new[i] < evals_med[i]):
                        pop_med[i] = new_med[i]
                        evals_med[i] = evals_med_new[i]
                
                update_archive_from_population(new_med, evals_med_new)

            # 3. Small Population Update (Fine Tuning)
            # DE/best/1 with high crossover and low mutation
            # Crossover rate: High (0.9)
            if time.time() < deadline and calls < budget:
                new_small = np.zeros_like(pop_small)
                if archive:
                    x_guide = archive[0][1]
                else:
                    x_guide = x_best
                    
                for i in range(n_small):
                    if time.time() > deadline or calls < budget:
                        break
                    r1 = np.random.choice(n_small)
                    while r1 == i: r1 = np.random.choice(n_small)
                    
                    F = 0.3 # Lower mutation for fine tuning
                    CR = 0.9 # High crossover to stay close to base
                    
                    # Mutation: towards best
                    # v = x_guide + F * (pop_small[r1] - pop_small[i]) ? 
                    # Standard DE/rand/1 but biased towards best?
                    # Let's use DE/current-to-best/1
                    v = pop_small[i] + F * (x_guide - pop_small[i]) + 0.1 * (np.random.uniform(-1, 1, dim))
                    
                    j_rand = np.random.randint(0, dim)
                    u = np.random.rand(dim)
                    trial = np.where(u < CR, v, pop_small[i])
                    trial[j_rand] = v[j_rand]
                    
                    new_small[i] = trial
                
                evals_small_new = evaluate_population(new_small)
                for i in range(n_small):
                    if evals_small_new[i] is not None and (evals_small[i] is None or evals_small_new[i] < evals_small[i]):
                        pop_small[i] = new_small[i]
                        evals_small[i] = evals_small_new[i]
                
                update_archive_from_population(new_small, evals_small_new)

        return x_best, f_best

# EVOLVE-BLOCK-END
