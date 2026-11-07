"""Bootstrap spawn: Multi-start population optimizer with periodic-aware refinement for 5D Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based optimizer exploiting Rastrigin periodicity with multi-start and adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return float(func(x))

        def time_ok():
            return calls[0] < budget and time.time() < deadline

        # Phase 1: Coarse random exploration to find diverse basins
        # Rastrigin has periodic minima with period 1 in each dimension
        # Use stratified sampling to cover the landscape more evenly
        xs = []
        fs = []
        best_x = None
        best_f = np.inf

        # Stratified sampling: divide each dimension into cells
        # For 5D, use 2 cells per dimension = 32 initial points
        # Plus some random points for diversity
        n_stratified = 32
        n_random = 28
        n_explore = min(n_stratified + n_random, budget)

        # Generate stratified samples
        for i in range(n_stratified):
            if not time_ok():
                break
            # Decode index i into 5D grid position (2 cells per dim)
            coords = []
            temp_i = i
            for d in range(dim):
                cell = temp_i % 2
                temp_i //= 2
                # Sample within the cell
                offset = rng.uniform(0.1, 0.9)
                val = -5 + (cell + offset) * 5.0
                coords.append(val)
            x = np.array(coords)
            fx = call_func(x)
            xs.append(x)
            fs.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Add random samples
        for i in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            xs.append(x)
            fs.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Multi-start local optimization
        # Select top K candidates from exploration and refine each
        K = min(5, len(xs))
        if K == 0:
            return best_x, best_f

        # Sort by fitness and take top K
        sorted_indices = np.argsort(fs)
        top_indices = sorted_indices[:K]

        # For each top candidate, do periodic-aware local search
        results = []
        for idx in top_indices:
            if not time_ok():
                break
            x = xs[idx].copy()
            fx = fs[idx]

            # Periodic-aware coordinate descent
            # Exploit the fact that Rastrigin has minima at integer shifts
            # Try to snap each coordinate to nearby integer values
            
            # First, do standard coordinate descent with adaptive steps
            for phase in range(4):
                if not time_ok():
                    break

                if phase == 0:
                    step_sizes = [0.5, 0.25, 0.1]
                elif phase == 1:
                    step_sizes = [0.1, 0.05, 0.02]
                elif phase == 2:
                    step_sizes = [0.02, 0.01, 0.005]
                else:
                    step_sizes = [0.005, 0.002, 0.001]

                for ss in step_sizes:
                    if not time_ok():
                        break
                    improved = True
                    while improved and time_ok():
                        improved = False
                        for d in range(dim):
                            if not time_ok():
                                break
                            for sign in [1, -1]:
                                x_trial = x.copy()
                                x_trial[d] += sign * ss
                                x_trial = np.clip(x_trial, -5, 5)
                                fx_trial = call_func(x_trial)
                                if fx_trial < fx:
                                    x = x_trial
                                    fx = fx_trial
                                    improved = True
                                    # Try to continue in same direction
                                    x_trial2 = x.copy()
                                    x_trial2[d] += sign * ss * 0.5
                                    x_trial2 = np.clip(x_trial2, -5, 5)
                                    fx_trial2 = call_func(x_trial2)
                                    if fx_trial2 < fx:
                                        x = x_trial2
                                        fx = fx_trial2
                                    break

            # Periodic snapping: try moving each coordinate to nearest integer
            # or nearby integer offsets (since Rastrigin minima are at integers)
            if time_ok():
                for d in range(dim):
                    if not time_ok():
                        break
                    base = int(np.round(x[d]))
                    # Try base-1, base, base+1
                    for offset in [-1, 0, 1]:
                        target = base + offset
                        if -5 <= target <= 5:
                            x_trial = x.copy()
                            x_trial[d] = target
                            fx_trial = call_func(x_trial)
                            if fx_trial < fx:
                                x = x_trial
                                fx = fx_trial
                                # Fine-tune around the integer
                                for ss in [0.1, 0.05, 0.02, 0.01]:
                                    if not time_ok():
                                        break
                                    for sign in [1, -1]:
                                        x_trial2 = x.copy()
                                        x_trial2[d] += sign * ss
                                        x_trial2 = np.clip(x_trial2, -5, 5)
                                        fx_trial2 = call_func(x_trial2)
                                        if fx_trial2 < fx:
                                            x = x_trial2
                                            fx = fx_trial2

            results.append((fx, x.copy()))

        # Phase 3: Differential Evolution on the refined candidates
        # Use the best refined points as the initial population
        if time_ok() and len(results) > 0:
            # Sort results by fitness
            results.sort(key=lambda t: t[0])
            
            # Create population from top results plus some random perturbations
            pop_size = min(12, budget - calls[0])
            pop = []
            pop_f = []
            
            # Add top refined points
            for fx, x in results[:min(5, pop_size)]:
                pop.append(x.copy())
                pop_f.append(fx)
            
            # Add random perturbations of the best point
            while len(pop) < pop_size and time_ok():
                if not time_ok():
                    break
                x_pert = results[0][1] + rng.standard_normal(dim) * 0.5
                x_pert = np.clip(x_pert, -5, 5)
                fx_pert = call_func(x_pert)
                pop.append(x_pert)
                pop_f.append(fx_pert)
                if fx_pert < best_f:
                    best_f = fx_pert
                    best_x = x_pert.copy()
            
            pop = np.array(pop)
            pop_f = np.array(pop_f)
            
            # DE parameters
            F = 0.8
            CR = 0.5
            
            # DE loop
            while time_ok():
                improved_any = False
                for i in range(len(pop)):
                    if not time_ok():
                        break
                    
                    # Select 3 distinct random indices
                    candidates = list(range(len(pop)))
                    candidates.remove(i)
                    r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                    
                    # Mutation
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    
                    # Crossover
                    j_rand = rng.integers(dim)
                    trial = pop[i].copy()
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                    
                    trial = np.clip(trial, -5, 5)
                    f_trial = call_func(trial)
                    
                    if f_trial < pop_f[i]:
                        pop[i] = trial
                        pop_f[i] = f_trial
                        improved_any = True
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()
                
                # If no improvement in a full pass, break
                if not improved_any:
                    break

        # Phase 4: Final fine-tuning with very small random walk
        if time_ok() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.005
            for i in range(min(30, budget - calls[0])):
                if not time_ok():
                    break
                step = sigma * rng.standard_normal(dim)
                x_trial = np.clip(x + step, -5, 5)
                fx_trial = call_func(x_trial)
                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    best_x = x.copy()
                    best_f = fx
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = call_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
