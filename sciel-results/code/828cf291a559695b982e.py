"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Parallel asynchronous Nelder-Mead simplex ensemble for Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Number of simplexes to maintain
        n_simplexes = 8
        
        # Initialize simplexes at random locations in [-4, 4]^dim
        # Using random initialization helps cover the space more effectively than fixed corners
        simplex_centers = [rng.uniform(-4, 4, dim) for _ in range(n_simplexes)]
        
        # Initial simplex size
        init_step = 0.5 
        
        simplexes = []
        simplex_fvals = []
        call_count = 0
        
        # Helper to evaluate and track calls
        def eval_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            fx = func(x)
            call_count += 1
            return fx

        # Initialize each simplex
        for center in simplex_centers:
            simplex = [center.copy()]
            for i in range(dim):
                v = center.copy()
                v[i] += init_step
                simplex.append(v)
            simplex = np.array(simplex)
            simplex = np.clip(simplex, -5, 5)
            simplexes.append(simplex)
            
            # Evaluate vertices
            fvals = np.array([eval_func(v) for v in simplex])
            simplex_fvals.append(fvals)
            
        # Global best tracking
        best_x = None
        best_f = np.inf
        
        for i in range(n_simplexes):
            min_idx = np.argmin(simplex_fvals[i])
            f_min = simplex_fvals[i][min_idx]
            if f_min < best_f:
                best_f = f_min
                best_x = simplexes[i][min_idx].copy()

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        stagnation_limit = 12  # Steps without improvement before restart
        stagnation_counters = [0] * n_simplexes
        simplex_best_f = [np.min(fv) for fv in simplex_fvals]
        
        # Track simplex age/restarts to potentially shrink search space or change strategy
        restart_counts = [0] * n_simplexes
        
        # Main loop
        # We use a round-robin approach but with immediate restarts for stagnant simplexes
        # This ensures we don't waste budget on dead ends
        while call_count < budget:
            for i in range(n_simplexes):
                if call_count >= budget:
                    break
                
                simplex = simplexes[i]
                fvals = simplex_fvals[i]
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Check stagnation
                current_best = fvals[0]
                if current_best >= simplex_best_f[i] - 1e-10:
                    stagnation_counters[i] += 1
                else:
                    simplex_best_f[i] = current_best
                    stagnation_counters[i] = 0
                    if current_best < best_f:
                        best_f = current_best
                        best_x = simplex[0].copy()
                
                # If stagnant, perform a full restart of this simplex
                if stagnation_counters[i] > stagnation_limit:
                    # Generate a new random simplex
                    # As restarts increase, we might want to search closer to the best found so far?
                    # Or just random. Random is safer for Rastrigin.
                    # Let's try a mix: 50% random, 50% near best
                    if restart_counts[i] > 2 and rng.random() < 0.5:
                         # Search near current global best
                         center = best_x + rng.normal(0, 1.0, dim)
                    else:
                         center = rng.uniform(-4, 4, dim)
                    
                    restart_counts[i] += 1
                    new_simplex = [center.copy()]
                    for k in range(dim):
                        v = center.copy()
                        v[k] += 0.5 # Smaller step for refinement? Or keep 1.0?
                        new_simplex.append(v)
                    new_simplex = np.array(new_simplex)
                    new_simplex = np.clip(new_simplex, -5, 5)
                    
                    # Evaluate new simplex vertices
                    # We need to be careful with budget
                    new_fvals = []
                    for v in new_simplex:
                        if call_count >= budget:
                            # If we run out of budget, break out of everything
                            new_fvals.append(np.inf)
                        else:
                            new_fvals.append(eval_func(v))
                            
                    if call_count >= budget:
                        break
                        
                    simplexes[i] = new_simplex
                    simplex_fvals[i] = np.array(new_fvals)
                    simplex_best_f[i] = np.min(new_fvals)
                    stagnation_counters[i] = 0
                    
                    # Update global best if the new simplex has a better point
                    min_idx_new = np.argmin(new_fvals)
                    if new_fvals[min_idx_new] < best_f:
                        best_f = new_fvals[min_idx_new]
                        best_x = new_simplex[min_idx_new].copy()
                        
                    continue
                
                # Standard Nelder-Mead step
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = fvals[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = eval_func(x_r)
                if call_count >= budget:
                    break
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = eval_func(x_e)
                    if call_count >= budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = eval_func(x_c)
                    if call_count >= budget:
                        break
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink: only shrink non-best vertices
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            if call_count >= budget:
                                break
                            fvals[j] = eval_func(simplex[j])
                        
                # Update global best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()
                
                simplexes[i] = simplex
                simplex_fvals[i] = fvals

        return best_x, best_f
# EVOLVE-BLOCK-END
