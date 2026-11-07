# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for RE37 (4 variables, 3 objectives, bounds [0,1]).
    Strategy:
      1. Phase 1: Halton sequence initialization for uniform space-filling.
      2. Phase 2: Iterative farthest-point sampling in objective space to cover the Pareto front.
      3. Phase 3: Differential Evolution (DE) for exploitation with larger guided steps.
      4. Deadline guard to stay within 300s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        rng = np.random.RandomState(42)

        # Archive of (x, f)
        archive_x = []
        archive_f = []
        eval_count = 0

        def is_dominated(f, archive_f_list):
            """Check if f is dominated by any point in archive_f_list."""
            for af in archive_f_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def add_to_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points."""
            nonlocal archive_x, archive_f
            # Check if new point is dominated
            if is_dominated(f, archive_f):
                return
            
            # Remove points from archive that are dominated by new point
            new_archive_x = []
            new_archive_f = []
            for ax, af in zip(archive_x, archive_f):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_x.append(ax)
                    new_archive_f.append(af)
            
            new_archive_x.append(x)
            new_archive_f.append(f)
            
            archive_x = new_archive_x
            archive_f = new_archive_f

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            if time.time() > deadline:
                return None
            fx = np.array(problem(x), dtype=float)
            eval_count += 1
            return fx

        def halton_sequence(n, dim):
            """Generate n points in [0,1]^dim using Halton sequence."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2  # Prime base: 2, 3, 5, 7, ...
                for i in range(n):
                    f = 1.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        i_val = (i_val - 1) // base
                    points[i, d] = f
            return points

        # --- Phase 1: Halton Sequence Initialization ---
        # Use 20% of budget for initial space-filling
        n_init = min(40, int(budget * 0.2))
        
        if n_init > 0:
            halton_pts = halton_sequence(n_init, dim)
            halton_pts = lo + halton_pts * (hi - lo)
            
            for i in range(n_init):
                x = halton_pts[i]
                f = evaluate(x)
                if f is None:
                    break
                add_to_archive(x, f)

        # --- Phase 2: Farthest-Point Sampling in Objective Space ---
        # Use 30% of budget for covering the front
        fps_budget = int(budget * 0.3)
        
        while eval_count < fps_budget and time.time() < deadline:
            if not archive_f:
                break
            
            # If we have very few points, just sample randomly
            if len(archive_f) < 5:
                x = lo + rng.random(dim) * (hi - lo)
            else:
                # Normalize objective space for distance calculation
                f_arr = np.array(archive_f)
                f_min = f_arr.min(axis=0)
                f_max = f_arr.max(axis=0)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1.0
                f_norm = (f_arr - f_min) / f_range
                
                # Select next point: maximize minimum distance to existing points in normalized objective space
                # We do this by proposing random candidates and picking the one with max min-distance
                n_candidates = 50
                best_candidate = None
                best_min_dist = -1
                
                for _ in range(n_candidates):
                    x_cand = lo + rng.random(dim) * (hi - lo)
                    f_cand = evaluate(x_cand)
                    if f_cand is None:
                        break
                    
                    # Check if dominated
                    if is_dominated(f_cand, archive_f):
                        continue
                    
                    # Calculate normalized distance
                    f_cand_norm = (f_cand - f_min) / f_range
                    dists = np.sqrt(np.sum((f_norm - f_cand_norm) ** 2, axis=1))
                    min_dist = np.min(dists)
                    
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_candidate = (x_cand, f_cand)
                
                if best_candidate is not None:
                    x, f = best_candidate
                    add_to_archive(x, f)
                else:
                    # No non-dominated candidate found, break to avoid wasting budget
                    break

        # --- Phase 3: Differential Evolution (DE) ---
        # Use remaining budget for exploitation
        de_budget = budget - eval_count
        
        if de_budget > 0 and archive_x:
            # Population size for DE
            pop_size = min(10, len(archive_x))
            if pop_size < 2:
                # If archive is too small, fill with random points first
                while len(archive_x) < 2 and eval_count < de_budget:
                    x = lo + rng.random(dim) * (hi - lo)
                    f = evaluate(x)
                    if f is None:
                        break
                    add_to_archive(x, f)
                if len(archive_x) < 2:
                    return archive_x[0] if archive_x else lo
            
            # DE parameters
            F = 0.8  # Scale factor
            CR = 0.9  # Crossover rate
            
            # Initial population: sample from archive
            de_pop_x = []
            de_pop_f = []
            
            for _ in range(pop_size):
                idx = rng.randint(len(archive_x))
                de_pop_x.append(archive_x[idx].copy())
                de_pop_f.append(archive_f[idx].copy())
            
            de_evals = 0
            max_de_evals = de_budget
            
            while de_evals < max_de_evals and time.time() < deadline:
                # For each individual in population, create a mutant
                for i in range(len(de_pop_x)):
                    if de_evals >= max_de_evals or time.time() > deadline:
                        break
                    
                    # Select two distinct random indices different from i
                    idx1 = rng.randint(len(de_pop_x))
                    while idx1 == i:
                        idx1 = rng.randint(len(de_pop_x))
                    idx2 = rng.randint(len(de_pop_x))
                    while idx2 == i or idx2 == idx1:
                        idx2 = rng.randint(len(de_pop_x))
                    
                    # Differential mutation: v = x_i + F * (x_idx1 - x_idx2)
                    v = de_pop_x[i] + F * (de_pop_x[idx1] - de_pop_x[idx2])
                    v = np.clip(v, lo, hi)
                    
                    # Evaluate mutant
                    f_v = evaluate(v)
                    if f_v is None:
                        break
                    de_evals += 1
                    
                    # Check if mutant is non-dominated
                    if not is_dominated(f_v, archive_f):
                        # Add to archive
                        add_to_archive(v, f_v)
                        
                        # Also update DE population if it's better than current individual
                        # (using simple dominance check)
                        if np.all(f_v <= de_pop_f[i]) and np.any(f_v < de_pop_f[i]):
                            de_pop_x[i] = v
                            de_pop_f[i] = f_v
                    else:
                        # Mutant is dominated, but we can still use it for diversity
                        # with some probability
                        if rng.random() < 0.1:
                            de_pop_x[i] = v
                            de_pop_f[i] = f_v

        # Final check: ensure we return something valid
        if not archive_x:
            return lo
            
        return archive_x[0]

# EVOLVE-BLOCK-END
