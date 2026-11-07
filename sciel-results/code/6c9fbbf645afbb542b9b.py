import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 325.0  # safety margin before 330s
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Track all evaluated points for hypervolume
        all_X = []
        all_F = []
        used = 0

        def is_pareto_better(f_new, f_existing_list):
            """Check if f_new is not dominated by any point in the list,
            and dominates at least one point in the list (or is new)."""
            for f in f_existing_list:
                # If f dominates f_new, reject
                if np.all(f <= f_new) and np.any(f < f_new):
                    return False
            # Check if f_new dominates any existing point
            for f in f_existing_list:
                if np.all(f_new <= f) and np.any(f_new < f):
                    return True
            return True  # non-dominated, add it

        def evaluate(x):
            nonlocal used
            f = np.array(problem(x), dtype=float)
            used += 1
            return f

        # Strategy: multi-start with local search
        # Each start: pick random point, then do local random-walk search
        # Accept moves that improve Pareto front (non-dominated)
        
        # Phase 1: Wide random sampling to get a rough idea of the front
        n_init = min(50, budget // 4)
        for _ in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            f = evaluate(x)
            all_X.append(x)
            all_F.append(f)

        # Build initial non-dominated set
        def get_non_dominated(F):
            """Return indices of non-dominated points."""
            n = len(F)
            nd = np.zeros(n, dtype=bool)
            for i in range(n):
                if nd[i]:
                    continue
                dominated = False
                for j in range(n):
                    if i == j or nd[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd[i] = True
            return [i for i in range(n) if nd[i]]

        # Local search from a starting point
        def local_search(x_start, f_start, max_steps):
            """Perform local random-walk search. Return list of (x, f) for new points found."""
            best_x = x_start.copy()
            best_f = f_start.copy()
            new_points = []
            
            for step in range(max_steps):
                if used >= budget or time.time() > deadline:
                    break
                
                # Random perturbation
                scale = 0.1 * (hi - lo)
                x_new = best_x + np.random.randn(dim) * scale
                x_new = np.clip(x_new, lo, hi)
                
                f_new = evaluate(x_new)
                all_X.append(x_new)
                all_F.append(f_new)
                
                # Accept if it's non-dominated relative to current best
                # Simple acceptance: if it improves one objective without worsening the other too much
                improves = False
                for i in range(n_obj):
                    if f_new[i] < best_f[i] - 1e-10:
                        # Check it doesn't worsen other objectives too much
                        other_worse = False
                        for j in range(n_obj):
                            if j != i and f_new[j] > best_f[j] * 1.1 + 1e-10:
                                other_worse = True
                                break
                        if not other_worse:
                            improves = True
                            break
                
                if improves or (np.random.random() < 0.3):  # Some exploration
                    best_x = x_new
                    best_f = f_new
                    new_points.append((x_new, f_new))
            
            return new_points

        # Phase 2: Multi-start local search
        n_starts = max(10, budget // 15)
        steps_per_start = max(5, (budget - used) // n_starts)
        
        # Get current non-dominated points as seeds for local search
        nd_indices = get_non_dominated(all_F)
        
        for start_idx in range(n_starts):
            if used >= budget or time.time() > deadline:
                break
            
            # Choose a starting point: mix of random and from non-dominated set
            if nd_indices and np.random.random() < 0.5:
                # Pick from non-dominated set
                idx = nd_indices[np.random.randint(len(nd_indices))]
                x_start = all_X[idx].copy()
                f_start = all_F[idx].copy()
            else:
                x_start = lo + (hi - lo) * np.random.random(dim)
                f_start = evaluate(x_start)
                all_X.append(x_start)
                all_F.append(f_start)
            
            # Run local search
            local_search(x_start, f_start, steps_per_start)
        
        # Phase 3: If budget remains, do more random sampling
        while used < budget and time.time() < deadline:
            x = lo + (hi - lo) * np.random.random(dim)
            f = evaluate(x)
            all_X.append(x)
            all_F.append(f)
        
        return all_X
