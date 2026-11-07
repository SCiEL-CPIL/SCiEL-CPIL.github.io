import numpy as np
import time
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = (-5.0, 5.0)
        
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin under 330s
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        def track(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                raise TimeoutError()
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, bounds[0], bounds[1])
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Coarse grid search sorted by distance to origin
        # Minimizer is in [-4,4]^5, so grid at -4, 0, 4 covers well
        levels = [-4.0, 0.0, 4.0]
        grid_points = list(itertools.product(levels, repeat=dim))
        
        # Sort by distance to origin for early progress
        origin = np.zeros(dim)
        grid_points.sort(key=lambda p: np.linalg.norm(np.array(p) - origin))
        
        for p in grid_points:
            if calls_made >= budget or time.time() > deadline:
                break
            try:
                track(np.array(p, dtype=np.float64))
            except Exception:
                break
        
        # Phase 2: Nelder-Mead from best points
        def nelder_mead(x0, max_evals):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return
            
            # Simplex initialization
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5 if x0[i] < 0 else -0.5
                simplex.append(np.clip(xi, bounds[0], bounds[1]))
            
            fvals = []
            for x in simplex:
                if calls_made >= budget or time.time() > deadline:
                    break
                try:
                    fvals.append(track(x))
                except Exception:
                    break
            
            if len(fvals) < dim + 1:
                return
            
            for _ in range(max_evals):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Convergence check
                if max(fvals) - min(fvals) < 1e-10:
                    break
                if np.max(np.std(simplex, axis=0)) < 1e-7:
                    break
                
                x_worst = simplex[-1]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, bounds[0], bounds[1])
                try:
                    f_ref = track(x_ref)
                except Exception:
                    break
                
                if f_ref < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = x_centroid + gamma * (x_centroid - x_worst)
                    x_exp = np.clip(x_exp, bounds[0], bounds[1])
                    try:
                        f_exp = track(x_exp)
                    except Exception:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = x_centroid + rho * (x_worst - x_centroid)
                    x_con = np.clip(x_con, bounds[0], bounds[1])
                    try:
                        f_con = track(x_con)
                    except Exception:
                        break
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls_made >= budget or time.time() > deadline:
                                break
                            simplex[i] = (simplex[0] + simplex[i]) / 2
                            simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                            try:
                                fvals[i] = track(simplex[i])
                            except Exception:
                                break
        
        # Collect top grid points for restarts
        # We need to re-evaluate to get sorted results, but we can use our tracking
        # Instead, let's do NM from the current best and a few other good points
        
        # Restart from best found
        if best_x is not None and calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            # Use ~60% of remaining for best, split rest among other candidates
            nm_budget_best = max(20, int(remaining * 0.5))
            nelder_mead(best_x, nm_budget_best)
        
        # Try a few more restarts from different regions if budget allows
        if calls_made < budget - 10 and time.time() < deadline:
            rng = np.random.default_rng(42)
            # Try 2-3 more random restarts in the box
            for _ in range(3):
                if calls_made >= budget - 10 or time.time() > deadline:
                    break
                x0 = rng.uniform(-4, 4, dim)
                nelder_mead(x0, 30)
        
        return best_x, best_f
