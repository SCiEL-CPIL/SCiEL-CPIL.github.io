"""Multi-start L-BFGS-B with random initialization for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with random initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def evaluate(x):
            nonlocal calls
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            record(x, f)
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Random initialization with LHS-like spread
        # Evaluate a diverse set of initial points
        n_initial = min(30, budget // 3)
        if n_initial > 0:
            # Latin Hypercube sampling for better coverage
            indices = np.arange(n_initial)
            for d in range(dim):
                perm = rng.permutation(n_initial)
                # Create stratified samples
                points = np.zeros((n_initial, dim))
                for i in range(n_initial):
                    # Use stratified uniform sampling
                    lo = -5.0 + i * (10.0 / n_initial)
                    hi = -5.0 + (i + 1) * (10.0 / n_initial)
                    points[i, d] = rng.uniform(lo, hi)
                # Actually, let's do proper LHS
                # Redo: for each dimension, permute the stratified intervals
            # Simpler: just use stratified sampling per dimension
            lhs_points = np.zeros((n_initial, dim))
            for d in range(dim):
                # Create n_initial stratified intervals
                edges = np.linspace(-5.0, 5.0, n_initial + 1)
                # For each interval, pick a random point
                for i in range(n_initial):
                    lhs_points[i, d] = rng.uniform(edges[i], edges[i + 1])
            
            # Shuffle to decorrelate
            perm = rng.permutation(n_initial)
            lhs_points = lhs_points[perm]
            
            for i in range(n_initial):
                if not time_ok() or not budget_ok():
                    break
                evaluate(lhs_points[i])

        # Phase 2: Multi-start L-BFGS-B from best initial points
        # Keep track of evaluated points and their values
        eval_points = []
        eval_values = []
        
        # Re-evaluate to track (we already have them in best tracking, but let's collect)
        # Actually, let's just use the best point found so far and a few random starts
        
        # Get the current best
        if best_x is not None:
            start_points = [best_x.copy()]
        else:
            start_points = [np.zeros(dim)]
        
        # Add a few more diverse starting points
        n_starts = min(8, budget // 10)
        for _ in range(n_starts):
            if not time_ok() or not budget_ok():
                break
            x_start = rng.uniform(-4, 4, dim)
            start_points.append(x_start)
        
        # Sort start points by their function values (evaluate if not done)
        # Actually, let's just run L-BFGS-B from each start point
        # Use the best one first, then others
        
        # For each start point, run L-BFGS-B with limited budget
        max_iter_per_start = 50
        for start_idx, x0 in enumerate(start_points):
            if not time_ok() or not budget_ok():
                break
            
            # Check if we've already evaluated this point (approximate check)
            # Just run L-BFGS-B
            def objective(x):
                return evaluate(x)
            
            # L-BFGS-B with bounds
            bounds = [(-5.0, 5.0)] * dim
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter_per_start, 'maxfun': min(50, budget - calls)}
                )
                if result.success or result.fun < best_f:
                    record(result.x, result.fun)
            except Exception:
                pass
            
            # If we've used a lot of budget, stop
            if calls > budget * 0.7:
                break

        # Phase 3: Final refinement with Nelder-Mead from the best point
        if best_x is not None and time_ok() and budget_ok():
            x0 = best_x.copy()
            f0 = best_f
            alpha = 0.1
            
            # Build simplex
            simplex = [x0.copy()]
            f_simplex = [f0]
            
            for i in range(dim):
                if not time_ok() or not budget_ok():
                    break
                x_i = x0.copy()
                x_i[i] += alpha
                x_i = np.clip(x_i, -5, 5)
                f_i = evaluate(x_i)
                simplex.append(x_i)
                f_simplex.append(f_i)
            
            if len(simplex) >= 2:
                gamma = 1.0
                rho = 2.0
                sigma_nm = 0.5
                shrink = 0.5
                phase_best_x = best_x.copy()
                phase_best_f = best_f
                
                max_nm_iter = min(100, (budget - calls) // 2)
                for _ in range(max_nm_iter):
                    if not time_ok() or not budget_ok():
                        break
                    
                    indices = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in indices]
                    f_simplex = [f_simplex[i] for i in indices]
                    
                    x_best = simplex[0]
                    f_best = f_simplex[0]
                    x_worst = simplex[-1]
                    f_worst = f_simplex[-1]
                    x_second_worst = simplex[-2]
                    f_second_worst = f_simplex[-2]
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    x_r = centroid + gamma * (centroid - x_worst)
                    x_r = np.clip(x_r, -5, 5)
                    if not time_ok() or not budget_ok():
                        break
                    f_r = evaluate(x_r)
                    if f_r < phase_best_f:
                        phase_best_f = f_r
                        phase_best_x = x_r.copy()
                    
                    if f_r < f_best:
                        x_e = centroid + rho * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        if not time_ok() or not budget_ok():
                            break
                        f_e = evaluate(x_e)
                        if f_e < phase_best_f:
                            phase_best_f = f_e
                            phase_best_x = x_e.copy()
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_simplex[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_simplex[-1] = f_r
                    elif f_r < f_second_worst:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                    else:
                        if f_r < f_worst:
                            x_c = centroid + sigma_nm * (x_r - centroid)
                        else:
                            x_c = centroid - sigma_nm * (x_worst - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        
                        if not time_ok() or not budget_ok():
                            break
                        f_c = evaluate(x_c)
                        if f_c < phase_best_f:
                            phase_best_f = f_c
                            phase_best_x = x_c.copy()
                        
                        if f_c < min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            for i in range(1, len(simplex)):
                                if not time_ok() or not budget_ok():
                                    break
                                simplex[i] = x_best + shrink * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_simplex[i] = evaluate(simplex[i])
                                if f_simplex[i] < phase_best_f:
                                    phase_best_f = f_simplex[i]
                                    phase_best_x = simplex[i].copy()
                            if not time_ok():
                                break
                    
                    f_spread = np.max(f_simplex) - np.min(f_simplex)
                    if f_spread < 1e-14:
                        break
                
                if phase_best_f < best_f:
                    best_f = phase_best_f
                    best_x = phase_best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
