"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start local search with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = func(x)
            calls_made += 1
            return f
        
        # Phase 1: Random sampling to find good basins
        n_samples = min(30, self.budget // 10)
        candidates = []
        for _ in range(n_samples):
            if time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, self.dim)
            f = safe_call(x)
            if f is not None:
                candidates.append((f, x))
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
        
        # Sort by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local optimization from top candidates
        remaining_budget = self.budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Allocate budget to top candidates
        n_local = min(len(candidates), max(1, remaining_budget // 40))
        budget_per_local = max(10, remaining_budget // max(1, n_local))
        
        for i in range(n_local):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break
            
            f0, x0 = candidates[i]
            x = x0.copy()
            fx = f0
            
            # Nelder-Mead with adaptive step
            simplex = [x.copy()]
            for j in range(self.dim):
                dx = x.copy()
                dx[j] += 0.5
                simplex.append(dx)
            
            # Evaluate simplex
            simplex_f = [fx]
            for j in range(1, self.dim + 1):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                fj = safe_call(simplex[j])
                if fj is not None:
                    simplex_f.append(fj)
                else:
                    simplex_f.append(np.inf)
            
            # Nelder-Mead iterations
            max_iter = min(budget_per_local - (self.dim + 1), 30)
            for _ in range(max_iter):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = safe_call(x_ref)
                if f_ref is None:
                    break
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = safe_call(x_exp)
                    if f_exp is not None and f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = safe_call(x_con)
                    if f_con is not None and f_con < simplex_f[-1]:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for j in range(1, self.dim + 1):
                            if calls_made >= self.budget:
                                break
                            simplex[j] = (simplex[0] + simplex[j]) / 2
                            fj = safe_call(simplex[j])
                            if fj is not None:
                                simplex_f[j] = fj
                            else:
                                simplex_f[j] = np.inf
                
                # Track best
                idx_min = np.argmin(simplex_f)
                if simplex_f[idx_min] < best_f:
                    best_f = simplex_f[idx_min]
                    best_x = simplex[idx_min].copy()
        
        # Phase 3: Polish with small random steps around best
        if best_x is not None and calls_made < self.budget:
            remaining = self.budget - calls_made
            for _ in range(min(remaining, 50)):
                if time.time() > deadline or calls_made >= self.budget:
                    break
                step = 0.1 * rng.standard_normal(self.dim)
                x_try = np.clip(best_x + step, -5, 5)
                f_try = safe_call(x_try)
                if f_try is not None and f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = safe_call(best_x) or 0.0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
