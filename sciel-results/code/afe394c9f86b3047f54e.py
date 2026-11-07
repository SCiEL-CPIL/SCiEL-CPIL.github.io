"""Multi-Start Nelder-Mead optimizer for BBOB f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Optimizer using Multi-Start Nelder-Mead.
    Phase 1: Global random sampling to find promising starting points.
    Phase 2: Run Nelder-Mead from the best starting points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin for 330s limit
        
        dim = self.dim
        total_budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        rng = np.random.default_rng(42)

        def record(x, fx):
            nonlocal n_calls, best_f, best_x
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float, copy=True)

        # Phase 1: Global Exploration
        n_explore = min(30, total_budget // 10)
        candidates = []
        
        for _ in range(n_explore):
            if n_calls >= total_budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            record(x, fx)
            candidates.append((fx, x))
        
        if best_x is None:
            return (np.zeros(dim), 0.0)

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Select top unique candidates for restarts
        starts = []
        min_dist = 1.0
        for fx, x in candidates:
            if len(starts) >= 5:
                break
            is_distinct = True
            for _, sx in starts:
                if np.linalg.norm(x - sx) < min_dist:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append((fx, x))
        
        if len(starts) < 3:
            for fx, x in candidates:
                if (fx, x) not in starts:
                    starts.append((fx, x))
                if len(starts) >= 3:
                    break

        # Phase 2: Multi-Start Nelder-Mead
        for i, (f_start, x_start) in enumerate(starts):
            if n_calls >= total_budget or time.time() > deadline:
                break
            
            # Initialize simplex
            x0 = np.array(x_start, dtype=float)
            simplex = [x0.copy()]
            f0 = func(x0)
            n_calls += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            simplex.append(x0)
            simplex[0] = x0.copy()
            simplex[1] = x0.copy()
            
            # Build initial simplex
            simplex = [x0.copy()]
            for j in range(dim):
                point = x0.copy()
                point[j] += 0.5
                simplex.append(point)
            
            # Evaluate initial simplex
            simplex_f = []
            for j in range(dim + 1):
                if n_calls >= total_budget or time.time() > deadline:
                    break
                fx = func(simplex[j])
                n_calls += 1
                simplex_f.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = simplex[j].copy()
            
            if len(simplex_f) < dim + 1:
                continue

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            max_iter = 200
            tol = 1e-10
            
            for iteration in range(max_iter):
                if n_calls >= total_budget or time.time() > deadline:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Check convergence
                if abs(simplex_f[-1] - simplex_f[0]) < tol:
                    break
                
                # Centroid (excluding worst point)
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                n_calls += 1
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    n_calls += 1
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                    if f_exp < f_ref:
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
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    n_calls += 1
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            if n_calls >= total_budget or time.time() > deadline:
                                break
                            f_shr = func(simplex[j])
                            n_calls += 1
                            simplex_f[j] = f_shr
                            if f_shr < best_f:
                                best_f = f_shr
                                best_x = simplex[j].copy()
                        if n_calls >= total_budget or time.time() > deadline:
                            break
                        continue
        
        return best_x, best_f
# EVOLVE-BLOCK-END
