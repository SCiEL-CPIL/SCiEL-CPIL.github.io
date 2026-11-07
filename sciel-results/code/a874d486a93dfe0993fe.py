"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Nelder-Mead for f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, budget, deadline):
        n = self.dim
        # Initialize simplex
        # Start at x0, other vertices offset by 0.05 in each dimension
        simplex = [x0.copy()]
        for i in range(n):
            v = x0.copy()
            v[i] += 0.05
            simplex.append(v)
        
        fs = [func(x) for x in simplex]
        calls = n + 1
        
        # Track best
        best_idx = np.argmin(fs)
        best_x = simplex[best_idx].copy()
        best_f = fs[best_idx]
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        max_iter = 500
        
        iter_count = 0
        while calls < budget and iter_count < max_iter and time.time() < deadline:
            iter_count += 1
            
            # Sort simplex by function value
            order = np.argsort(fs)
            simplex = [simplex[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update global best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = simplex[0].copy()
            
            # Check convergence (simplex spread)
            centroid = np.mean(simplex[:-1], axis=0)
            spread = np.linalg.norm(simplex[-1] - centroid)
            if spread < 1e-8:
                break
            
            # Centroid of all but the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            f_worst = fs[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_r < fs[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fs[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fs[-1] = f_r
            elif f_r < fs[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fs[-1] = f_r
            else:
                # Contraction
                # Check if it's outside or inside contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        fs[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fs[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        fs[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fs[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
                if calls >= budget:
                    break

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        budget = self.budget
        n = self.dim
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Phase 1: Multi-Start Nelder-Mead
        # We divide the budget among several random starts to handle the hidden shift/rotation robustly.
        # 5 starts with 100 calls each is a reasonable heuristic for 500 total.
        n_starts = 5
        budget_per_start = budget // n_starts
        
        # Use a fixed seed for reproducibility within the run, but distinct from instance seed?
        # The task says "hidden seeded shift", implying the evaluator sets the seed for the function,
        # but our random numbers for search can be arbitrary.
        rng = np.random.default_rng(42)
        
        for i in range(n_starts):
            if total_calls >= budget or time.time() > deadline:
                break
            
            # Generate random start point in [-5, 5]^n
            x0 = rng.uniform(-5, 5, size=n)
            
            # Ensure we don't exceed budget
            remaining = budget - total_calls
            b_i = min(budget_per_start, remaining)
            
            try:
                bx, bf, used = self._nelder_mead(func, x0, b_i, deadline)
                total_calls += used
                if bf < best_f:
                    best_f = bf
                    best_x = bx
            except Exception:
                pass
        
        # Phase 2: Final Polish with a tight Nelder-Mead from the best point
        if total_calls < budget and time.time() < deadline - 5 and best_x is not None:
            # Use a smaller initial spread for polishing
            x0 = best_x.copy()
            # We need to restart Nelder-Mead with a small spread
            # We can just call _nelder_mead again, but it uses 0.05 spread.
            # To make it tighter, we can temporarily modify the method or just rely on the fact
            # that NM converges. However, 0.05 might be too large if we are very close.
            # Let's create a specialized tight NM call.
            
            def _tight_nelder_mead(func, x0, budget, deadline):
                n = self.dim
                simplex = [x0.copy()]
                for i in range(n):
                    v = x0.copy()
                    v[i] += 0.01  # Smaller spread
                    simplex.append(v)
                
                fs = [func(x) for x in simplex]
                calls = n + 1
                
                best_idx = np.argmin(fs)
                best_x_local = simplex[best_idx].copy()
                best_f_local = fs[best_idx]
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                max_iter = 100
                
                iter_count = 0
                while calls < budget and iter_count < max_iter and time.time() < deadline:
                    iter_count += 1
                    order = np.argsort(fs)
                    simplex = [simplex[i] for i in order]
                    fs = [fs[i] for i in order]
                    
                    if fs[0] < best_f_local:
                        best_f_local = fs[0]
                        best_x_local = simplex[0].copy()
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    spread = np.linalg.norm(simplex[-1] - centroid)
                    if spread < 1e-10:
                        break
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    worst = simplex[-1]
                    f_worst = fs[-1]
                    
                    x_r = centroid + alpha * (centroid - worst)
                    x_r = np.clip(x_r, -5, 5)
                    f_r = func(x_r)
                    calls += 1
                    
                    if f_r < fs[0]:
                        x_e = centroid + gamma * (centroid - worst)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = func(x_e)
                        calls += 1
                        if f_e < f_r:
                            simplex[-1] = x_e
                            fs[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            fs[-1] = f_r
                    elif f_r < fs[-2]:
                        simplex[-1] = x_r
                        fs[-1] = f_r
                    else:
                        if f_r < f_worst:
                            x_c = centroid + rho * (x_r - centroid)
                            x_c = np.clip(x_c, -5, 5)
                            f_c = func(x_c)
                            calls += 1
                            if f_c <= f_r:
                                simplex[-1] = x_c
                                fs[-1] = f_c
                            else:
                                for i in range(1, n + 1):
                                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                    fs[i] = func(simplex[i])
                                    calls += 1
                                    if calls >= budget:
                                        break
                        else:
                            x_c = centroid - rho * (x_r - centroid)
                            x_c = np.clip(x_c, -5, 5)
                            f_c = func(x_c)
                            calls += 1
                            if f_c < f_worst:
                                simplex[-1] = x_c
                                fs[-1] = f_c
                            else:
                                for i in range(1, n + 1):
                                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                    fs[i] = func(simplex[i])
                                    calls += 1
                                    if calls >= budget:
                                        break
                        if calls >= budget:
                            break
                return best_x_local, best_f_local, calls

            remaining = budget - total_calls
            try:
                bx, bf, used = _tight_nelder_mead(func, best_x, remaining, deadline)
                total_calls += used
                if bf < best_f:
                    best_f = bf
                    best_x = bx
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
