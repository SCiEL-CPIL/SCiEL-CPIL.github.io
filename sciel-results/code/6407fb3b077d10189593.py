import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_f, best_x
            fx = func(np.clip(x, -5, 5))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = np.clip(x, -5, 5).copy()
            return fx

        # Phase 1: Minimal initialization - center is strong for sphere
        start_x = np.zeros(dim)
        fx = call_func(start_x)
        
        # One quick probe to check if we're already near optimum
        probe_x = start_x + rng.uniform(-1, 1, dim)
        probe_x = np.clip(probe_x, -4.5, 4.5)
        fx_probe = call_func(probe_x)
        if fx_probe < fx:
            start_x = probe_x.copy()
            best_f = fx_probe
        else:
            start_x = np.zeros(dim)
            best_f = fx

        # Phase 2: Nelder-Mead simplex method with adaptive step size
        simplex = [start_x.copy()]
        simplex_f = [best_f]

        # Start with moderate step size, will adapt
        step_size = 1.0
        for i in range(dim):
            v = start_x.copy()
            v[i] += step_size
            fv = call_func(v)
            simplex.append(v.copy())
            simplex_f.append(fv)

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5  # shrink
        max_iter = 350
        tol = 1e-12
        no_improve_count = 0
        best_f_nm = best_f
        step_shrink_counter = 0

        for iteration in range(max_iter):
            if calls_made >= budget or time.time() > deadline:
                break

            # Sort simplex by fitness
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]

            # Check convergence
            if abs(simplex_f[-1] - simplex_f[0]) < tol:
                break

            # Track no-improvement for restart
            if simplex_f[0] < best_f_nm - 1e-15:
                no_improve_count = 0
                best_f_nm = simplex_f[0]
            else:
                no_improve_count += 1

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = call_func(x_ref)

            if f_ref < simplex_f[0]:
                # Expansion
                x_exp = centroid + gamma * (centroid - simplex[-1])
                f_exp = call_func(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    simplex_f[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                # Simple reflection
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Contraction
                if f_ref < simplex_f[-1]:
                    # Outside contraction
                    x_con = centroid + 0.5 * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - 0.5 * (centroid - simplex[-1])
                f_con = call_func(x_con)
                if f_con < min(f_ref, simplex_f[-1]):
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex_f[i] = call_func(simplex[i])

            # Adaptive step size: shrink when making steady progress
            if no_improve_count == 0 and step_shrink_counter >= 10:
                step_size = max(0.1, step_size * 0.8)
                step_shrink_counter = 0
            if no_improve_count == 0:
                step_shrink_counter += 1

            # Adaptive restart: if stuck, try a new random point closer to current best
            if no_improve_count >= 20 and calls_made < budget - 15:
                # Try points near current best with decreasing perturbation
                for attempt in range(3):
                    if calls_made >= budget - 15:
                        break
                    perturbation = max(0.2, 1.0 * (0.7 ** attempt))
                    x_new = best_x + rng.normal(0, perturbation, dim)
                    x_new = np.clip(x_new, -4.5, 4.5)
                    f_new = call_func(x_new)
                    if f_new < simplex_f[0]:
                        simplex = [x_new.copy()]
                        simplex_f = [f_new]
                        new_step = max(0.3, step_size * 0.6)
                        for i in range(dim):
                            v = x_new.copy()
                            v[i] += new_step
                            fv = call_func(v)
                            simplex.append(v.copy())
                            simplex_f.append(fv)
                        step_size = new_step
                        no_improve_count = 0
                        best_f_nm = f_new
                        step_shrink_counter = 0
                        break
                else:
                    # If all attempts failed, try shrinking step size more aggressively
                    step_size = max(0.1, step_size * 0.5)
                    no_improve_count = 10

        # Phase 3: Coordinate descent polish for high precision
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline - 5:
            # Use coordinate descent with adaptive step sizes
            x_cur = best_x.copy()
            f_cur = best_f
            cd_step = max(0.05, step_size * 0.3)
            
            for _ in range(min(remaining // 2, 100)):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                improved = False
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    # Try positive direction
                    x_try = x_cur.copy()
                    x_try[d] += cd_step
                    x_try = np.clip(x_try, -4.5, 4.5)
                    f_try = call_func(x_try)
                    if f_try < f_cur:
                        x_cur = x_try
                        f_cur = f_try
                        improved = True
                        if f_cur < best_f:
                            best_f = f_cur
                            best_x = x_cur.copy()
                    
                    # Try negative direction
                    x_try = x_cur.copy()
                    x_try[d] -= cd_step
                    x_try = np.clip(x_try, -4.5, 4.5)
                    f_try = call_func(x_try)
                    if f_try < f_cur:
                        x_cur = x_try
                        f_cur = f_try
                        improved = True
                        if f_cur < best_f:
                            best_f = f_cur
                            best_x = x_cur.copy()
                
                if not improved:
                    # Shrink step size if no improvement
                    cd_step *= 0.5
                    if cd_step < 1e-6:
                        break

        return best_x, best_f
