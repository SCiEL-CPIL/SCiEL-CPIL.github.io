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
        deadline = start_time + 280.0

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

        # Phase 1: Deterministic grid-search warm-up to bracket the minimum
        # Use a coarse grid over [-4, 4]^5 with strategic points
        # For dim=5, we can't do full grid, so use axis-aligned + diagonal probes
        
        # Start at center
        center = np.zeros(dim)
        f_center = call_func(center)
        
        # Axis-aligned probes: test each dimension independently
        # This helps identify which dimensions need exploration
        grid_points = [center.copy()]
        
        # Probe along each axis at a few levels
        for d in range(dim):
            for level in [-3.0, -1.5, 1.5, 3.0]:
                x = center.copy()
                x[d] = level
                if calls_made < budget and time.time() < deadline:
                    f = call_func(x)
                    grid_points.append((f, x.copy()))
        
        # Add some diagonal probes
        for scale in [1.0, 2.0, 3.0]:
            for sign in [1, -1]:
                x = sign * scale * np.ones(dim)
                if calls_made < budget and time.time() < deadline:
                    f = call_func(x)
                    grid_points.append((f, x.copy()))
        
        # Add a few random points for robustness
        for _ in range(6):
            if calls_made < budget and time.time() < deadline:
                x = rng.uniform(-4, 4, dim)
                f = call_func(x)
                grid_points.append((f, x.copy()))
        
        # Sort by fitness and pick the best as starting point
        grid_points.sort(key=lambda t: t[0])
        start_x = grid_points[0][1].copy()
        start_f = grid_points[0][0]
        
        # Determine a bounding box from the grid results for CMA-ES initialization
        # Use the spread of good points to set initial sigma
        good_points = [p[1] for p in grid_points[:min(10, len(grid_points))]]
        if len(good_points) > 1:
            good_arr = np.array(good_points)
            spread = np.std(good_arr, axis=0)
            init_sigma = max(0.5, np.mean(spread) * 0.5)
        else:
            init_sigma = 1.0
        
        # Phase 2: CMA-ES refinement
        # Implement a lightweight CMA-ES
        pop_size = max(4 + int(3 * np.log(dim)), 8)  # ~11 for dim=5
        c1 = (1 + 2 / ((dim + 1) ** 2 - 2)) / (dim + 1) ** 2  # cmu=0.25
        cmu = min(c1, (2 * (2 - 1 / (dim + 1)) / (2 + dim + 2 / (dim + 1))) / (dim + 1) ** 2)
        cs = (mu_eff + 1.5 + cmu) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        mu_eff = 0.5 * pop_size  # for mu = pop_size/2
        
        mean = start_x.copy()
        sigma = init_sigma
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute Cholesky
        try:
            C_inv_sqrt = np.linalg.inv(C_sqrt)
        except:
            C_inv_sqrt = np.eye(dim)
        
        c_cma = 1 + 2 / ((dim + 1) ** 2 - 2)
        mu = pop_size // 2  # number of recombination weights
        # Weights: linearly decreasing, normalized
        w = np.arange(1, mu + 1, dtype=float)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Recompute rates with correct mu_eff
        c1 = (1 + 2 / ((dim + 1) ** 2 - 2)) / (dim + 1) ** 2
        cmu = min(c1, (2 * (2 - 1 / (dim + 1)) / (2 + dim + 2 / (dim + 1))) / (dim + 1) ** 2)
        cs = (mu_eff + 1.5 + cmu) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        
        max_cma_iter = 60
        no_improve_cma = 0
        best_f_cma = start_f
        
        for it in range(max_cma_iter):
            if calls_made >= budget - 20 or time.time() > deadline - 5:
                break
            
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(pop_size):
                if calls_made >= budget - 20 or time.time() > deadline - 5:
                    break
                z = rng.standard_normal(dim)
                # Use Cholesky of C for sampling
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    C_sqrt = np.eye(dim)
                y = C_sqrt @ z
                x_new = mean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                f_new = call_func(x_new)
                offspring.append(x_new)
                offspring_f.append(f_new)
            
            if len(offspring) < 2:
                break
            
            # Sort by fitness
            indices = np.argsort(offspring_f)
            offspring = [offspring[i] for i in indices]
            offspring_f = [offspring_f[i] for i in indices]
            
            # Update best
            if offspring_f[0] < best_f_cma - 1e-15:
                no_improve_cma = 0
                best_f_cma = offspring_f[0]
            else:
                no_improve_cma += 1
            
            # Update mean using weighted recombination
            y_mean = np.zeros(dim)
            for i in range(mu):
                y_mean += w[i] * (offspring[i] - mean) / sigma
            mean = mean + sigma * y_mean
            mean = np.clip(mean, -5, 5)
            
            # Update step size (CSA)
            y_sigma = (np.linalg.norm(C_inv_sqrt @ y_mean) / chi_n - 1)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * y_sigma * C_inv_sqrt @ y_mean
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) <
                       (1.4 + 2 / (dim + 1)) * chi_n)
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)
            
            # Update evolution path
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * y_mean
            c_cma = 1 + 2 / ((dim + 1) ** 2 - 2)
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * cc * (2 - cc)
            outer = np.outer(p_c, p_c)
            for i in range(mu):
                y_i = (offspring[i] - mean) / sigma
                C = (1 + c_cma * c1 * delta_h - c1 - cmu) * C + c1 * outer + cmu * sum(w[j] * np.outer(y_i, y_i) for j in [i])
            
            # Regularize C
            C = 0.5 * (C + C.T)
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Adaptive restart: if stuck, try a new random point
            if no_improve_cma >= 15 and calls_made < budget - 30:
                # Try a point near current best
                perturbation = max(0.3, sigma * 0.5)
                x_new = mean + rng.normal(0, perturbation, dim)
                x_new = np.clip(x_new, -4.5, 4.5)
                f_new = call_func(x_new)
                if f_new < best_f_cma:
                    mean = x_new.copy()
                    sigma = max(0.3, sigma * 0.7)
                    C = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    no_improve_cma = 0
                    best_f_cma = f_new
                else:
                    no_improve_cma = 8
        
        # Phase 3: Final Nelder-Mead polish with tight tolerance
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline - 3:
            # Initialize simplex around current best
            simplex = [mean.copy()]
            simplex_f = [best_f_cma]
            
            polish_step = max(0.1, sigma * 0.3)
            for i in range(dim):
                if calls_made >= budget - 5 or time.time() > deadline - 3:
                    break
                v = mean.copy()
                v[i] += polish_step
                fv = call_func(v)
                simplex.append(v.copy())
                simplex_f.append(fv)
            
            alpha = 1.0
            gamma = 2.0
            sigma_nm = 0.5
            polish_tol = 1e-14
            
            polish_iters = min(remaining // 2, 50)
            for _ in range(polish_iters):
                if calls_made >= budget or time.time() > deadline - 3:
                    break
                
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                if abs(simplex_f[-1] - simplex_f[0]) < polish_tol:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = call_func(x_ref)
                
                if f_ref < simplex_f[0]:
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    f_exp = call_func(x_exp)
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
                    if f_ref < simplex_f[-1]:
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        x_con = centroid - 0.5 * (centroid - simplex[-1])
                    f_con = call_func(x_con)
                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = call_func(simplex[i])
        
        return best_x, best_f
