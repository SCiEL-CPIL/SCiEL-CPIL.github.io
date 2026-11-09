import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Time guard: evaluator kills at 330s, leave 40s safety margin
        start_time = time.time()
        deadline = start_time + 290

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        evals_used = 0
        eval_log = []

        def try_func(x):
            nonlocal evals_used, best_f, best_x
            if evals_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x)
            if not np.isfinite(f_val):
                return np.inf
            evals_used += 1
            eval_log.append((x.copy(), f_val))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def cma_es(x_start, sigma_init, max_evals):
            """CMA-ES for the main search phase."""
            nonlocal evals_used, best_f, best_x, deadline
            
            if evals_used >= budget or time.time() > deadline:
                return
            
            # CMA-ES parameters for dim=5
            lam = 10  # population size
            mu = lam // 2
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            w_full = np.concatenate([w, np.zeros(lam - mu)])
            
            mueff = 1.0 / np.sum(w**2)
            
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = (2 - mueff / (dim + 2) + c_c) / ((dim + 1.3) ** 2 + mueff + c_c)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            
            x_mean = x_start.copy()
            sigma = sigma_init
            
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            def cholesky_safe(C_mat):
                try:
                    L = np.linalg.cholesky(C_mat + 1e-12 * np.eye(dim))
                    return L
                except:
                    for reg in [1e-10, 1e-8, 1e-6]:
                        try:
                            L = np.linalg.cholesky(C_mat + reg * np.eye(dim))
                            return L
                        except:
                            pass
                    return np.eye(dim)
            
            L = cholesky_safe(C)
            
            while evals_used < budget and time.time() < deadline:
                if budget - evals_used < lam:
                    break
                
                # Generate offspring
                ys = []
                xs = []
                fs = []
                
                for i in range(lam):
                    if evals_used >= budget or time.time() > deadline:
                        break
                    
                    z = rng.standard_normal(dim)
                    y = L @ z
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5.0, 5.0)
                    
                    f_val = try_func(x_new)
                    if np.isinf(f_val):
                        break
                    
                    ys.append(y)
                    xs.append(x_new)
                    fs.append(f_val)
                
                if len(xs) < lam:
                    break
                
                ys = np.array(ys)
                xs = np.array(xs)
                fs = np.array(fs)
                
                idx = np.argsort(fs)
                xs = xs[idx]
                ys = ys[idx]
                fs = fs[idx]
                
                x_mean_old = x_mean.copy()
                x_mean = np.sum(w_full[:mu, None] * xs[:mu], axis=0)
                
                # Update sigma
                y_bar = (x_mean - x_mean_old) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_bar
                norm_ps = np.linalg.norm(p_sigma)
                chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 50.0)
                
                # Update p_c and C
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * y_bar
                sum_w_y = np.zeros((dim, dim))
                for i in range(mu):
                    sum_w_y += w[i] * np.outer(ys[i], ys[i])
                
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
                C = (C + C.T) / 2
                C += 1e-12 * np.eye(dim)
                L = cholesky_safe(C)
                
                # Early stopping if sigma is very small or x_mean hasn't changed much
                if sigma < 1e-10 or np.linalg.norm(x_mean - x_mean_old) < 1e-10:
                    break

        def nelder_mead(x_start, max_iter=100, initial_step=0.5):
            """Nelder-Mead for final refinement."""
            nonlocal evals_used, best_f, best_x, deadline
            
            if time.time() > deadline or evals_used >= budget:
                return
            
            # Initialize simplex
            step = initial_step
            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += step
                simplex.append(xi)
            
            # Evaluate initial simplex
            fs = []
            for i in range(dim + 1):
                if evals_used >= budget or time.time() > deadline:
                    return
                f_val = try_func(simplex[i])
                fs.append(f_val)
            
            # Nelder-Mead parameters
            alpha_nm = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5
            
            for it in range(max_iter):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                # Sort simplex
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Check convergence
                if fs[-1] - fs[0] < 1e-10:
                    break
                
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                x_ref = centroid + alpha_nm * (centroid - simplex[-1])
                f_ref = try_func(x_ref)
                
                if f_ref < fs[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = try_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fs[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fs[-1] = f_ref
                elif f_ref < fs[-2]:
                    simplex[-1] = x_ref
                    fs[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    f_con = try_func(x_con)
                    if f_con < fs[-1]:
                        simplex[-1] = x_con
                        fs[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            fs[i] = try_func(simplex[i])
                        if evals_used >= budget or time.time() > deadline:
                            break
                
                # Update global best
                for i in range(dim + 1):
                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = simplex[i].copy()

        # Phase 1: Quick exploration to find a promising region
        # Evaluate center and a few random points
        try_func(np.zeros(dim))
        
        for i in range(15):
            if evals_used >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, size=dim)
            try_func(x)
        
        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: CMA-ES for main search (70% of remaining budget)
        # Start from the best point found so far
        x_start = best_x.copy()
        sigma_init = 2.0  # Moderate initial step size
        
        # Allocate ~70% of remaining budget to CMA-ES
        remaining = budget - evals_used
        cma_budget = int(remaining * 0.7)
        
        cma_es(x_start, sigma_init, cma_budget)
        
        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 3: Nelder-Mead for final refinement (30% of remaining budget)
        # Start from the current best
        x_start = best_x.copy()
        sigma_init = 0.5  # Smaller step for refinement
        
        remaining = budget - evals_used
        nm_max_iter = min(100, remaining // (dim + 1))
        
        nelder_mead(x_start, max_iter=nm_max_iter, initial_step=sigma_init)
        
        return best_x, best_f
