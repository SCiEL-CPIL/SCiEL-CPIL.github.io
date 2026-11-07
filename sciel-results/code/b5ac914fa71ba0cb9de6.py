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
        deadline = start_time + 295  # Safety margin before 330s kill

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_left():
            return deadline - time.time()

        # --- Phase 1: Coarse Global Search (Random + Hill Climbing) ---
        # Use a portion of the budget for diverse initial points and quick local improvement.
        # This helps handle multimodality of Gallagher 21 peaks.
        n_random = 20
        n_hill = 10
        hill_steps = 10
        hill_step_size = 0.5

        # Generate random points
        rng = np.random.default_rng(42)
        for _ in range(n_random):
            if calls >= budget or time_left() < 5:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # For the best few points, do a simple random walk hill climbing
        # We don't store all points, but we can re-sample or keep a small population.
        # Let's keep a small population of top candidates.
        pop = []
        # Re-evaluate? No, we just used random. Let's keep track of top points.
        # Actually, the above loop didn't store points. Let's restart this phase properly.
        
        # Reset for a cleaner structure
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0
        
        # Phase 1: Random Sampling with Local Refinement
        pop = []
        for i in range(15):
            if calls >= budget or time_left() < 10:
                break
            x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            pop.append((f, x))
        
        # Sort and keep top 5
        pop.sort(key=lambda t: t[0])
        pop = pop[:5]
        
        # Local refinement on top 5
        for f, x in pop:
            for _ in range(5):  # 5 restarts of local search
                if calls >= budget or time_left() < 10:
                    break
                cx = x.copy()
                cf = f
                for _ in range(20):  # 20 steps per local search
                    if calls >= budget or time_left() < 10:
                        break
                    # Random step
                    step = rng.normal(0, 0.5, dim)
                    nx = cx + step
                    nf = evaluate(nx)
                    if nf < cf:
                        cx = nx
                        cf = nf
        
        # Phase 2: CMA-ES from best point
        if calls < budget and time_left() > 5:
            x0 = best_x.copy()
            sigma = 1.0
            lam = 8
            mu = lam // 2
            
            m = x0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Parameters
            cs = (lam + 2) / (dim + lam + 5)
            ds = 1 + 2 * max(0, np.sqrt((lam - 2) / (dim + 2)) - 1) + cs
            cm = 1.0 / (dim * dim) * (4 - (lam + 2) / (dim + lam + 5) + 2 / (dim + 2 - 1))
            
            # Weights
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            chi_N = np.sqrt(dim) * (1 - 1.0/(4.0*dim) + 1.0/(21.0*dim*dim))
            
            L = np.linalg.cholesky(C)
            
            while calls + lam <= budget and time_left() > 5:
                # Generate population
                points = []
                fvals = []
                for _ in range(lam):
                    z = rng.standard_normal(dim)
                    y = m + sigma * (L @ z)
                    points.append(y)
                
                for p in points:
                    f = evaluate(p)
                    fvals.append(f)
                
                # Sort
                idx = np.argsort(fvals)
                points = [points[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Select top mu
                y_vec = np.zeros(dim)
                for i in range(mu):
                    y_vec += weights[i] * (points[i] - m)
                
                # Update paths
                y_sigma = y_vec / sigma
                term = np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ y_sigma)
                p_sigma = (1 - cs) * p_sigma + term
                
                threshold = (1.4 + 2.0 / (dim + 1)) * (1 - cs)
                if np.linalg.norm(p_sigma) / sigma > threshold:
                    hc = 1.0
                else:
                    hc = 0.0
                
                p_c = (1 - cs) * p_c + hc * np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ y_sigma)
                
                # Update sigma
                norm_p_sigma = np.linalg.norm(p_sigma)
                sigma = sigma * np.exp(cs / ds * (norm_p_sigma / sigma - chi_N))
                
                # Update C
                c_m_update = cm * (1 + 2 * cm / mu_eff)
                rank1 = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i_sigma = (points[i] - m) / sigma
                    rank_mu += weights[i] * np.outer(y_i_sigma, y_i_sigma)
                
                C = (1 - c_m_update) * C + cm * (hc * rank1 + rank_mu)
                
                # Update mean
                m = m + y_vec
                
                # Update Cholesky
                C = (C + C.T) / 2.0
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-6 * np.eye(dim)
                    L = np.linalg.cholesky(C)
                
                if sigma < 1e-10:
                    break

        # Phase 3: Nelder-Mead Polishing on best point
        if calls < budget and time_left() > 5:
            # Initialize simplex
            simplex = [best_x.copy()]
            for i in range(dim):
                x_temp = best_x.copy()
                x_temp[i] += 0.1 if x_temp[i] < 4.9 else -0.1
                simplex.append(x_temp)
            
            fvals = [evaluate(x) for x in simplex]
            
            # Nelder-Mead
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            
            max_iter = 100
            for _ in range(max_iter):
                if calls >= budget or time_left() < 5:
                    break
                
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Centroid
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = evaluate(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = evaluate(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            fvals[i] = evaluate(simplex[i])
            
            # Update best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_f:
                best_f = fvals[idx_best]
                best_x = simplex[idx_best].copy()

        return best_x, best_f
