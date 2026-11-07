"""
Hybrid CMA-ES and Nelder-Mead Optimizer for BBOB f20 (Schwefel).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0
        
        call_count = [0]
        
        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            try:
                return float(func(x))
            except Exception:
                return None

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: CMA-ES Global Search
        # Population size lambda
        lam = max(4, int(4 + 3 * np.log(dim))) 
        # Initial sigma
        sigma = 1.0
        # Initial mean: center of the box
        m = np.zeros(dim)
        
        # CMA-ES parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute square root of C
        def update_sqrt_C():
            nonlocal C
            # Use Cholesky with fallback to SVD if needed
            try:
                Csqrt = np.linalg.cholesky(C)
                return Csqrt
            except np.linalg.LinAlgError:
                # Symmetrize and try again or use SVD
                C = 0.5 * (C + C.T)
                try:
                    Csqrt = np.linalg.cholesky(C)
                    return Csqrt
                except np.linalg.LinAlgError:
                    U, s, Vt = np.linalg.svd(C)
                    Csqrt = (U * np.sqrt(s)) @ Vt
                    return Csqrt

        Csqrt = update_sqrt_C()
        
        # Generate initial population
        def generate_population():
            z = np.random.randn(lam, dim)
            return m + sigma * (z @ Csqrt.T)
        
        # CMA-ES Loop
        while call_count[0] < budget // 2:
            if time.time() > deadline - 60:
                break
                
            # Generate and evaluate population
            pop = generate_population()
            fitnesses = np.zeros(lam)
            
            for i in range(lam):
                if call_count[0] >= budget // 2:
                    break
                if time.time() > deadline - 60:
                    break
                f = make_call(pop[i])
                if f is None:
                    f = np.inf
                fitnesses[i] = f
                update_best(pop[i], f)
            
            if call_count[0] >= budget // 2:
                break
                
            # Sort by fitness
            idx = np.argsort(fitnesses)
            sorted_pop = pop[idx]
            sorted_fit = fitnesses[idx]
            
            # Update mean
            m_new = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)
            
            # Update evolution paths
            # Need inverse sqrt of C
            Cinv_sqrt = np.linalg.inv(Csqrt)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * Cinv_sqrt @ (m_new - m)
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (budget - call_count[0])))) / np.sqrt(mu_eff) < 1.4 + 2 / (dim + 1)
            
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m)
            
            # Update covariance matrix
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = Cinv_sqrt @ (sorted_pop[i] - m)
                rank_mu += weights[i] * np.outer(y_i, y_i)
                
            C = (1 + c_1 * (hsig * (2 - c_c) - c_c) + c_mu) * C + c_1 * (rank_one - (1 - c_c) * C) + c_mu * rank_mu
            
            # Symmetrize
            C = 0.5 * (C + C.T)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 0.01, 10.0)
            
            m = m_new
            Csqrt = update_sqrt_C()
            
            # Check for stagnation or convergence
            if np.linalg.norm(m_new - m) < 1e-8 * sigma:
                break

        # Phase 2: Nelder-Mead Local Refinement
        if best_x is not None and call_count[0] < budget - 10 and time.time() < deadline - 10:
            x = best_x.copy()
            fx = best_f
            
            # Initialize simplex
            simplex = []
            simplex.append(x.copy())
            simplex_f = [fx]
            
            # Initial step size for simplex vertices
            step = 0.1
            for i in range(dim):
                v1 = x.copy()
                v1[i] += step
                v1 = np.clip(v1, -5.0, 5.0)
                f1 = make_call(v1)
                if f1 is None:
                    f1 = np.inf
                simplex.append(v1)
                simplex_f.append(f1)
                update_best(v1, f1)
                
                v2 = x.copy()
                v2[i] -= step
                v2 = np.clip(v2, -5.0, 5.0)
                f2 = make_call(v2)
                if f2 is None:
                    f2 = np.inf
                simplex.append(v2)
                simplex_f.append(f2)
                update_best(v2, f2)
                
                if call_count[0] >= budget - 5:
                    break
            
            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            
            while call_count[0] < budget - 5:
                if time.time() > deadline - 5:
                    break
                    
                # Sort simplex
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                # Check convergence
                if simplex_f[0] - simplex_f[-1] < 1e-8:
                    break
                    
                # Centroid of best points (excluding worst)
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = make_call(x_r)
                if f_r is None:
                    f_r = np.inf
                update_best(x_r, f_r)
                
                if f_r < simplex_f[-2]:
                    # Expansion
                    if f_r < simplex_f[0]:
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5.0, 5.0)
                        f_e = make_call(x_e)
                        if f_e is None:
                            f_e = np.inf
                        update_best(x_e, f_e)
                        
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        f_c = make_call(x_c)
                        if f_c is None:
                            f_c = np.inf
                        update_best(x_c, f_c)
                        
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Inside contraction
                            x_ic = centroid - rho * (centroid - simplex[-1])
                            x_ic = np.clip(x_ic, -5.0, 5.0)
                            f_ic = make_call(x_ic)
                            if f_ic is None:
                                f_ic = np.inf
                            update_best(x_ic, f_ic)
                            
                            if f_ic < simplex_f[-1]:
                                simplex[-1] = x_ic
                                simplex_f[-1] = f_ic
                            else:
                                # Shrink
                                for i in range(1, len(simplex)):
                                    simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                    f_i = make_call(simplex[i])
                                    if f_i is None:
                                        f_i = np.inf
                                    simplex_f[i] = f_i
                                    update_best(simplex[i], f_i)
                    else:
                        # Inside contraction
                        x_ic = centroid - rho * (centroid - simplex[-1])
                        x_ic = np.clip(x_ic, -5.0, 5.0)
                        f_ic = make_call(x_ic)
                        if f_ic is None:
                            f_ic = np.inf
                        update_best(x_ic, f_ic)
                        
                        if f_ic < simplex_f[-1]:
                            simplex[-1] = x_ic
                            simplex_f[-1] = f_ic
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                f_i = make_call(simplex[i])
                                if f_i is None:
                                    f_i = np.inf
                                simplex_f[i] = f_i
                                update_best(simplex[i], f_i)

        return best_x, best_f
# EVOLVE-BLOCK-END
