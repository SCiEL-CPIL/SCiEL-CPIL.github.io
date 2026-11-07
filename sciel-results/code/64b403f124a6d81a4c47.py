"""Optimizer for BBOB-style Sphere f1 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        calls = 0

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Newton-style direct minimizer estimation
        # Sphere is f(x) = (x-x*)^T A (x-x*) + f_opt
        # Gradient at 0 is -2 A x*. 
        # If we assume A ~ I, step is x* approx.
        
        center = np.zeros(dim)
        fc = func(center)
        calls += 1
        if fc < best_f:
            best_f = fc
            best_x = center.copy()
        
        # Estimate gradient at center using finite differences
        h = 0.1
        grad = np.zeros(dim)
        for i in range(dim):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            xp = center.copy()
            xp[i] += h
            xp = clip(xp)
            fplus = func(xp)
            calls += 1
            if fplus < best_f:
                best_f = fplus
                best_x = xp.copy()
            
            xm = center.copy()
            xm[i] -= h
            xm = clip(xm)
            fminus = func(xm)
            calls += 1
            if fminus < best_f:
                best_f = fminus
                best_x = xm.copy()
            
            grad[i] = (fplus - fminus) / (2 * h)
        
        # Take a Newton-like step
        x = center.copy()
        fx = fc
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6 and calls < budget:
            # For sphere, if A=I, grad = 2(x-x*). Step = -grad/2 = x* - x.
            # So x_new = x - 0.5 * grad
            # However, if A is not I, this is an approximation.
            # We use a damped step to be safe.
            step_size = 0.5 
            x_new = clip(x - step_size * grad)
            fx_new = func(x_new)
            calls += 1
            if fx_new < fx:
                x, fx = x_new, fx_new
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # If the step didn't help, try a smaller one or just keep x
                # Try half step
                if calls < budget:
                    x_new2 = clip(x - 0.25 * grad)
                    fx_new2 = func(x_new2)
                    calls += 1
                    if fx_new2 < fx:
                        x, fx = x_new2, fx_new2
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        # Phase 2: CMA-ES with adaptive step size
        # More robust than Nelder-Mead for quadratic functions
        
        x_mean = best_x.copy()
        f_best = best_f
        sigma = 0.3  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        pc = np.zeros(dim)  # Evolution path for C
        ps = np.zeros(dim)  # Evolution path for sigma
        
        # CMA-ES parameters
        pop_size = 4  # lambda
        cc = (4 + dim/3) / (dim + 4 + 2)  # Correlation for C
        cs = (4 + dim/3) / (dim + 4 + 3*cc)  # Correlation for sigma
        c1 = 2 / (dim**1.7 + 10)  # Learning rate for C
        cmu = min(1 - c1, 2 / (dim**1.8 + 10))  # Learning rate for C from mutation
        damps = 1 + 2*max(0, sqrt((dim-2)/(dim+2)) - 1) + cs
        
        chi_n = sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        max_iter = min(budget - calls, 200)
        no_improve = 0
        
        for it in range(max_iter):
            if calls >= budget or time.time() > deadline:
                break
            
            # Generate population
            w = np.zeros(pop_size)
            rho = int(np.floor(pop_size / 2))
            w[:rho] = np.log(pop_size/2) - np.log(np.arange(1, rho+1))
            w /= w.sum()
            mu_eff = 1 / (w[:rho] ** 2).sum()
            
            B, D = np.linalg.eigh(C)
            B = B * 1.0 / D
            D = np.sqrt(D)
            
            # Sample and evaluate
            offspring = []
            offspring_f = []
            for k in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                z = np.random.default_rng(42 + it*pop_size + k).standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = clip(y)
                fy = func(y)
                calls += 1
                offspring.append(y)
                offspring_f.append(fy)
                if fy < f_best:
                    f_best = fy
                    best_x = y.copy()
                    best_f = fy
                    no_improve = 0
                else:
                    no_improve += 1
            
            if len(offspring) == 0:
                break
            
            # Sort by fitness
            order = np.argsort(offspring_f)
            offspring = [offspring[i] for i in order]
            offspring_f = [offspring_f[i] for i in order]
            
            # Update mean
            x_new_mean = sum(w[k] * offspring[k] for k in range(rho))
            
            # Update evolution paths
            ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * (B.T @ (D * (x_new_mean - x_mean))) / sigma
            hsig = 1 if np.linalg.norm(ps) / sqrt(1 - (1-cs)**(2*(it+1))) < (1.4 + 2/(dim+1)) * chi_n else 0
            pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            # Update covariance
            delta_hs = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * (mu_eff - 2 + delta_hs) - c1) * C
            for k in range(rho):
                yk = (x_new_mean - x_mean) / sigma
                C += c1 * w[k] * np.outer(yk, yk)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            if np.min(eigenvalues) < 1e-12:
                C = np.eye(dim)
            
            x_mean = x_new_mean
            
            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(1e-10, min(sigma, 5.0))
            
            # Convergence check
            if sigma < 1e-8 or no_improve > pop_size * 5:
                break
        
        # If CMA-ES didn't improve much, fall back to Nelder-Mead
        if f_best >= best_f - 1e-10:
            simplex = []
            simplex.append(best_x.copy())
            simplex_f = [best_f]
            
            delta = 0.3
            for i in range(dim):
                if calls >= budget:
                    break
                v = best_x.copy()
                v[i] += delta
                v = clip(v)
                simplex.append(v)
                fv = func(v)
                calls += 1
                simplex_f.append(fv)
                if fv < best_f:
                    best_f = fv
                    best_x = v.copy()

        # Nelder-Mead fallback (only if CMA-ES didn't help)
        if len(simplex) > 1:
            alpha = 1.0
            gamma = 2.0
            rho_c = 0.5
            sigma_s = 0.5
            tol = 1e-12
            max_iter = budget - calls

            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                if np.std(simplex_f) < tol:
                    break
                    
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = clip(x_r)
                f_r = func(x_r)
                calls += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                
                if f_r < simplex_f[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = clip(x_e)
                    f_e = func(x_e)
                    calls += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    if f_r < f_worst:
                        x_c = centroid + rho_c * (x_r - centroid)
                    else:
                        x_c = centroid + rho_c * (x_worst - centroid)
                    x_c = clip(x_c)
                    f_c = func(x_c)
                    calls += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = centroid + sigma_s * (simplex[i] - centroid)
                            simplex[i] = clip(simplex[i])
                            if calls >= budget:
                                break
                            f_s = func(simplex[i])
                            calls += 1
                            simplex_f[i] = f_s
                            if f_s < best_f:
                                best_f = f_s
                                best_x = simplex[i].copy()
                        if calls >= budget:
                            break

        # Phase 3: Final high-precision polish with adaptive (1+1)-ES
        if calls < budget and time.time() < deadline:
            x_local = best_x.copy()
            fx_local = best_f
            sigma = 0.05  # Start with larger step for better exploration
            rng = np.random.default_rng(123)
            no_improve_count = 0
            max_no_improve = 50
            
            while calls < budget and time.time() < deadline:
                y = clip(x_local + sigma * rng.standard_normal(dim))
                fy = func(y)
                calls += 1
                
                if fy < fx_local:
                    x_local, fx_local = y, fy
                    sigma = min(sigma * 1.05, 0.5)
                    no_improve_count = 0
                    if fx_local < best_f:
                        best_f = fx_local
                        best_x = x_local.copy()
                else:
                    sigma *= 0.95
                    no_improve_count += 1
                
                if sigma < 1e-12 or no_improve_count > max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
