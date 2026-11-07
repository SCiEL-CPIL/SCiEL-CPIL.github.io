"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Quadratic-aware optimizer for Sphere: finite-difference Newton + CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        t_start = time.time()
        deadline = 285.0  # safety margin below the 300s wall-clock limit

        calls = [0]

        def f(x):
            if calls[0] >= budget:
                return best_fx
            calls[0] += 1
            return func(x)

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx

        # Start at center (minimizer is in [-4,4]^5)
        x0 = np.zeros(dim)
        fx0 = f(x0)
        best_x = x0.copy()
        best_fx = fx0

        # ---- Phase 1: Estimate gradient and Hessian via finite differences ----
        # For a quadratic f(x) = 0.5*(x-c)^T A (x-c) + f0, we can estimate
        # the minimizer c and matrix A from function evaluations.

        # Step 1: Estimate gradient at x0 using central differences with step h
        h = 0.01
        grad = np.zeros(dim)
        for i in range(dim):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += h
            x_minus[i] -= h
            x_plus = np.clip(x_plus, -5.0, 5.0)
            x_minus = np.clip(x_minus, -5.0, 5.0)
            fp = f(x_plus)
            fm = f(x_minus)
            grad[i] = (fp - fm) / (2 * h)
            update_best(x_plus, fp)
            update_best(x_minus, fm)
            if calls[0] >= budget:
                break

        # If we've used up budget, return what we have
        if calls[0] >= budget:
            return best_x, best_fx

        # Step 2: Estimate Hessian using central second differences
        # For quadratic, Hessian is constant, so one estimate suffices
        hess = np.zeros((dim, dim))
        h2 = 0.1  # larger step for second derivatives
        for i in range(dim):
            for j in range(i, dim):
                if calls[0] >= budget:
                    break
                xpp = x0.copy()
                xpp[i] += h2
                xpp[j] += h2
                xpp = np.clip(xpp, -5.0, 5.0)

                xmm = x0.copy()
                xmm[i] -= h2
                xmm[j] -= h2
                xmm = np.clip(xmm, -5.0, 5.0)

                xpm = x0.copy()
                xpm[i] += h2
                xpm[j] -= h2
                xpm = np.clip(xpm, -5.0, 5.0)

                xmp = x0.copy()
                xmp[i] -= h2
                xmp[j] += h2
                xmp = np.clip(xmp, -5.0, 5.0)

                if calls[0] >= budget:
                    break
                fpp = f(xpp)
                fmm = f(xmm)
                fpm = f(xpm)
                fmp = f(xmp)

                hess_val = (fpp + fmm - fpm - fmp) / (4 * h2 * h2)
                hess[i, j] = hess_val
                hess[j, i] = hess_val

                # Track best from these evaluations
                update_best(xpp, fpp)
                update_best(xmm, fmm)
                update_best(xpm, fpm)
                update_best(xmp, fmp)

        if calls[0] >= budget:
            return best_x, best_fx

        # Step 3: Solve for minimizer: x* = x0 - H^{-1} * grad
        # For quadratic: grad(x) = H(x - c), so c = x0 - H^{-1} grad(x0)
        try:
            # Regularize Hessian for numerical stability
            eigvals, eigvecs = np.linalg.eigh(hess)
            # Clip eigenvalues to be positive (Sphere should have positive definite Hessian)
            eigvals_clipped = np.maximum(eigvals, 1e-6)
            hess_inv = eigvecs @ np.diag(1.0 / eigvals_clipped) @ eigvecs.T
            x_star = x0 - hess_inv @ grad
            x_star = np.clip(x_star, -5.0, 5.0)

            fx_star = f(x_star)
            update_best(x_star, fx_star)
        except np.linalg.LinAlgError:
            pass

        if calls[0] >= budget:
            return best_x, best_fx

        # ---- Phase 2: Multi-start CMA-ES with progressively smaller step sizes ----
        # Use deterministic multi-start strategy with CMA-ES
        # Start with the Newton estimate as initial center
        
        def cma_es(center, sigma0, max_iter, pop_size):
            nonlocal best_x, best_fx, calls
            
            if calls[0] >= budget:
                return
            
            # CMA-ES parameters
            n = dim
            # Population size
            lam = max(pop_size, 4 + int(3 * np.log(n)))
            
            # Initial population
            mean = center.copy()
            sigma = sigma0
            
            # Covariance matrix
            C = np.eye(n)
            # Step size adaptation
            cs = 1 / (np.sqrt(n) + 2)
            ds = 1 + 2 * max(0, (n + 1) / 3 - 1)
            c1 = 2 / ((n + 1.3)**2 + 5)
            cmu = min(1 - c1, 2 * ((n - 2 + 1/n) / (n + 2)**2 - 1))
            
            # Evolution paths
            ps = np.zeros(n)
            ph = np.zeros(n)
            
            # Eigen decomposition cache
            D_inv = np.ones(n)
            B = np.eye(n)
            D = np.ones(n)
            
            # Initial eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs
            D_inv = 1.0 / D
            
            # Generate initial population
            def generate_population():
                nonlocal B, D_inv
                pop = []
                for k in range(lam):
                    z = np.random.standard_normal(n)
                    y = B @ (D_inv * z)
                    x_new = mean + sigma * y
                    x_new = np.clip(x_new, -5.0, 5.0)
                    pop.append(x_new)
                return pop
            
            # Evaluate initial population
            pop = generate_population()
            fvals = []
            for x in pop:
                if calls[0] >= budget:
                    break
                fv = f(x)
                fvals.append(fv)
                update_best(x, fv)
            
            if len(fvals) < lam:
                return
            
            # Main CMA-ES loop
            for iteration in range(max_iter):
                if calls[0] >= budget or (time.time() - t_start) > deadline:
                    break
                
                # Sort by fitness
                order = np.argsort(fvals)
                pop_sorted = [pop[i] for i in order]
                fvals_sorted = [fvals[i] for i in order]
                
                # Update mean (weighted)
                weights = np.zeros(lam)
                mu = lam // 2
                # Logarithmic weights
                w = np.log(np.arange(1, mu + 1))
                w = w / w.sum()
                weights[:mu] = w
                weights[mu] = 1 - weights[:mu].sum()  # Ensure sum = 1
                weights = np.maximum(weights, 0)
                weights = weights / weights.sum()
                
                mean_new = np.sum([weights[i] * pop_sorted[i] for i in range(lam)], axis=0)
                
                # Check if we improved
                if fvals_sorted[0] < best_fx:
                    update_best(pop_sorted[0], fvals_sorted[0])
                
                # Update evolution paths
                # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mu_eff) * C^(-1/2) * (mean_new - mean)/sigma
                mu_eff = 1.0 / np.sum(weights[:mu]**2) if mu > 0 else 1.0
                C_inv_sqrt = B @ (D_inv * (B.T @ (mean_new - mean))) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt
                
                # Check for termination based on step size
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (iteration + 1))) > 1.4 + 4 / n:
                    break
                
                # Update ph
                ph = (1 - ds) * ph + np.sqrt(ds * (2 - ds) * mu_eff) * (mean_new - mean) / (sigma * np.linalg.norm(C_inv_sqrt) + 1e-20)
                
                # Update covariance
                # C = (1-c1-cmu)*C + c1*(ps*ps.T + (1-cs)*C) + cmu*sum(wi*(yi*yi.T))
                y_vectors = []
                for i in range(mu):
                    yi = (pop_sorted[i] - mean) / (sigma + 1e-20)
                    y_vectors.append(yi)
                
                delta_h = (1 + (1 - ds) * 2 * (n + 1) / 3) * np.sqrt((1 - (1 - ds)**2) / (1 - (1 - ds)**(2 * (iteration + 1))))
                c1_eff = c1
                cmu_eff = cmu if np.linalg.norm(ph) / np.sqrt(1 - (1 - ds)**(2 * (iteration + 1))) < (1.4 + 2 / (n + 1)) * delta_h else 0
                
                C = (1 - c1_eff - cmu_eff) * C + c1_eff * (np.outer(ps, ps) + (1 - cs) * C)
                if cmu_eff > 0:
                    for i in range(mu):
                        C += cmu_eff * weights[i] * np.outer(y_vectors[i], y_vectors[i])
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                eigvals_C, eigvecs_C = np.linalg.eigh(C)
                eigvals_C = np.maximum(eigvals_C, 1e-20)
                C = eigvecs_C @ np.diag(eigvals_C) @ eigvecs_C.T
                
                # Update eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                B = eigvecs
                D_inv = 1.0 / D
                
                # Update sigma (step size adaptation)
                sigma = sigma * np.exp(cs / (5 * (n + 2)) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (iteration + 1))) - np.sqrt(n * (1 - 1/(n+1)))))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Update mean
                mean = mean_new
                
                # Generate next population
                pop = []
                fvals = []
                for k in range(lam):
                    if calls[0] >= budget:
                        break
                    z = np.random.standard_normal(n)
                    y = B @ (D_inv * z)
                    x_new = mean + sigma * y
                    x_new = np.clip(x_new, -5.0, 5.0)
                    pop.append(x_new)
                    fv = f(x_new)
                    fvals.append(fv)
                    update_best(x_new, fv)
                
                if len(fvals) < lam:
                    break
                
                # Stagnation check: if no improvement in last 20 iterations, break
                if iteration > 20 and fvals_sorted[0] >= fvals[0] * (1 + 1e-10):
                    # Check if we've been stuck
                    pass  # Continue for now, let budget limit handle it

        # Multi-start CMA-ES with progressively smaller step sizes
        # Start with larger step size, then refine
        initial_center = best_x.copy()
        
        # First CMA-ES run with moderate step size
        cma_es(initial_center, sigma0=1.0, max_iter=50, pop_size=10)
        
        if calls[0] >= budget:
            return best_x, best_fx
        
        # Second CMA-ES run with smaller step size for refinement
        cma_es(best_x, sigma0=0.1, max_iter=50, pop_size=8)
        
        if calls[0] >= budget:
            return best_x, best_fx
        
        # Third CMA-ES run with very small step size for fine-tuning
        cma_es(best_x, sigma0=0.01, max_iter=30, pop_size=6)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
