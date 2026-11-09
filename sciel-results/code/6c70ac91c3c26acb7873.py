"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

class Optimizer:
    """
    Optimized CMA-ES for BBOB f18 (Schaffers F7 ill-conditioned).
    - Stratified random initialization for better coverage.
    - PCA-based covariance initialization to align with funnel axes.
    - Robust CMA-ES with restarts on stagnation or small sigma.
    - Nelder-Mead polishing phase.
    - Strict time and budget guards.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility in this context, 
        # but random is also fine. Fixed seed ensures consistent behavior 
        # if the evaluator runs multiple times, though the problem instance 
        # has a hidden shift/rotation.
        rng = np.random.default_rng(42)

        # Phase 1: Stratified Initialization
        # We want to cover the space [-5, 5]^dim.
        # Use Latin Hypercube Sampling (LHS) style.
        n_init = min(100, budget // 5) # Use up to 100 calls for init
        if n_init < 10:
            n_init = 10
            
        # Generate LHS points
        # Split [0,1] into n_init intervals, pick one random point in each, map to [-5,5]
        u = np.zeros((n_init, dim))
        for d in range(dim):
            # Random permutation of intervals
            perm = rng.permutation(n_init)
            # Random points within intervals
            r = rng.uniform(0, 1, n_init)
            u[:, d] = (perm + r) / n_init
        x_init = u * 10.0 - 5.0
        
        best_x = None
        best_fx = float('inf')
        samples = []
        calls_used = 0

        for i in range(n_init):
            if time.time() > deadline - 10.0: break
            x = x_init[i]
            fx = func(x)
            calls_used += 1
            samples.append((fx, x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback if no calls made (should not happen)
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls_used += 1

        # Initialize CMA-ES parameters
        # Population size lambda
        lam = max(4, int(4 + 3 * np.log(dim)))
        # For dim=5, lam=9. Let's use 10 for stability.
        lam = max(lam, 10)
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        # Adaptation rates
        c_sigma = (lam + 2) / (dim + 4)
        c1 = 2 / ((dim + 1.3)**2 + lam)
        cmu = min(0.5, c1 * (lam // 2 - 1) / 2)
        d_sigma = 1 + 2 * max(0, (dim / lam - 1) / 3) + cmu
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Initialize state from best sample and PCA of top samples
        samples.sort(key=lambda t: t[0])
        n_pca = max(5, min(20, len(samples) // 5))
        top_points = np.array([s[1] for s in samples[:n_pca]])
        
        m = top_points.mean(axis=0)
        centered = top_points - m
        C = np.cov(centered, rowvar=False)
        
        # Ensure C is positive definite and scaled
        # Scale C so that the average step size is reasonable (e.g., 1.0)
        # Trace(C) approx dim * sigma^2. We want sigma ~ 1.
        # So we want trace(C) ~ dim.
        trace_C = np.trace(C)
        if trace_C > 1e-6:
            C = C * (dim / trace_C)
        else:
            C = np.eye(dim)
            
        # Add small regularization
        C += 1e-4 * np.eye(dim)
        
        # Cholesky
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            
        sigma = 1.0 # Initial step size
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Stagnation tracking
        stall_count = 0
        max_stall = 15
        consecutive_restarts = 0
        max_consecutive_restarts = 3
        restart_sigma_mult = 1.5

        while calls_used < budget:
            if time.time() > deadline - 5.0:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = m + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            evaluated = []
            for y in offspring:
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                fy = func(y)
                calls_used += 1
                evaluated.append((fy, y))

            if len(evaluated) < lam:
                break

            # Sort by fitness
            evaluated.sort(key=lambda t: t[0])
            
            # Update best
            if evaluated[0][0] < best_fx:
                best_fx = evaluated[0][0]
                best_x = evaluated[0][1].copy()
                stall_count = 0
            else:
                stall_count += 1

            # Selection
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * evaluated[i][1]

            # Calculate weighted mean in normalized space
            z_w = np.zeros(dim)
            for i in range(mu):
                # y_i = m + sigma * C_sqrt @ z_i  =>  z_i = C_sqrt^{-1} (y_i - m) / sigma
                # Solve C_sqrt z_i = (y_i - m) / sigma
                try:
                    z_i = np.linalg.solve(C_sqrt, (evaluated[i][1] - m) / sigma)
                except np.linalg.LinAlgError:
                    z_i = (evaluated[i][1] - m) / sigma
                z_w += weights[i] * z_i

            # Update p_sigma
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(dim+1)) * chi_n) else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_w

            # Update sigma
            if h_sigma:
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            sigma = float(np.clip(sigma, 1e-6, 10.0))

            # Update p_c
            h_c = 1.0 if (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c1)**2) < (1.4 + 2/(dim+1)) * np.sqrt((c1 / (2 - c1)) - 1)) else 0.0
            p_c = (1 - c1) * p_c + h_c * np.sqrt(c1 * (2 - c1) * mu_eff) * z_w

            # Update Covariance C
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(p_c, p_c) + (1 - h_c) * c1 * (1 - c1) * C)
            
            for i in range(mu):
                try:
                    z_i = np.linalg.solve(C_sqrt, (evaluated[i][1] - m) / sigma)
                except np.linalg.LinAlgError:
                    z_i = (evaluated[i][1] - m) / sigma
                C += cmu * weights[i] * np.outer(z_i, z_i)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update mean
            m = y_w
            m = np.clip(m, -5, 5)

            # Recompute C_sqrt
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(dim)

            # Restart logic
            # Restart if sigma is too small or stagnation is high
            if sigma < 1e-5 or stall_count >= max_stall:
                consecutive_restarts += 1
                # Reset covariance to isotropic
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                sigma = max(sigma * restart_sigma_mult, 0.5)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stall_count = 0
                max_stall = min(max_stall + 5, 30)
                restart_sigma_mult *= 1.2
                
                # If too many restarts, do a fresh random search
                if consecutive_restarts >= max_consecutive_restarts:
                    n_fresh = min(20, budget - calls_used)
                    if n_fresh > 0:
                        for _ in range(n_fresh):
                            if calls_used >= budget or time.time() > deadline - 5.0:
                                break
                            x = rng.uniform(-5, 5, dim)
                            fx = func(x)
                            calls_used += 1
                            if fx < best_fx:
                                best_fx = fx
                                best_x = x.copy()
                        m = best_x.copy()
                        sigma = 1.0
                        consecutive_restarts = 0
                        restart_sigma_mult = 1.5
                        max_stall = 15

        # Phase 2: Polish with Nelder-Mead
        if calls_used < budget - 10 and time.time() < deadline - 2.0:
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            max_iter_nm = min(100, (budget - calls_used) // dim)
            
            # Initialize simplex
            simplex = []
            simplex_fx = []
            
            x0 = best_x.copy()
            f0 = func(x0)
            calls_used += 1
            simplex.append(x0.copy())
            simplex_fx.append(f0)
            if f0 < best_fx:
                best_fx = f0
                best_x = x0.copy()
            
            step = 0.1
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_used += 1
                simplex.append(xi.copy())
                simplex_fx.append(fi)
                if fi < best_fx:
                    best_fx = fi
                    best_x = xi.copy()
            
            simplex = np.array(simplex)
            simplex_fx = np.array(simplex_fx)
            
            for _ in range(max_iter_nm):
                if calls_used >= budget or time.time() > deadline - 1.0:
                    break
                
                order = np.argsort(simplex_fx)
                simplex = simplex[order]
                simplex_fx = simplex_fx[order]
                
                if simplex_fx[0] < best_fx:
                    best_fx = simplex_fx[0]
                    best_x = simplex[0].copy()
                
                centroid = simplex[:-1].mean(axis=0)
                
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_used += 1
                if f_ref < best_fx:
                    best_fx = f_ref
                    best_x = x_ref.copy()
                
                if f_ref < simplex_fx[0]:
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_used += 1
                    if f_exp < best_fx:
                        best_fx = f_exp
                        best_x = x_exp.copy()
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_fx[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_fx[-1] = f_ref
                elif f_ref < simplex_fx[-2]:
                    simplex[-1] = x_ref
                    simplex_fx[-1] = f_ref
                else:
                    if f_ref < simplex_fx[-1]:
                        x_con = centroid + rho * (simplex[-1] - centroid)
                    else:
                        x_con = centroid - rho * (centroid - simplex[-1])
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_used += 1
                    if f_con < best_fx:
                        best_fx = f_con
                        best_x = x_con.copy()
                    if f_con < f_ref:
                        simplex[-1] = x_con
                        simplex_fx[-1] = f_con
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_i = func(simplex[i])
                            calls_used += 1
                            simplex_fx[i] = f_i
                            if f_i < best_fx:
                                best_fx = f_i
                                best_x = simplex[i].copy()
                        if calls_used >= budget or time.time() > deadline - 1.0:
                            break

        return best_x, best_fx
# EVOLVE-BLOCK-END
