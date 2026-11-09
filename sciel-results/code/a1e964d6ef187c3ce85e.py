"""Adaptive Halton-Scan + Lightweight CMA-ES Optimizer."""
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
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin

        calls_used = 0
        best_x = None
        best_f = float('inf')

        def eval_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget or time.time() >= deadline:
                return float('inf')
            calls_used += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Halton Sequence Global Scan ---
        # Halton sequences provide low-discrepancy coverage, ideal for multimodal
        # functions like Rastrigin and Gallagher.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = []
            for i in range(1, n + 1):
                point = []
                for j in range(dim):
                    # Use the first 'dim' prime numbers
                    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                    base = primes[j]
                    point.append(halton(i, base))
                points.append(point)
            return np.array(points)

        # Allocate ~30% of budget to global scan, but cap at 90 calls to leave room for refinement
        n_scan = min(90, int(0.3 * budget))
        
        # Generate Halton points in [0, 1]^d, then map to [-5, 5]^d
        halton_pts = halton_sequence(n_scan, dim)
        halton_pts = halton_pts * 10.0 - 5.0
        
        # Evaluate Halton points
        for i in range(n_scan):
            if calls_used >= budget or time.time() >= deadline:
                break
            x = halton_pts[i]
            eval_func(x)

        # If no best found yet (unlikely), fallback to center
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_func(best_x)

        # --- Phase 2: Lightweight CMA-ES / (1+lambda)-ES Hybrid ---
        # Use the remaining budget for local refinement around the best point.
        # A simple (1+lambda)-ES with adaptive step size and covariance is efficient
        # for 5 dimensions. We implement a simplified CMA-ES to leverage covariance
        # adaptation for rotated functions (Rastrigin rotated, Rosenbrock).
        
        remaining_budget = budget - calls_used
        if remaining_budget <= 0:
            return best_x, best_f

        # Initialize CMA-ES state
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # Start with a moderate step size
        
        # CMA-ES Parameters
        lam = 10  # Population size
        mu = lam // 2  # Number of recombination points
        weights = np.log(lam/2) - np.log(np.arange(1, lam+1))
        weights = weights / weights.sum()
        mu_w = weights[:mu]
        mu_w = mu_w / mu_w.sum()
        mu_eff = 1.0 / np.sum(mu_w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + dim / c_sigma) / (dim + 4 + 5 * c_sigma)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Matrices
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Precompute inverse of C (identity initially)
        C_inv = np.eye(dim)
        
        # Main Loop
        while calls_used < budget and time.time() < deadline:
            # Generate population
            ys = []
            fys = []
            for i in range(lam):
                if calls_used >= budget or time.time() >= deadline:
                    break
                z = np.random.standard_normal(dim)
                # Transform z using B and D
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = eval_func(y)
                ys.append(y)
                fys.append(fy)
            
            if len(ys) < lam:
                break
                
            # Sort by fitness
            idx = np.argsort(fys)
            ys = [ys[i] for i in idx]
            fys = [fys[i] for i in idx]
            
            # Update best
            if fys[0] < best_f:
                best_f = fys[0]
                best_x = ys[0].copy()
                
            # Recombination
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += mu_w[i] * ys[i]
            
            d = y_w - x
            
            # Update path states
            norm_d = np.linalg.norm(d)
            if norm_d > 1e-16:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (d / (sigma * norm_d))
            else:
                p_sigma = (1 - c_sigma) * p_sigma
            
            h_sigma = 1.0
            if norm_d > 1e-16:
                # Approximation for h_sigma (avoiding complex trace calculations for speed)
                # A simpler check: if norm of p_sigma is too large, reduce influence
                if np.linalg.norm(p_sigma) > 1.4 * chi_n:
                    h_sigma = 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * c_c * d / sigma
            
            # Update mean
            x = y_w
            
            # Update Covariance Matrix
            C = (1 - c_1 - c_mu) * C
            if h_sigma > 0:
                C += c_1 * (2 + h_sigma * c_c * (1 - c_c)) * np.outer(p_c, p_c)
            
            for i in range(mu):
                diff = ys[i] - y_w
                C += c_mu * mu_w[i] * np.outer(diff, diff)
            
            # Eigen-decomposition to update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-16) # Ensure positive definiteness
            B = eigvecs
            D = np.sqrt(eigvals)
            
            # Update sigma
            sigma *= np.exp(c_sigma / (5 * dim) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Safety: prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f

# EVOLVE-BLOCK-END
