"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Black-box continuous optimizer for BBOB f22 (Gallagher 21 peaks).
    Uses a two-phase strategy:
      Phase 1: CMA-ES with a small population to quickly find a good basin.
      Phase 2: Fine-tuning with a local search (Nelder-Mead or (1+1)-ES)
               to polish the solution.
    
    The key insight is that f22 has many local minima, so we need robust
    global exploration early on, then exploit locally.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: CMA-ES with small population ---
        # Use about 70% of the budget for CMA-ES
        n_es = int(0.7 * budget)
        
        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        
        # Initial sigma: large enough to explore [-5, 5]^5
        sigma = 2.0
        
        # Population size for CMA-ES: lambda = 4 + floor(3*log(dim))
        lam = 4 + int(3 * np.log(dim)) if dim > 1 else 4
        mu = lam // 2  # number of parents
        
        # Weights for recombination
        w = np.log(lam / 2 + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        # Step size control parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        best_x = x.copy()
        best_f = fx
        n_evals = 1
        
        for gen in range(n_es - 1):
            if n_evals >= budget:
                break
                
            # Generate population
            ys = []
            fs = []
            for i in range(lam):
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                n_evals += 1
                if n_evals >= budget:
                    break
            
            if n_evals >= budget:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = np.array(ys[:len(order)])
            fs = np.array(fs[:len(order)])
            ys = ys[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
            
            # Recombination
            y_mean = np.sum(w[:, None] * ys[:mu], axis=0)
            
            # Update mean
            x = y_mean
            
            # Update covariance matrix
            y_diff = (y_mean - x) / sigma
            pc = (1 - c_c) * pc + c_c * y_diff
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (y_mean - x)
            
            # Covariance matrix adaptation
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) / chi_n < 1.4 + 2 / (dim + 1)
            B, D = self._update_covariance(B, D, pc, ps, c_c, c_1, c_mu, w, ys[:mu], x, sigma, hsig, dim, gen)
            
            # Sigma adaptation
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)
            
            # Check convergence
            if np.all(np.abs(D) < 1e-6) or sigma < 1e-8:
                break
        
        # --- Phase 2: Local refinement with Nelder-Mead ---
        remaining = budget - n_evals
        if remaining > 5 and best_f > -1000:
            x = best_x.copy()
            fx = best_f
            
            # Nelder-Mead simplex method
            simplex = [x.copy()]
            for i in range(dim):
                xi = x.copy()
                step = max(0.1, 0.05 * sigma)
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
            
            fsimplex = [fx] + [func(s) for s in simplex[1:]]
            n_evals += dim
            
            for it in range(min(remaining - 1, 200)):
                if n_evals >= budget:
                    break
                
                # Sort simplex
                order = np.argsort(fsimplex)
                simplex = [simplex[i] for i in order]
                fsimplex = [fsimplex[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                n_evals += 1
                
                if fsimplex[0] <= fr < fsimplex[-2]:
                    simplex[-1] = xr
                    fsimplex[-1] = fr
                elif fr < fsimplex[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    n_evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fsimplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        fsimplex[-1] = fr
                else:
                    # Contraction
                    beta = 0.5
                    xc = centroid + beta * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    n_evals += 1
                    if fc < fsimplex[-1]:
                        simplex[-1] = xc
                        fsimplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fsimplex[i] = func(simplex[i])
                            n_evals += 1
                            if n_evals >= budget:
                                break
                        
                        if n_evals >= budget:
                            break
                
                # Update best
                min_idx = np.argmin(fsimplex)
                if fsimplex[min_idx] < best_f:
                    best_f = fsimplex[min_idx]
                    best_x = simplex[min_idx].copy()
        
        return best_x, best_f

    def _update_covariance(self, B, D, pc, ps, c_c, c_1, c_mu, w, ys, x, sigma, hsig, dim, gen):
        """Update the eigendecomposition of the covariance matrix."""
        # This is a simplified update; in practice, CMA-ES updates the full matrix
        # For simplicity, we'll use a diagonal approximation
        c = np.outer(pc, pc)
        ps_norm_sq = np.dot(ps, ps)
        delta_hsig = (1 - hsig) * c_c * (2 - c_c)
        
        # Update eigenvalues
        D2 = D**2
        D2 += -2 * c_1 * D2
        D2 += c_1 * (2 * mueff_1 * D2 + delta_hsig * np.outer(pc, pc).diagonal() if False else 0)
        
        # Simplified: just adjust D slightly
        D = np.maximum(D * 0.99 + 0.01 * np.mean(D), 1e-8)
        
        return B, D
# EVOLVE-BLOCK-END
