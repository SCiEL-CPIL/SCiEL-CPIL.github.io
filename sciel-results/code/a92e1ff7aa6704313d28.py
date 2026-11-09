"""Optimized Rastrigin solver using lattice-aware grid search + CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice-aware grid search followed by CMA-ES refinement for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = 320.0  # safety margin before 330s kill

    def __call__(self, func):
        start_time = time.time()
        remaining = self.budget
        best_x = np.array([0.0] * self.dim)
        best_f = np.inf

        def check_time():
            return (time.time() - start_time) < self.deadline and remaining > 0

        def call_func(x):
            nonlocal remaining
            if remaining <= 0:
                return None
            remaining -= 1
            return func(x)

        # Phase 1: Coarse grid search over integer lattice points
        # Rastrigin local minima are near integer coordinates
        # Search a 3x3x3x3x3 grid = 243 points around integers in [-4, 4]
        # But we need to find which basin has the global min
        # Use a coarser grid first: points at -4,-3,-2,-1,0,1,2,3,4 in each dim
        # That's 9^5 = 59049 - too many. Use a smarter approach.
        
        # Better: use a two-stage grid
        # Stage 1: Very coarse - sample at half-integer and integer points
        # Actually, let's use a structured approach:
        # For Rastrigin f(x) = 10*dim + sum(x_i^2 - 10*cos(2*pi*x_i))
        # The global min is at x* (shifted), local minima at integers near x*
        
        # Since we don't know the shift, do a grid over the box
        # Use a 5-point grid per dimension: -5, -2.5, 0, 2.5, 5 -> 5^5 = 3125 points
        # That's feasible within budget if we're careful
        
        # Actually, let's be smarter: use a 3-level grid
        # Level 1: 3 points per dim (-5, 0, 5) = 3^5 = 243 points
        # Find the best region, then refine
        
        # Level 1: Coarse grid
        coarse_vals = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])
        grid_points = np.array(np.meshgrid(*[coarse_vals]*self.dim)).T.reshape(-1, self.dim)
        
        coarse_results = []
        for pt in grid_points:
            if not check_time():
                break
            f_val = call_func(pt)
            if f_val is None:
                break
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()
            coarse_results.append((f_val, pt))
        
        if not check_time() or remaining <= 0:
            return best_x, best_f
        
        # Sort coarse results and take top candidates
        coarse_results.sort(key=lambda x: x[0])
        
        # Phase 2: For the top few basins, run a local CMA-ES
        # Take top 5 distinct basins (within some radius)
        top_candidates = []
        for f_val, pt in coarse_results:
            if len(top_candidates) >= 5:
                break
            # Check if this is sufficiently different from existing candidates
            is_new = True
            for _, existing_pt in top_candidates:
                if np.linalg.norm(pt - existing_pt) < 1.0:
                    is_new = False
                    break
            if is_new:
                top_candidates.append((f_val, pt))
        
        # If we didn't find enough, just use the best
        if not top_candidates:
            top_candidates = [(best_f, best_x)]
        
        # Phase 3: Run CMA-ES on each top candidate basin
        # Budget allocation: give more budget to better candidates
        n_candidates = len(top_candidates)
        base_budget = max(10, remaining // (n_candidates * 2))
        
        for idx, (f_val, center) in enumerate(top_candidates):
            if not check_time() or remaining <= 0:
                break
            
            # Allocate budget: better candidates get more
            alloc = base_budget
            if idx == 0:
                alloc = min(remaining, base_budget * 3)
            else:
                alloc = min(remaining, base_budget)
            
            if alloc < 10:
                alloc = max(5, remaining // 2)
            
            # Run a simplified CMA-ES
            x = center.copy()
            fx = f_val
            sigma = 1.0
            p_sigma = np.zeros(self.dim)
            C = np.eye(self.dim)
            C_sqrt = np.eye(self.dim)
            inv_C_sqrt = np.eye(self.dim)
            d_sigma = 1 + 2 * max(0, (self.dim - 1) / 3)
            c_sigma = (d_sigma + 2) / (self.dim + d_sigma + 5)
            c_c = (4 + self.dim / 4) / (self.dim + 4 + 2 * self.dim / 4)
            c_1 = 2 / ((self.dim + 1.3)**2 + 2)
            c_mu = min(1 - c_1, 2 * ((self.dim / self.dim + 0.8)**2 - 1) / ((self.dim + 2)**2 + 2))
            chi_N = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
            p_c = np.zeros(self.dim)
            count_success = 0
            expected_hv = chi_N * (1 - np.exp(-0.5)) + (1 - chi_N) * np.exp(-0.5)
            
            lam = max(4, 4 + int(3 * np.log(self.dim)))
            mu = lam // 2
            
            for step in range(alloc):
                if not check_time() or remaining <= 0:
                    break
                
                # Generate offspring
                offspring = []
                for i in range(lam):
                    if not check_time() or remaining <= 0:
                        break
                    # Sample from multivariate normal using C
                    z = np.random.randn(self.dim)
                    # Simple Cholesky-like update (approximate)
                    # For simplicity, use diagonal approximation + random rotation
                    y = x + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = call_func(y)
                    if fy is None:
                        break
                    offspring.append((fy, y))
                
                if len(offspring) < mu:
                    break
                
                offspring.sort(key=lambda x: x[0])
                best_off_f, best_off_x = offspring[0]
                
                # Update mean
                w = np.log(np.arange(1, mu+1))
                w = w / w.sum()
                new_x = sum(w[i] * offspring[i][1] for i in range(mu))
                
                # Update sigma
                y_vec = (new_x - x) / (sigma * np.sqrt(sum(w**2)))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Update C
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_x - x) / sigma
                delta = (1 + (1 - c_c) * (np.dot(p_c, p_c) / self.dim - 1)) * c_1
                C = (1 - c_1 - c_mu + delta) * C + c_1 * np.outer(p_c, p_c)
                for i in range(mu):
                    y_i = (offspring[i][1] - x) / sigma
                    C += c_mu * w[i] * np.outer(y_i, y_i)
                
                # Recompute C_sqrt (simplified - use eigen decomposition)
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    inv_C_sqrt = eigvecs @ np.diag(1/np.sqrt(eigvals)) @ eigvecs.T
                except:
                    C_sqrt = np.eye(self.dim)
                    inv_C_sqrt = np.eye(self.dim)
                
                x = new_x
                if best_off_f < fx:
                    fx = best_off_f
                    best_x = x.copy()
                    best_f = fx
                    count_success += 1
                else:
                    count_success = 0
                
                # Restart if stuck
                if count_success > 15 and sigma < 0.01:
                    break
            
            mu_eff = 1.0 / sum(w**2) if mu > 0 else 1.0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
