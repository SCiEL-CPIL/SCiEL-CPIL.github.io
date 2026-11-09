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
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)

        # Phase 1: Deterministic grid search warm-up
        # Use a 3x3x3x3x3 grid over [-4, 4]^5 to bracket the minimum
        # That's 243 evaluations, leaving 257 for CMA-ES refinement
        grid_size = 3
        grid_range = [-4.0, 4.0]
        grid_points = []
        
        # Generate grid points
        step = (grid_range[1] - grid_range[0]) / (grid_size - 1)
        coords = [grid_range[0] + i * step for i in range(grid_size)]
        
        for x0 in coords:
            for x1 in coords:
                for x2 in coords:
                    for x3 in coords:
                        for x4 in coords:
                            grid_points.append([x0, x1, x2, x3, x4])
        
        # Evaluate grid points
        best_f = np.inf
        best_x = None
        calls_made = 0
        
        for pt in grid_points:
            if time.time() > deadline:
                break
            f_val = func(pt)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(pt)
        
        if best_x is None:
            # Fallback if no grid points evaluated
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1
        
        # Phase 2: CMA-ES refinement using grid's bounding box as initial distribution
        # Initialize CMA-ES with mean at best grid point and sigma based on grid spacing
        lam = 12  # Larger population for better exploration
        mu = lam // 2
        
        # Initial sigma: use grid step size for local refinement
        sigma = step * 0.5
        
        # Initial mean: best grid point found
        mean = best_x.copy()
        
        # CMA-ES weights
        weights = np.log(np.arange(lam, 0, -1)) - np.log(np.arange(lam, 0, -1)).sum()
        weights = weights[:mu] / weights[:mu].sum()
        
        # Effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Helper: eigendecomposition of C
        def update_covariance():
            nonlocal C, C_inv, C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        update_covariance()
        
        # CMA-ES loop with remaining budget
        remaining_budget = budget - calls_made
        for gen in range(remaining_budget):
            if time.time() > deadline:
                break
            
            # Generate lambda candidates
            candidates = []
            z_samples = rng.standard_normal((lam, dim))
            for j in range(lam):
                y = C_inv_sqrt @ z_samples[j]
                x_j = mean + sigma * y
                x_j = np.clip(x_j, -5, 5)
                candidates.append(x_j)
            
            # Evaluate all candidates
            fvals = []
            for x_j in candidates:
                if time.time() > deadline:
                    fvals.append(best_f)
                    continue
                f_j = func(x_j)
                calls_made += 1
                fvals.append(f_j)
                
                if f_j < best_f:
                    best_f = f_j
                    best_x = x_j.copy()
            
            fvals = np.array(fvals)
            
            # Sort by fitness
            order = np.argsort(fvals)
            x_selected = np.array([candidates[i] for i in order[:mu]])
            f_selected = fvals[order[:mu]]
            
            # Update mean
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * x_selected, axis=0)
            
            # Success check
            if np.linalg.norm(mean - mean_old) > 1e-10:
                # Update path p_sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_inv_sqrt @ (mean - mean_old))
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
                
                # Update path p_c
                p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mueff) / sigma * (mean - mean_old)
                
                # Rank-one and rank-mu updates
                delta_hsig = (1.0 if hsig else 0.0)
                C = (1 + c1 * delta_hsig + cmu) * C - c1 * np.outer(p_c, p_c)
                C -= cmu * np.sum(weights[:, None, None] * np.outer((C_inv_sqrt @ (x_selected - mean_old).T).T, (C_inv_sqrt @ (x_selected - mean_old).T)), axis=0)
                
                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                # Adapt sigma
                sigma = sigma * np.exp(cs / (dim + 3) * (np.linalg.norm(p_sigma) / np.sqrt((1 - (1 - cs)**(gen + 1))) - 1))
                
                # Clip sigma
                sigma = np.clip(sigma, 1e-6, 10.0)
            
            if calls_made >= budget:
                break
        
        return best_x, best_f
