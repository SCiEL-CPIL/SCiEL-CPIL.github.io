"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hierarchical deterministic grid scan followed by localized CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse grid scan to identify the global basin.
        # Use ~400 calls for a structured grid over [-5, 5]^dim.
        # For dim=5, we can't do a full 5D grid with 400 points in a fine way,
        # so we use a Latin Hypercube or a stratified random sampling approach,
        # but the directive says "coarse grid scan". Let's use a deterministic
        # space-filling approach: we'll use a grid in lower dimensions and
        # random in others, or simply a well-distributed set of points.
        # Actually, for 5D with 400 points, a full grid is impossible (even 3^5=243,
        # 4^5=1024). So let's use 3^5 = 243 grid points + some refinement,
        # or use a Halton sequence for quasi-random sampling which is deterministic.
        
        # Let's use a Halton sequence for the first ~350 points to cover the space well.
        # Then use the remaining ~150 for a local CMA-ES around the best point.
        
        # Actually, let's stick closer to the directive: "coarse grid scan".
        # We'll use a grid with 3 points per dimension: 3^5 = 243 points.
        # That leaves 500 - 243 = 257 calls for local optimization.
        
        # Generate grid points
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            grid_points.append([i * 5 - 5, j * 5 - 5, k * 5 - 5, 
                                                 l * 5 - 5, m * 5 - 5])
        # This gives 243 points at -5, 0, 5 in each dimension
        
        # Evaluate all grid points
        best_x = None
        best_f = np.inf
        for point in grid_points:
            x = np.array(point, dtype=float)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x
        
        # Phase 2: Localized CMA-ES around the best grid point
        # Use remaining budget for CMA-ES
        remaining_budget = budget - len(grid_points)
        
        # Initialize CMA-ES parameters
        x = best_x.copy()
        sigma = 1.0  # Start with a moderate step size
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim)
        ps = np.zeros(dim)  # Path for covariance adaptation
        pc = np.zeros(dim)  # Path for step-size adaptation
        cs = 1 / (dim + 4)  # Step-size adaptation time constant
        c1 = 2 / ((dim + 1.3) ** 2 + 10)  # Rank-one update weight
        cmu = min(1 - c1, 2 * ((dim - 2) + 2 / (dim + 1)) / ((dim + 2) ** 2 + 15))  # Rank-mu update weight
        cmu = min(cmu, 1 - c1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Number of offspring per iteration
        lambda_ = min(4 * dim, remaining_budget // 2) if remaining_budget > 10 else 2
        lambda_ = max(lambda_, 2)
        
        # CMA-ES loop
        calls_used = len(grid_points)
        while calls_used < budget:
            # Generate lambda_ offspring
            offspring = []
            offspring_f = []
            for _ in range(lambda_):
                if calls_used >= budget:
                    break
                # Sample from N(x, sigma^2 * C)
                z = np.random.default_rng().standard_normal(dim)
                # Whitening transformation
                y = x + sigma * (C_inv @ z)  # Simplified, should use Cholesky
                # Actually, let's use a simpler approach: just use standard normal scaled by sigma
                # and apply the covariance via a simple method
                y = x + sigma * np.random.default_rng().standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append(y)
                offspring_f.append(fy)
                calls_used += 1
            
            if not offspring:
                break
            
            # Sort by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]
            
            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_offspring[0].copy()
            
            # Update mean
            mu_eff = int(np.floor(lambda_ / 2))
            weights = np.ones(mu_eff)
            weights[0] = 2
            weights /= weights.sum()
            
            new_mean = np.sum([weights[i] * sorted_offspring[i] for i in range(mu_eff)], axis=0)
            
            # Update paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv @ (new_mean - x) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used - len(grid_points) + 1)))) < \
                   (1 + 2 / (dim + 1)) * chi_n
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * (new_mean - x) / sigma
            
            # Update covariance
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                c1 * np.outer(pc, pc) + \
                cmu * np.sum([weights[i] * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x) 
                              for i in range(mu_eff)], axis=0)
            
            # Update sigma
            sigma *= np.exp(cs / (dim + 4) * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            x = new_mean
            
            # Update C_inv (simplified, should use Cholesky decomposition)
            try:
                C_inv = np.linalg.inv(C)
            except:
                C_inv = np.eye(dim)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
