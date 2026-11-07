"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer for Lunacek bi-Rastrigin (f24).
    Phase 1: Dense Latin Hypercube Sampling for global basin identification.
    Phase 2: CMA-ES for adaptive global/local exploitation.
    Phase 3: Nelder-Mead simplex for final local polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin for 300s limit

        global_best = None
        global_fx = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, global_best, global_fx
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < global_fx:
                global_fx = fx
                global_best = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # --- Phase 1: Global Search with Dense LHS ---
        # Use 20% of budget for global search to ensure basin identification
        n_global = int(0.2 * budget)
        n_global = max(n_global, 50)
        n_global = min(n_global, budget - 10)
        
        # LHS points in [-4.5, 4.5]^dim
        # The optimum is guaranteed to be in [-4, 4]^5
        lhs_points = np.zeros((n_global, dim))
        for d in range(dim):
            edges = np.linspace(-4.5, 4.5, n_global + 1)
            for i in range(n_global):
                lhs_points[i, d] = rng.uniform(edges[i], edges[i+1])
        
        # Shuffle to ensure diversity in evaluation order
        perm = rng.permutation(n_global)
        lhs_points = lhs_points[perm]

        for i in range(n_global):
            if calls_made >= budget - 10 or time.time() > deadline - 10:
                break
            evaluate(lhs_points[i])

        if global_best is None:
            return None, np.inf

        # --- Phase 2: CMA-ES for Adaptive Optimization ---
        # CMA-ES adapts step size and covariance, good for multimodal functions
        if calls_made < budget - 20 and time.time() < deadline - 10:
            x0 = global_best.copy()
            
            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            sigma = 1.0  # Initial step size
            mean = x0.copy()
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Evolution strategy parameters
            cs = (pop_size + 2) / ((dim + 2) ** 2 + pop_size)
            ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cs
            cc = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
            c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
            cmu = min(1 - c1, 2 * (pop_size / 2 - 1 + 1 / pop_size) / ((dim + 2) ** 2 + pop_size))
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / (weights ** 2).sum()
            
            # Precompute C^0.5 using Cholesky decomposition
            def get_c_sqrt(C):
                try:
                    # Add small regularization for numerical stability
                    C_reg = C + 1e-12 * np.eye(dim)
                    L = np.linalg.cholesky(C_reg)
                    return L
                except np.linalg.LinAlgError:
                    # Fallback to eigen decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    return eigvecs @ np.diag(np.sqrt(eigvals))
            
            C_sqrt = get_c_sqrt(C)
            
            max_cma_iters = 100
            for it in range(max_cma_iters):
                if calls_made >= budget - 10 or time.time() > deadline - 5:
                    break
                
                # Generate population
                offspring = []
                for _ in range(pop_size):
                    z = rng.standard_normal(dim)
                    x = mean + sigma * (C_sqrt @ z)
                    x = np.clip(x, -5.0, 5.0)
                    fx = evaluate(x)
                    offspring.append((fx, x))
                
                if len(offspring) < pop_size:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda x: x[0])
                
                # Update mean
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * offspring[i][1]
                
                # Update evolution paths
                delta_mean = new_mean - mean
                mean = new_mean
                
                ps = (1 - cs) * ps + cs * np.sqrt(mueff) * delta_mean / sigma
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1)) < \
                        (1.4 + 2 / (dim + 1)) * np.sqrt((mueff - 1 + 1) / (dim + 1)) else 0
                pc = (1 - cc) * pc + hsig * cc * delta_mean / sigma
                
                # Update covariance matrix
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    diff = offspring[i][1] - mean
                    rankmu += weights[i] * np.outer(diff, diff)
                
                C = (1 + c1 + cmu - c1) * C + c1 * (rank1 - C) + cmu * rankmu
                
                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = get_c_sqrt(C)
                
                # Update step size
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(mueff) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
        
        # --- Phase 3: Local Polish with Nelder-Mead ---
        if global_best is not None and calls_made < budget - 5 and time.time() < deadline - 5:
            x0 = global_best.copy()
            
            # Nelder-Mead parameters
            spread = 0.3
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i, i-1] += spread
            
            # Evaluate initial simplex
            simplex_f = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= budget - 5 or time.time() > deadline - 5:
                    break
                simplex_f[i] = evaluate(simplex[i])
            
            # Nelder-Mead constants
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma_nm = 0.5   # Shrinkage
            
            # Main Nelder-Mead loop
            max_iters = 200
            for it in range(max_iters):
                if calls_made >= budget - 5 or time.time() > deadline - 5:
                    break
                
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Convergence check
                if np.max(simplex_f) - np.min(simplex_f) < 1e-10 * (1 + abs(np.min(simplex_f))):
                    break
                
                x_worst = simplex[-1]
                x_best = simplex[0]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                f_ref = evaluate(x_ref)
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    x_con = x_centroid + rho * (x_worst - x_centroid)
                    f_con = evaluate(x_con)
                    if f_con < simplex_f[-1]:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma_nm * (simplex[i] - x_best)
                            if calls_made >= budget - 5 or time.time() > deadline - 5:
                                break
                        for i in range(1, dim + 1):
                            if calls_made >= budget - 5 or time.time() > deadline - 5:
                                break
                            simplex_f[i] = evaluate(simplex[i])
        
        return global_best, global_fx
# EVOLVE-BLOCK-END
