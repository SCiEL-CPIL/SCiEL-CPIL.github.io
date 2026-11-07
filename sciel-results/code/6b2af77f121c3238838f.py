"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid CMA-ES and Nelder-Mead optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        call_count = 0
        rng = np.random.default_rng(42)

        # Helper to evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return float('inf')
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: CMA-ES
        # We allocate 70% of the budget to CMA-ES
        cma_budget = int(budget * 0.7)
        
        # CMA-ES Parameters
        # Smaller population for faster adaptation in low dimensions
        lambda_pop = 6
        mu = lambda_pop // 2  # Selection pressure
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        # State variables
        mean = np.zeros(dim)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        sigma = 2.0
        
        # Precompute constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Initial evaluation
        f_mean = evaluate(mean)
        
        # Main CMA-ES Loop
        while call_count < cma_budget and time.time() < deadline:
            # Generate population
            Z_list = []
            X = []
            F = []
            
            # Try to compute Cholesky once per generation if possible, or per sample
            # For dim=5, per sample is fast enough and safer if C becomes ill-conditioned
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim) # Fallback
            
            for i in range(lambda_pop):
                if call_count >= cma_budget or time.time() > deadline:
                    break
                Z = rng.standard_normal(dim)
                y = L @ Z
                x = mean + sigma * y
                f = evaluate(x)
                
                Z_list.append(Z)
                X.append(x)
                F.append(f)
            
            if len(X) < mu:
                break
                
            # Sort by fitness
            sort_idx = np.argsort(F)
            X = [X[i] for i in sort_idx]
            Z_list = [Z_list[i] for i in sort_idx]
            F = [F[i] for i in sort_idx]
            
            # Update Mean
            old_mean = mean.copy()
            mean = np.sum([w * X[i] for i, w in enumerate(weights)], axis=0)
            
            # Update Covariance Matrix
            # Normalize Z for covariance update
            Z_norm = [Z_list[i] / sigma for i in range(mu)]
            
            # Rank-mu update
            C_update = np.zeros((dim, dim))
            for i in range(mu):
                C_update += weights[i] * np.outer(Z_norm[i], Z_norm[i])
            
            # Update pc and ps
            step = (mean - old_mean) / sigma
            pc = (1 - c_sigma) * pc + c_sigma * np.sqrt(mu_eff) * step
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * step
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_sigma)**2) * C) + c_mu * C_update
            C = (C + C.T) / 2
            
            # Adapt sigma
            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))
            
            # Clamp sigma
            if sigma < 1e-12:
                break
            if sigma > 10:
                sigma = 10

        # Phase 2: Nelder-Mead Polish
        # Use the best point found by CMA-ES as the starting point
        # Allocate remaining budget (up to 30% or until time limit)
        nm_budget = budget - call_count
        
        if nm_budget > 10 and time.time() < deadline:
            # Initialize simplex
            # Standard simplex: best point + delta * e_i
            delta = 0.5
            simplex = [best_x.copy()]
            for i in range(dim):
                point = best_x.copy()
                point[i] += delta
                simplex.append(point)
            
            # Evaluate initial simplex
            simplex_f = []
            for p in simplex:
                if call_count >= budget or time.time() > deadline:
                    break
                simplex_f.append(evaluate(p))
            
            if len(simplex_f) == dim + 1:
                # NM Loop
                while call_count < budget and time.time() < deadline:
                    # Sort simplex by fitness
                    idx = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in idx]
                    simplex_f = [simplex_f[i] for i in idx]
                    
                    # Check convergence
                    if simplex_f[0] - simplex_f[-1] < 1e-12:
                        break
                    
                    # Centroid of all but worst point
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    xr = centroid + (centroid - simplex[-1])
                    fr = evaluate(xr)
                    
                    if fr < simplex_f[0]:
                        # Expansion
                        xe = centroid + 2 * (centroid - simplex[-1])
                        fe = evaluate(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                    elif fr < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    else:
                        # Contraction
                        # Outside contraction if fr < worst, inside if fr >= worst
                        if fr < simplex_f[-1]:
                            # Outside contraction
                            xc = centroid + 0.5 * (centroid - simplex[-1])
                            fc = evaluate(xc)
                            if fc <= fr:
                                simplex[-1] = xc
                                simplex_f[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, dim + 1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    if call_count >= budget or time.time() > deadline:
                                        break
                                    simplex_f[i] = evaluate(simplex[i])
                                break
                        else:
                            # Inside contraction
                            xc = centroid - 0.5 * (centroid - simplex[-1])
                            fc = evaluate(xc)
                            if fc < simplex_f[-1]:
                                simplex[-1] = xc
                                simplex_f[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, dim + 1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    if call_count >= budget or time.time() > deadline:
                                        break
                                    simplex_f[i] = evaluate(simplex[i])
                                break

        return best_x, best_f

# EVOLVE-BLOCK-END
