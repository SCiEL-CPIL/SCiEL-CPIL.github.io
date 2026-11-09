"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Nelder-Mead refinement for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills after 330s. We aim to finish well before.
        deadline = start_time + 300.0
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Initial Exploration
        # Sample a few points to find a reasonable starting basin.
        # Rastrigin has many local minima, so a few random starts are cheap.
        n_starts = 5
        for _ in range(n_starts):
            if call_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            if fx is None:
                break

        # Phase 2: CMA-ES Global Search
        # Use CMA-ES to navigate the landscape.
        # We reserve some budget for Nelder-Mead later.
        cma_budget_limit = int(budget * 0.7)
        
        if call_count < cma_budget_limit and time.time() < deadline:
            # CMA-ES Parameters
            # lambda_=6, mu=3 is efficient for small dims and rugged landscapes
            lambda_ = 6
            mu = lambda_ // 2  # mu = 3
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            # Initialize CMA-ES state
            x_mean = best_x.copy()
            sigma = 2.0  # Start with a relatively large step to explore
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            gen = 0
            stagnation_count = 0
            last_best_f = best_f
            max_stagnation = 20  # Generations without improvement before restarting/stopping CMA

            while call_count < cma_budget_limit and time.time() < deadline:
                # Generate offspring
                offspring = []
                offspring_f = []
                
                for k in range(lambda_):
                    if call_count >= cma_budget_limit or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = call_func(y)
                    if fy is None:
                        break
                    offspring.append(y)
                    offspring_f.append(fy)
                
                if len(offspring) < mu:
                    break
                
                # Sort by fitness
                sorted_idx = np.argsort(offspring_f)
                selected_x = [offspring[i] for i in sorted_idx[:mu]]
                
                # Check for improvement
                current_best_f = min(offspring_f)
                if current_best_f < last_best_f - 1e-10:
                    stagnation_count = 0
                    last_best_f = current_best_f
                else:
                    stagnation_count += 1
                
                # If stagnated too long, break out to use Nelder-Mead or restart
                if stagnation_count > max_stagnation:
                    break

                # Weighted mean
                m = sum(w * xk for w, xk in zip(weights, selected_x))

                # Update evolution paths
                diff = m - x_mean
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ diff)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * diff

                # Sigma adaptation
                # h_sigma condition
                h_sigma = 1.0
                if np.linalg.norm(p_sigma) > 0:
                    norm_ps = np.linalg.norm(p_sigma)
                    threshold = (1.4 + 2 / (dim + 1)) * chi_n
                    # Approximation of the expectation of the norm
                    # Using a simple check is often sufficient for small dims
                    if norm_ps / chi_n < threshold:
                        sigma *= np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))

                # Update covariance matrix
                delta_d = (1 + c_mu * (mueff - 1)) * diff
                rank_mu = np.outer(delta_d, delta_d)
                
                # h_c condition
                h_c = 1.0
                norm_pc = np.linalg.norm(p_c)
                if norm_pc > 0:
                    threshold_c = (1.4 + 2 / (dim + 1)) * chi_n
                    if norm_pc / chi_n < threshold_c:
                        h_c = 1.0
                    else:
                        h_c = 0.0
                else:
                    h_c = 1.0
                
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * np.eye(dim)) + c_mu * rank_mu

                # Update C_inv_sqrt periodically to maintain stability
                if gen % 5 == 0:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-10)
                        C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                    except np.linalg.LinAlgError:
                        C_inv_sqrt = np.eye(dim)

                x_mean = m
                gen += 1

        # Phase 3: Nelder-Mead Local Refinement
        # Use the remaining budget to polish the best solution found.
        # We can run NM on the best point. If budget allows, we can also try on a few
        # other recent good points to escape local minima.
        
        # Simple Nelder-Mead implementation
        def nelder_mead(x_start, max_calls, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            nonlocal best_x, best_f, call_count, deadline
            
            # Initialize simplex
            simplex = [x_start.copy()]
            simplex_f = [call_func(x_start)]
            if simplex_f[0] is None:
                return
            
            for d in range(dim):
                if call_count >= max_calls or time.time() > deadline:
                    return
                pt = x_start.copy()
                # Step size for initial simplex: small enough to stay in basin, large enough to define it
                step = 0.1 if x_start[d] > 4.9 else -0.1 if x_start[d] < -4.9 else 0.1
                pt[d] += step
                pt = np.clip(pt, -5, 5)
                fpt = call_func(pt)
                if fpt is None:
                    return
                simplex.append(pt)
                simplex_f.append(fpt)
            
            if len(simplex) < dim + 1:
                return

            # NM Iterations
            while call_count < max_calls and time.time() < deadline:
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Check for convergence (optional, but good for stopping early)
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-8:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = call_func(x_ref)
                if f_ref is None:
                    break

                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = call_func(x_exp)
                    if f_exp is None:
                        break
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
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = call_func(x_con)
                    if f_con is None:
                        break
                    if f_con < f_ref:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if call_count >= max_calls or time.time() > deadline:
                                return
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_shr = call_func(simplex[i])
                            if f_shr is None:
                                return
                            simplex_f[i] = f_shr
                        # Re-sort after shrink is not strictly necessary if we just replaced, 
                        # but the next iteration will sort anyway.

        # Run NM on the global best
        remaining_budget = budget - call_count
        if remaining_budget > 10:
            # Use most of the remaining budget for NM on the best point
            nm_calls = int(remaining_budget * 0.9)
            nelder_mead(best_x, nm_calls)
            
            # If we have significant budget left, try NM on a perturbed version of best_x
            # to see if we can find a better nearby basin
            if call_count < budget - 10:
                perturbed = best_x + rng.normal(0, 0.5, dim)
                perturbed = np.clip(perturbed, -5, 5)
                # Check if perturbed point is actually better or different
                # We just run NM on it, it will update best_x if it finds something better
                nelder_mead(perturbed, budget - call_count)

        return best_x, best_f
# EVOLVE-BLOCK-END
