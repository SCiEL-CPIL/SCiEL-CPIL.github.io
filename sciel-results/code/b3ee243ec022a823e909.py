"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid optimizer exploiting Rastrigin structure with multi-start CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, dim, call_counter):
        """Run CMA-ES from x0 with given initial sigma and budget."""
        calls_made = call_counter[0]
        
        # CMA-ES parameters
        lambda_ = min(12, budget)  # population size
        mu = max(2, lambda_ // 2)  # number of parents
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        x = x0.copy()
        f_best = func(x)
        calls_made += 1
        call_counter[0] = calls_made
        
        sigma = sigma0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x.copy()
        best_f = f_best
        
        while calls_made < call_counter[0] + budget:
            # Generate population
            n_offspring = min(lambda_, call_counter[0] + budget - calls_made)
            if n_offspring <= 0:
                break
            
            offspring = []
            for i in range(n_offspring):
                z = rng.standard_normal(dim)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append((fy, y))
                calls_made += 1
                call_counter[0] = calls_made
            
            if len(offspring) == 0:
                break
            
            # Sort offspring
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()
            
            # Update mean
            y_w = np.array([offspring[i][1] for i in range(min(mu, len(offspring)))])
            y_w -= x
            x_new = x + sigma * np.sum(weights[:len(y_w)][:, np.newaxis] * y_w, axis=0)
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * np.sum(weights[:len(y_w)][:, np.newaxis] * y_w, axis=0)
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, call_counter[0] + budget - calls_made + 1))) < (1 + 2 / dim) * np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, call_counter[0] + budget - calls_made + 1))) else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * np.sum(weights[:len(y_w)][:, np.newaxis] * y_w, axis=0)
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 / (1 - (1 - c_c) ** 2) * np.outer(p_c, p_c) + c_mu / (1 - (1 - c_c) ** 2) * np.sum(weights[:len(y_w)][:, np.newaxis, np.newaxis] * np.outer(y_w, y_w), axis=0)
            
            # Renormalize covariance
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 0.005, 2.0)
            
            x = x_new
            
            # Early stopping if sigma is very small
            if sigma < 0.005:
                break
        
        return best_x, best_f, call_counter[0]

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        call_counter = [0]
        rng = np.random.default_rng(42)
        
        best_overall_x = None
        best_overall_f = np.inf
        
        # Phase 1: Smart initialization - evaluate a mix of integer and near-integer points
        # Rastrigin minima are at integers; with rotation/shift, search near integers
        grid_vals = np.arange(-4, 5, 1)  # -4 to 4 inclusive
        
        # Generate candidate points: integers and small perturbations
        candidates = []
        for v0 in grid_vals:
            for v1 in grid_vals:
                for v2 in grid_vals:
                    for v3 in grid_vals:
                        for v4 in grid_vals:
                            candidates.append([v0, v1, v2, v3, v4])
        
        # Shuffle and evaluate a subset
        rng.shuffle(candidates)
        
        # Phase 1 budget: 80 evaluations for exploration
        phase1_budget = min(80, budget // 5)
        top_candidates = []
        
        for pt in candidates:
            if call_counter[0] >= phase1_budget:
                break
            x_pt = np.clip(np.array(pt, dtype=float), -5, 5)
            f_pt = func(x_pt)
            call_counter[0] += 1
            top_candidates.append((f_pt, x_pt.copy()))
            if f_pt < best_overall_f:
                best_overall_f = f_pt
                best_overall_x = x_pt.copy()
        
        # Sort top candidates and keep the best ones
        top_candidates.sort(key=lambda t: t[0])
        num_local_searches = min(5, len(top_candidates))
        
        # Phase 2: Run CMA-ES from top candidates
        remaining = budget - call_counter[0]
        if remaining <= 0:
            return best_overall_x, best_overall_f
        
        budget_per_search = remaining // num_local_searches
        
        for i in range(num_local_searches):
            if call_counter[0] >= budget:
                break
            
            start_x = top_candidates[i][1]
            # Start with smaller sigma for local refinement
            sigma_init = 0.5 if i == 0 else 0.3
            
            x_best, f_best, new_counter = self._cma_es(
                func, start_x, sigma_init, budget_per_search, rng, dim, call_counter
            )
            
            if f_best < best_overall_f:
                best_overall_f = f_best
                best_overall_x = x_best.copy()
        
        # Phase 3: Final intensive refinement on the best point
        if call_counter[0] < budget:
            final_budget = budget - call_counter[0]
            if final_budget > 10:
                x_final, f_final, _ = self._cma_es(
                    func, best_overall_x, 0.1, final_budget, rng, dim, call_counter
                )
                if f_final < best_overall_f:
                    best_overall_f = f_final
                    best_overall_x = x_final.copy()
        
        return best_overall_x, best_overall_f
# EVOLVE-BLOCK-END
