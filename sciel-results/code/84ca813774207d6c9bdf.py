"""Combined optimizer: Rastrigin-aware grid initialization + CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Rastrigin-aware initialization followed by CMA-ES for rotated/shifted f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        calls_made = 0

        # --- Phase 0: Rastrigin-aware grid initialization ---
        # For Rastrigin f(x) = 10*dim + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Global minima are at integer coordinates. With rotation and shift,
        # the true minimum is near integer grid points in the rotated frame.
        # We sample a sparse grid of integer candidates to find a good starting point.
        
        # Generate candidate points on integer grid within [-4, 4]^dim (where minima lie)
        # But dim=5, so full grid is 9^5 = 59049 - too many. Use sparse sampling.
        
        # Strategy: sample from the set of integer points, prioritizing those near center
        # and using a few random integer points to capture the shift/rotation effect.
        
        candidates = []
        
        # Center point
        candidates.append(np.zeros(dim))
        
        # Integer grid points: sample a subset
        # For dim=5, we can't evaluate all 9^5 points. Sample strategically.
        # Add axis-aligned integer points and some random integer points.
        
        # Axis-aligned: one coordinate at -4,-3,-2,-1,0,1,2,3,4, others at 0
        for coord in range(dim):
            for val in range(-4, 5):
                pt = np.zeros(dim)
                pt[coord] = val
                candidates.append(pt)
        
        # Random integer points
        for _ in range(30):
            pt = rng.integers(-4, 5, dim).astype(float)
            candidates.append(pt)
        
        # Deduplicate and evaluate
        seen = set()
        eval_candidates = []
        for c in candidates:
            key = tuple(np.round(c, 4))
            if key not in seen:
                seen.add(key)
                eval_candidates.append(c)
        
        # Evaluate candidates, tracking best
        best_x = None
        best_fx = float('inf')
        
        for c in eval_candidates:
            if calls_made >= self.budget:
                break
            if time.time() > deadline - 5:
                break
            c_clipped = np.clip(c, -5, 5)
            fc = func(c_clipped)
            calls_made += 1
            if fc < best_fx:
                best_fx = fc
                best_x = c_clipped.copy()
        
        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls_made += 1
        
        # --- Phase 1: CMA-ES refinement around the best candidate ---
        # Use CMA-ES to adapt to the hidden rotation and fine-tune the location.
        
        x = best_x.copy()
        fx = best_fx
        sigma = 0.5  # Start with smaller sigma since we're near a good point
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Compute C_sqrt and C_inv_sqrt
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        # CMA-ES parameters
        mu = max(1, dim // 2)
        lambda_ = 2 * mu
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Run CMA-ES steps
        while calls_made + lambda_ <= self.budget:
            if time.time() > deadline - 10:
                break
            
            # Generate lambda_ candidates
            candidates_step = []
            for i in range(lambda_):
                if time.time() > deadline - 5:
                    break
                z = rng.standard_normal(dim)
                y = C_sqrt @ z
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                calls_made += 1
                candidates_step.append((fx_new, x_new, y))
            
            if len(candidates_step) < mu:
                break
            
            # Sort by fitness
            candidates_step.sort(key=lambda t: t[0])
            
            # Update best
            if candidates_step[0][0] < best_fx:
                best_fx = candidates_step[0][0]
                best_x = candidates_step[0][1].copy()
            
            # Update mean
            x_new_mean = sum(weights[i] * candidates_step[i][1] for i in range(mu))
            x_new_mean = np.clip(x_new_mean, -5, 5)
            
            # Compute evolution path for sigma
            y_w = sum(weights[i] * candidates_step[i][2] for i in range(mu))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ y_w
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2)) / chi_n < (1.4 + 2 / (dim + 1))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 5.0)
            
            # Update path for covariance
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x)
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            outer = np.outer(p_c, p_c)
            C = (1 + c_1 * (dim + 1) + c_mu - c_1 - delta_h) * C \
                + c_1 * (outer - C) \
                + c_mu * sum(weights[i] * np.outer(candidates_step[i][2], candidates_step[i][2]) for i in range(mu))
            
            # Ensure positive definiteness
            eigvals_c, eigvecs_c = np.linalg.eigh(C)
            eigvals_c = np.maximum(eigvals_c, 1e-12)
            C = eigvecs_c @ np.diag(eigvals_c) @ eigvecs_c.T
            C = (C + C.T) / 2
            
            # Update inverse and square roots
            try:
                C_inv = np.linalg.inv(C)
                C_inv = (C_inv + C_inv.T) / 2
                eigvals_inv, eigvecs_inv = np.linalg.eigh(C_inv)
                eigvals_inv = np.maximum(eigvals_inv, 1e-12)
                C_inv = eigvecs_inv @ np.diag(eigvals_inv) @ eigvecs_inv.T
                
                eigvals_c2, eigvecs_c2 = np.linalg.eigh(C)
                eigvals_c2 = np.maximum(eigvals_c2, 1e-12)
                C_sqrt = eigvecs_c2 @ np.diag(np.sqrt(eigvals_c2)) @ eigvecs_c2.T
                C_inv_sqrt = eigvecs_inv @ np.diag(np.sqrt(eigvals_inv)) @ eigvecs_inv.T
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
            
            x = x_new_mean
        
        # --- Phase 2: Final local refinement if budget remains ---
        if calls_made < self.budget and time.time() < deadline - 5:
            x_refine = best_x.copy()
            fx_refine = best_fx
            steps_left = self.budget - calls_made
            max_refine_calls = min(30, steps_left)
            
            # Adaptive local search
            step_size = max(sigma * 0.1, 0.01)
            for _ in range(max_refine_calls // dim):
                if calls_made >= self.budget or time.time() > deadline - 3:
                    break
                improved = False
                for coord in range(dim):
                    if calls_made >= self.budget or time.time() > deadline - 3:
                        break
                    for direction in [-1, 1]:
                        if calls_made >= self.budget or time.time() > deadline - 3:
                            break
                        x_trial = x_refine.copy()
                        x_trial[coord] += direction * step_size
                        x_trial = np.clip(x_trial, -5, 5)
                        ft = func(x_trial)
                        calls_made += 1
                        if ft < fx_refine:
                            x_refine = x_trial
                            fx_refine = ft
                            improved = True
                            if ft < best_fx:
                                best_fx = ft
                                best_x = x_refine.copy()
                if not improved:
                    step_size *= 0.5
                    if step_size < 0.001:
                        break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
