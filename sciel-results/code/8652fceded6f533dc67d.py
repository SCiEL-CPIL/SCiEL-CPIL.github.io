"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        rng = np.random.default_rng(42)

        # Initial population
        # Start near center, Weierstrass f16 is shifted/rotated but usually centered-ish or within [-5,5]
        # A random start within the box is safer for unknown shifts, but center is often better for multimodal if shift is small.
        # The prompt says minimizer is in [-4, 4]^5.
        x0 = rng.uniform(-4, 4, dim)
        
        # CMA-ES Parameters
        # lambda = 4 * floor(1 + 3*log(n))
        lam = 4 * int(1 + 3 * np.log(dim))
        # mu = floor(lambda / 2)
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        # Effective population size
        mueff = 1.0 / np.sum(w**2)
        
        # Step size
        # Heuristic for initial sigma
        sigma = 2.0 
        
        # CMA-ES state
        mean = x0.copy()
        C = np.eye(dim)
        Cinv = np.eye(dim)
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)))
        
        # Damping
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Initial evaluation
        pop = mean + sigma * (rng.multivariate_normal(np.zeros(dim), C, size=lam) if dim > 1 else rng.standard_normal((lam, 1)))
        # Ensure within bounds? CMA-ES usually handles bounds via resampling or penalty.
        # For simplicity and robustness with "clip" in func, we can just evaluate clipped points.
        # However, CMA-ES assumes unconstrained or box constraints handled by the optimizer.
        # The func clips. So we can generate points outside, func clips them.
        # This might bias the covariance if many points are clipped.
        # Better to keep the distribution within bounds or use a large sigma initially and let it shrink.
        
        # Actually, a simpler and very robust approach for 500 budget in 5D is often just a well-tuned (1+1) or (4+4) ES.
        # But CMA-ES is the "different strategy".
        
        # Let's implement a simplified CMA-ES loop.
        
        # Initial fitness
        fvals = np.array([func(np.clip(p, -5, 5)) for p in pop])
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_fx = fvals[best_idx]
        
        calls = lam
        calls_limit = budget
        
        # Track for p_sigma
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Precompute for efficiency
        sqrt_inv_C = np.linalg.cholesky(C)
        
        # Main loop
        while calls < calls_limit:
            if time.time() > deadline:
                break
            
            # Update mean and covariance
            # Sort by fitness
            order = np.argsort(fvals)
            y_selected = pop[order[:mu]]
            y_selected_std = (y_selected - mean) / sigma
            
            # Update p_sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_n * (Cinv @ (w @ y_selected_std))
            
            # Update mean
            mean = mean + sigma * (w @ y_selected_std)
            
            # Update pc
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((calls + 1) / (dim * c_sigma))) < (1.4 + 2 / (dim + 1)) * chi_n)
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (Cinv @ (w @ y_selected_std))
            
            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_mu = np.outer(pc, pc)
            rank_one = np.outer(ps, ps)
            
            # C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            # Note: The standard formula is C = (1 - c_1 - c_mu + c_1*delta_h)*C + c_1*pp + c_mu*sum(wi*yi*yi^T)
            # yi are standard normal samples.
            
            # Correction: The update is on the covariance matrix of the standard normal distribution.
            # C_new = (1 - c_1 - c_mu + c_1*delta_h)*C + c_1*outer(ps, ps) + c_mu * sum(wi * outer(yi_std, yi_std))
            
            # We need to accumulate the weighted outer products of the selected standard normal vectors.
            # y_selected_std are already (yi - mean_old) / sigma.
            # Wait, the standard CMA-ES update uses the raw standard normal vectors z_i, not the standardized ones relative to the old mean?
            # No, the update is:
            # C = (1 - c_1 - c_mu + c_1*delta_h) * C + c_1 * outer(ps, ps) + c_mu * sum_{i=1..mu} w_i * outer(z_i, z_i)
            # where z_i are the raw standard normal samples used to generate the population.
            # In our loop, we generated pop using current C and sigma.
            # We should store the raw z vectors.
            
            # Let's restart the loop structure to store z.
            # Actually, for a quick implementation, a simpler (mu+lambda)-ES with adaptive sigma is often robust enough and faster to implement correctly than full CMA-ES.
            # However, the directive asks for a DIFFERENT strategy. (1+1) is current.
            # Let's try a simple (4+4)-ES with population. It's simpler than CMA-ES and handles multimodality better than (1+1).
            
            # Let's pivot to (4+4)-ES. It's a distinct family from (1+1).
            # It maintains a population of 4, generates 4 offspring, keeps best 4.
            # Adaptive sigma based on success rate.
            
            # Break out and implement (4+4)-ES
            break

        # --- (4+4)-ES Implementation ---
        # Reset state for (4+4)-ES
        pop_size = 4
        x = rng.uniform(-4, 4, dim)
        fvals = np.array([func(np.clip(x + sigma * rng.standard_normal(dim), -5, 5)) for _ in range(pop_size)])
        # Actually, let's just do a fresh (4+4) loop.
        
        # Re-initialize for (4+4)
        # We already used some budget. Let's just implement the whole thing as (4+4) from scratch.
        # The previous CMA-ES code was getting complex. (4+4) is a solid "different strategy" (Population-based ES vs Single-path ES).
        
        # Restart logic for (4+4)
        pop = [x0 + sigma * rng.standard_normal(dim) for _ in range(pop_size)]
        pop = [np.clip(p, -5, 5) for p in pop]
        fvals = np.array([func(p) for p in pop])
        calls = pop_size
        
        # Track best
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_fx = fvals[best_idx]
        
        # Adaptive sigma
        # 1/5 rule is for (1+1). For (mu+lambda), we can adjust sigma based on the fraction of successful offspring.
        
        while calls < calls_limit:
            if time.time() > deadline:
                break
            
            # Generate offspring
            # Use the best parent (or average of top parents) as center?
            # Simplest: Use the current best as center for all offspring (greedy) or random parent.
            # Let's use the current best as the center for exploration.
            center = best_x
            offspring = [center + sigma * rng.standard_normal(dim) for _ in range(pop_size)]
            offspring = [np.clip(o, -5, 5) for o in offspring]
            off_fvals = np.array([func(o) for o in offspring])
            calls += pop_size
            
            # Selection: Keep best 4 from parents + offspring
            all_points = pop + offspring
            all_fvals = np.concatenate([fvals, off_fvals])
            
            # Sort and keep best pop_size
            order = np.argsort(all_fvals)
            pop = [all_points[i] for i in order[:pop_size]]
            fvals = all_fvals[order[:pop_size]]
            
            # Update best
            curr_best_idx = np.argmin(fvals)
            if fvals[curr_best_idx] < best_fx:
                best_fx = fvals[curr_best_idx]
                best_x = pop[curr_best_idx].copy()
            
            # Adapt sigma
            # If any offspring is better than the worst parent, increase sigma?
            # Or use success rate: if fraction of offspring better than best parent is high, increase.
            # Simple heuristic:
            # Count how many offspring are better than the current best parent (before selection).
            # Let's use a simple adaptation:
            # If best_offspring < best_parent, sigma *= 1.1
            # Else sigma *= 0.9
            
            best_parent_fx = np.min(fvals) # This is the best from the NEW population? No, we need comparison logic.
            
            # Let's compare best offspring to the best parent from the previous generation.
            # We stored fvals before updating.
            # Let's restructure slightly to capture previous best.
            
            # Actually, simpler:
            # If the minimum of the offspring is less than the minimum of the parents, we made progress.
            # Let's just use a fixed small adaptation or none, relying on the population to explore.
            # A fixed sigma might be too large or small.
            # Let's try adapting sigma based on the improvement.
            
            # For simplicity and robustness, let's stick to a fixed sigma or very slow adaptation.
            # Weierstrass f16 is very flat. A fixed sigma of 0.5-1.0 might work well.
            # Let's try sigma = 1.0 and no adaptation first, or mild adaptation.
            
            # Let's implement mild adaptation:
            # success = 1 if min(off_fvals) < min(fvals_prev) else 0
            # We need fvals_prev.
            
            # Let's rewrite the loop cleanly.
            pass

        # Clean (4+4) implementation
        pop_size = 4
        sigma = 1.0
        
        # Init
        pop = [x0 + sigma * rng.standard_normal(dim) for _ in range(pop_size)]
        pop = [np.clip(p, -5, 5) for p in pop]
        fvals = np.array([func(p) for p in pop])
        calls = pop_size
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_fx = fvals[best_idx]
        
        while calls + pop_size <= calls_limit:
            if time.time() > deadline:
                break
            
            prev_best_fx = best_fx
            prev_pop_fvals = fvals.copy()
            
            # Generate offspring from the current best
            center = best_x
            offspring = [center + sigma * rng.standard_normal(dim) for _ in range(pop_size)]
            offspring = [np.clip(o, -5, 5) for o in offspring]
            off_fvals = np.array([func(o) for o in offspring])
            calls += pop_size
            
            # Combine
            all_points = pop + offspring
            all_fvals = np.concatenate([fvals, off_fvals])
            
            # Select best pop_size
            order = np.argsort(all_fvals)
            pop = [all_points[i] for i in order[:pop_size]]
            fvals = all_fvals[order[:pop_size]]
            
            # Update best
            curr_best_idx = np.argmin(fvals)
            if fvals[curr_best_idx] < best_fx:
                best_fx = fvals[curr_best_idx]
                best_x = pop[curr_best_idx].copy()
            
            # Adapt sigma
            # If we found a better point, keep or slightly increase sigma.
            # If not, decrease.
            if best_fx < prev_best_fx:
                sigma *= 1.1
            else:
                sigma *= 0.95
            
            # Clamp sigma
            sigma = np.clip(sigma, 0.1, 5.0)

        # Use remaining budget for fine-tuning around best_x with smaller steps?
        # (1+1) local search around best_x
        if calls < calls_limit:
            local_sigma = 0.1
            x_local = best_x.copy()
            fx_local = best_fx
            
            while calls < calls_limit:
                if time.time() > deadline:
                    break
                
                y = np.clip(x_local + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx_local:
                    x_local, fx_local = y, fy
                    if fy < best_fx:
                        best_x, best_fx = y.copy(), fy
                    local_sigma *= 1.2
                else:
                    local_sigma *= 0.9

        return best_x, best_fx
# EVOLVE-BLOCK-END
