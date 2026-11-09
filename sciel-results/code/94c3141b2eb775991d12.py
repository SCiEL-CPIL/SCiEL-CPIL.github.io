"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer for Bent Cigar function f12 using CMA-ES.
    
    Strategy:
    1. Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) which is 
       highly effective for anisotropic functions like Bent Cigar.
    2. CMA-ES adapts the covariance matrix of the sampling distribution to 
       align with the principal axes of the function's contours, efficiently 
       navigating the narrow valley structure.
    3. Strict budget and deadline management.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit
        d = self.dim
        budget = self.budget

        state = {
            "count": 0,
            "ok": True,
            "best_x": None,
            "best_f": np.inf
        }

        def wf(x):
            if not state["ok"]:
                return 1e300
            if time.perf_counter() > deadline:
                state["ok"] = False
                return 1e300
            
            x = np.asarray(x, dtype=float)
            if not np.all(np.isfinite(x)):
                return 1e300
                
            try:
                v = float(func(x))
            except Exception:
                return 1e300
                
            state["count"] += 1
            if not np.isfinite(v):
                return 1e300
                
            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = x.copy()
                
            return v

        # CMA-ES Parameters
        # Population size lambda
        # lambda = 4 + 3 * log(d) is a standard recommendation
        lam = int(4 + 3 * np.log(d))
        # Cap lambda if budget is small, but 500 is plenty for d=5
        lam = min(lam, max(10, budget // 20))
        
        # Step size sigma
        # Initial sigma should cover the box roughly. Box is [-5, 5], range 10.
        # Standard deviation to cover ~95% of the box in 1D is 10/4 = 2.5.
        # In d dimensions, a larger sigma might be safer for initial exploration.
        sigma = 2.5

        # Mean of the distribution
        # Start at the center of the box
        mean = np.zeros(d)
        
        # Covariance matrix
        C = np.eye(d)
        
        # Eigen decomposition of C for sampling and updating
        # C = B * D * B.T
        B, D = np.linalg.eigh(C)
        # Ensure D is positive
        D = np.abs(D)
        
        # Strategy parameters
        mu = lam // 2  # Number of parents
        weights = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        
        # Normalization constant
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        
        # Hessian update parameters
        h_sigma = 1.0
        ps = np.zeros(d)
        pc = np.zeros(d)
        
        # Track convergence
        # Stopping criteria:
        # 1. Budget exhausted
        # 2. Time limit
        # 3. Sigma is too small
        # 4. No improvement over last several iterations
        
        best_f = np.inf
        best_x = mean.copy()
        last_best_f = np.inf
        no_improve_count = 0
        max_no_improve = 20  # Stop if no improvement for 20 generations
        
        while state["count"] < budget and state["ok"]:
            # Check time
            if time.perf_counter() > deadline:
                break
                
            # Sample population
            # y = B * D^(1/2) * z
            # We need to generate lam points
            z = np.random.randn(lam, d)
            # Transform to x
            # x_i = mean + sigma * B * D^(1/2) * z_i
            # For efficiency, we can do:
            # y_i = D^(1/2) * z_i
            y = z * np.sqrt(D)[None, :]
            # x = mean + sigma * B @ y.T
            pop = mean[None, :] + sigma * (y @ B.T)
            
            # Evaluate population
            fitness = []
            for i in range(lam):
                if not state["ok"]:
                    break
                f_val = wf(pop[i])
                fitness.append(f_val)
                
            if not state["ok"]:
                break
                
            # Convert to array
            fitness = np.array(fitness)
            
            # Sort by fitness
            idx = np.argsort(fitness)
            pop_sorted = pop[idx]
            fitness_sorted = fitness[idx]
            
            # Update best
            if fitness_sorted[0] < best_f:
                best_f = fitness_sorted[0]
                best_x = pop_sorted[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Check stopping condition based on improvement
            if no_improve_count > max_no_improve:
                # If no improvement, we might be stuck. 
                # In Bent Cigar, this is unlikely unless sigma is too small or we are very close.
                # We can break if we've been stuck for a while.
                if sigma < 1e-8:
                    break
                # Otherwise, continue for a bit longer, but maybe reduce sigma slightly?
                # Standard CMA-ES handles this. Let's just break if budget is getting low.
                if state["count"] > budget * 0.8:
                    break

            # Selection
            parents = pop_sorted[:mu]
            
            # New mean
            new_mean = np.sum(weights[:, None] * parents, axis=0)
            
            # Calculate step direction in evolution path space
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B^T * (new_mean - mean)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B.T @ (new_mean - mean))
            
            # Update sigma
            # sigma *= exp(c_sigma / chi_d * (||ps|| - chi_d))
            sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(ps) - chi_d))
            
            # Update covariance matrix
            # pc = (1 - c_c) * pc + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (new_mean - mean)
            # h_sigma = (||ps|| / sqrt(1 - (1 - c_sigma)^{2*(gen+1)})) < (1 + 2/d) * chi_d
            # For simplicity, we use a standard update rule that approximates h_sigma
            # A common simplification is to check if the norm of ps is reasonable
            h_sigma = 1.0 if np.linalg.norm(ps) < (1 + 2/d) * chi_d else 0.0
            
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (new_mean - mean)
            
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - h_sigma) * c_c * (2 - c_c) * C) 
            #     + c_mu * sum(weights_i * y_i * y_i.T)
            
            # Compute the weighted outer products of the transformed parents (y_i)
            # y_i = B^-1 * (x_i - mean) / sigma ? No, x_i = mean + sigma * B * D^0.5 * z_i
            # So y_i in the C space is D^0.5 * z_i.
            # The update formula uses y_i such that x_i = mean + sigma * B * y_i
            # So y_i = D^0.5 * z_i
            
            # We need sum(weights_i * y_i * y_i.T)
            # y_sorted = z_sorted * sqrt(D)
            z_sorted = np.random.randn(mu, d) # This is wrong, we need the z corresponding to parents
            # We didn't store z_sorted. Let's recompute or store it.
            # It's better to store z during sampling.
            # Let's modify the sampling part to store z.
            
            # For this rewrite, let's use a simpler covariance update that is numerically stable 
            # and doesn't require storing z if we can recover y.
            # y_i = B.T @ (parents_i - mean) / sigma
            y_parents = (B.T @ (parents - mean).T).T / sigma
            
            # Compute the Hessian update term
            term_hessian = np.zeros((d, d))
            for i in range(mu):
                term_hessian += weights[i] * np.outer(y_parents[i], y_parents[i])
                
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * term_hessian
            
            # Ensure C is symmetric
            C = (C + C.T) / 2.0
            
            # Update eigen decomposition
            # This is the most expensive part, but for d=5 it's negligible.
            B, D = np.linalg.eigh(C)
            D = np.abs(D)
            
            # Prevent numerical issues
            if not np.all(np.isfinite(B)) or not np.all(np.isfinite(D)):
                break
                
            mean = new_mean
            
            # Safety check on sigma
            if sigma < 1e-15:
                break

        # Final update
        if best_x is not None:
            state["best_x"] = best_x
            state["best_f"] = best_f

        return state["best_x"], state["best_f"]
# EVOLVE-BLOCK-END
