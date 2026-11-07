"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and rotation-invariant search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Use a deterministic seed for reproducibility within an instance
        rng = np.random.default_rng(42)
        
        # Initial point: center of the box, but slightly randomized to avoid symmetry issues
        # Since the shift is hidden, starting at 0 is a reasonable heuristic as the shift is in [-4,4]
        x = np.zeros(n)
        fx = func(x)
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(n))  # Standard CMA-ES lambda
        mu = lam // 2  # Number of best solutions for recombination
        
        # Step size
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(n)
        sqrtC = np.eye(n)
        
        # Evolution path for sigma
        ps = np.zeros(n)
        # Evolution path for C
        pc = np.zeros(n)
        
        # Update rates
        cs = (mu + 2) / (n + mu + 5)
        c1 = 2 / ((n + 1.3)**2 + mu)
        cmu = min(1 - c1, 2 * (mu - 2 + 1/mu) / ((n + 2)**2 + mu))
        
        # Damping for ps and pc
        ds = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + cs
        dc = (1 + c1) * ds  # Approximation
        
        # Counter for covariance update
        counter = 0
        
        # History of best points
        best_x = x.copy()
        best_f = fx
        
        calls_made = 1
        
        while calls_made < budget:
            # Generate population
            # Ensure sigma doesn't explode or vanish
            if sigma < 1e-12:
                sigma = 1e-12
            if sigma > 100:
                sigma = 100
                
            Y = []
            for i in range(lam):
                # Sample from multivariate normal
                z = rng.multivariate_normal(np.zeros(n), C, 1)[0]
                y = x + sigma * z
                # Clip to box
                y = np.clip(y, -5, 5)
                Y.append(y)
                
            # Evaluate population
            F = []
            for y in Y:
                fy = func(y)
                F.append(fy)
                calls_made += 1
                if calls_made >= budget:
                    break
            
            # If we hit the budget limit mid-population, handle remaining
            # For simplicity in this constrained budget, we just proceed with what we have
            # But CMA-ES needs a full population. Let's ensure we don't exceed budget.
            # The loop condition `calls_made < budget` checks before generating the full population.
            # However, inside the loop, we might exceed it.
            # Let's refine: check if we have enough budget for the whole population.
            
            # Actually, the loop above breaks if budget is hit. 
            # If F has fewer than lam elements, we should probably stop or handle gracefully.
            # Given the strict budget, let's check before the population loop.
            
            # Re-evaluating logic: The while loop checks `calls_made < budget`.
            # If `budget - calls_made < lam`, we can't generate a full population.
            # We should either reduce lambda or stop.
            # Let's stop if we can't fit a full population, or use a smaller population.
            # For robustness, let's just break if we can't fit the population.
            # But we already started evaluating. Let's restructure.
            
            # Better approach: Check budget before generating population.
            # We are inside the while loop now. Let's assume we checked before.
            # Wait, the code above generated Y and evaluated.
            # If calls_made >= budget, we break out of the inner loop.
            # Then we proceed to selection. If len(F) < lam, we need to adjust.
            
            # Let's restart the logic to be safer.
            # I will rewrite the loop structure to check budget BEFORE generating the population.
            
            # ... (Self-correction: The code above is messy. Let's write a clean version.)
            pass # Placeholder, will rewrite below

    # Rewriting __call__ cleanly
    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_made = 1
        
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        sigma = 1.0
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        cs = (mu + 2) / (n + mu + 5)
        c1 = 2 / ((n + 1.3)**2 + mu)
        cmu = min(1 - c1, 2 * (mu - 2 + 1/mu) / ((n + 2)**2 + mu))
        ds = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + cs
        dc = (1 + c1) * ds
        
        while calls_made + lam <= budget:
            # Generate population
            Y = np.array([x + sigma * rng.multivariate_normal(np.zeros(n), C, 1)[0] for _ in range(lam)])
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            calls_made += lam
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = Y[idx_min].copy()
            
            # Sort indices
            idx_sorted = np.argsort(F)
            Y_sorted = Y[idx_sorted]
            F_sorted = F[idx_sorted]
            
            # Recombination
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= np.sum(weights)
            
            new_x = np.sum(weights[:, np.newaxis] * Y_sorted[:mu], axis=0)
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C @ (new_x - x)) / sigma
            mu_eff = np.sum(weights**2)
            
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - cs)**2):
                pc = (1 - dc) * pc + dc / sigma * (new_x - x)
                
            # Update covariance matrix
            delta = (np.linalg.norm(ps)**2 / (1 - (1 - cs)**2) - 1) * cs * (2 - cs)
            
            C = (1 - c1 - cmu + c1 * delta) * C
            C += c1 * np.outer(pc, pc)
            for i in range(mu):
                C += cmu * weights[i] * np.outer(Y_sorted[i] - x, Y_sorted[i] - x)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) - 1))
            
            # Update mean
            x = new_x
            
            # Handle potential numerical issues
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(sigma)):
                break
                
        # If we have remaining budget, use it for a local refinement or simple (1+1) steps
        # This helps if CMA-ES stops early due to budget constraints or convergence
        while calls_made < budget:
            # Simple local step
            step = sigma * rng.standard_normal(n)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                x = y
            else:
                sigma *= 0.9
            if sigma < 1e-10:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
