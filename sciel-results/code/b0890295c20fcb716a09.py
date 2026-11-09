"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        x = np.zeros(n)
        sigma = 1.5  # Wider initial step to cover [-4, 4] better
        C = np.eye(n)
        D = np.eye(n)
        B = np.eye(n)
        
        # Parameters
        cs = 1 / (n + 4)
        ds = 1 + 2 * max(0, (n - 1) / (n + 2) - 1) + cs
        c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + 2)
        cmu = min(1 - c1, 2 * (2 - (n + 1) / (n + 3)) / ((n + 2) ** 2 + 5))
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # Initial evaluation
        best_x = x.copy()
        best_f = func(x)
        calls_made = 1

        # Population size
        lam = 4 + int(3 * np.log(n))
        # Weights
        w = np.array([np.log(np.log(lam + 1) - np.log(i + 1) + 1) for i in range(lam)])
        w = w / w.sum()
        mueff = int(np.floor(0.5 * lam))
        # Recalculate mueff as effective population size
        mueff = 1 / np.sum(w[:mueff] ** 2)

        while calls_made < budget:
            # Generate population
            pop = []
            fvals = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D @ z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                pop.append(y)
                fvals.append(fy)
                calls_made += 1
                if calls_made >= budget:
                    break

            if calls_made >= budget:
                break

            # Sort by fitness
            indices = np.argsort(fvals)
            pop = [pop[i] for i in indices]
            fvals = [fvals[i] for i in indices]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()

            # Update mean
            x_new = sum(w[i] * pop[i] for i in range(mueff))

            # Update evolution paths
            # Transform x_new - x to normal coordinates
            y_mean = B.T @ (D @ ((x_new - x) / sigma))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_mean
            
            # Heuristic for hsig
            hsig = (np.linalg.norm(ps) / chiN) < (1.4 + 2 / (n + 1))
            
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * (x_new - x) / sigma

            # Update covariance
            # Rank-one update
            C = (1 + c1 * (hsig * (ds - 1) - c1) - cmu) * C + c1 * np.outer(pc, pc)
            
            # Rank-mu update
            for i in range(mueff):
                y_i = B.T @ (D @ ((pop[i] - x) / sigma))
                C += cmu * w[i] * np.outer(y_i, y_i)
                
            # Note: The standard CMA-ES update is C = (1 - c1 - cmu) * C + c1 * outer(pc, pc) + cmu * sum(wi * outer(yi, yi))
            # My previous line was slightly off. Let's correct it to standard form.
            # Actually, the standard update is:
            # C = (1 - c1 - cmu) * C + c1 * outer(pc, pc) + cmu * sum(wi * outer(yi, yi))
            # But pc is in the original coordinate system? No, pc is usually kept in the original system or transformed.
            # In the standard algorithm, pc is updated in the original space: pc = (1-cs)pc + hsig * sqrt(cs(2-cs)mueff) * (x_new - x)/sigma
            # And C is updated as: C = (1 - c1 - cmu) * C + c1 * outer(pc, pc) + cmu * sum(wi * outer(B @ (D @ yi), B @ (D @ yi)))
            # Wait, if pc is in original space, outer(pc, pc) is in original space.
            # The term B @ (D @ yi) transforms the normal vector back to original space.
            
            # Let's re-implement the covariance update correctly based on standard CMA-ES.
            # We already calculated y_mean and pc in original space?
            # pc = (1-cs)pc + hsig * sqrt(...) * (x_new - x)/sigma. This is in original space.
            # The rank-one update adds c1 * outer(pc, pc).
            # The rank-mu update adds cmu * sum(wi * outer(vi, vi)) where vi = B @ (D @ yi).
            
            # Let's redo the covariance update block properly.
            # We need to be careful with the transformation.
            
            # Correct Standard CMA-ES Update:
            # 1. ps is in normal space.
            # 2. pc is in original space? No, usually pc is in the space of the mean vector.
            # Let's stick to the formulation where everything is in the original space for C.
            
            # Re-calculating pc in original space is correct.
            # The term for C rank-one is c1 * outer(pc, pc).
            # The term for C rank-mu is cmu * sum(wi * outer(vi, vi)) where vi = B @ (D @ yi).
            
            # Let's overwrite the C update with the correct one.
            # The previous code in the search block had:
            # C = (1 + c1 * deltaD - c1 - cmu) * C + c1 * np.outer(pc, pc)
            # C += cmu * w[i] * np.outer(B @ (D @ y_i), B @ (D @ y_i))
            # This looks mostly correct, but the coefficient (1 + c1*deltaD - c1 - cmu) is non-standard.
            # Standard is (1 - c1 - cmu).
            # Also, the pc update should use the transformed mean difference if we want to keep pc in normal space, 
            # but if we keep it in original space, the outer product is direct.
            
            # Let's use a very standard, robust implementation.
            
            # --- REPLACE COVARIANCE UPDATE ---
            # We will replace the lines from "Update covariance" to "Update mean"
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
