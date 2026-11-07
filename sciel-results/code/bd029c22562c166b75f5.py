"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # For dim=5, lambda=10 is a good default
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Strategy parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1)) + cs

        # Initial state
        # Start near center, but Schwefel minimum is shifted. 
        # A random start is safer, but we can try to be a bit more aggressive.
        x = rng.uniform(-2, 2, dim) 
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Pre-compute weights for sorting
        w = weights

        for gen in range(budget):
            # Check time
            if time.time() > deadline - 5:
                break

            # Sample points
            B = np.linalg.cholesky(C)
            z = rng.standard_normal((lam, dim))
            y = z @ B
            # Normalize y to have unit norm for the step size
            y_norm = np.linalg.norm(y, axis=1)
            y_norm[y_norm == 0] = 1.0
            y = y / y_norm[:, np.newaxis]
            
            pop = x + sigma * y
            pop = np.clip(pop, -5, 5)
            
            # Evaluate
            fvals = np.array([func(p) for p in pop])
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = pop[min_idx].copy()
                fx = best_f
                x = best_x.copy()

            # Sort
            idx = np.argsort(fvals)
            y = y[idx]
            fvals = fvals[idx]
            
            # Mean update
            y_w = np.sum(w[:, np.newaxis] * y[:mu], axis=0)
            x_old = x.copy()
            x = x + sigma * y_w
            
            # Covariance matrix adaptation
            ys = y[:mu]
            y_diff = ys - y_w[np.newaxis, :]
            
            # Update step size control
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (C ** -0.5) @ (y_w / np.linalg.norm(y_w)))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1 + 2 / dim) * np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
            
            # Update C
            C = (1 + c1 + cmu - c1) * C
            C += c1 * (np.outer(pc, pc) - (1 - cc) * C)
            C += cmu * np.sum(w[:, np.newaxis, np.newaxis] * np.transpose(ys, (1, 0, 2)) * ys[:, np.newaxis, :], axis=0)
            
            # Actually, standard update is:
            # C = (1 - c1 - cmu) * C + c1 * (pc*pc.T + (1-cc)C) + cmu * sum(wi * (yi-yw)(yi-yw).T)
            # Let's re-implement carefully to avoid bugs.
            
            # Re-do C update properly
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + (1 - cc) * np.eye(dim))
            rank_mu_term = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu_term += w[i] * np.outer(ys[i] - y_w, ys[i] - y_w)
            C += cmu * rank_mu_term

            # Update evolution path pc
            pc = (1 - cc) * pc + cc * (1 + c1 + cmu) * (y_w / np.linalg.norm(y_w) * np.linalg.norm(y_w)) 
            # Correction: pc update uses the step in x-space normalized by sigma? 
            # Standard CMA-ES: pc = (1-cc)pc + cc * (1 + c1 + cmu) * (y_w / ||y_w|| * ||y_w||) is wrong.
            # It is: pc = (1-cc)pc + cc * (1 + c1 + cmu) * (y_w / ||y_w||) * ||y_w|| ? No.
            # The step is y_w. The direction is y_w / ||y_w||.
            # The update is pc = (1-cc)pc + cc * (1 + c1 + cmu) * (y_w / ||y_w||) * ||y_w||? 
            # Actually, the term is just the mean step vector y_w, but scaled?
            # In CMA-ES, the step is sigma * y_w.
            # The evolution path pc is updated with the *normalized* step?
            # Standard formula: pc = (1-cc)pc + cc * (1 + c1 + cmu) * (y_w / ||y_w||) * ||y_w|| is redundant.
            # It is simply: pc = (1-cc)pc + cc * (1 + c1 + cmu) * (y_w / ||y_w||) * ||y_w||?
            # Let's check the reference: pc = (1-cc)pc + cc * (1 + c1 + cmu) * (y_w / ||y_w||) * ||y_w||
            # Wait, y_w is the weighted mean of the *normalized* directions? No, y is normalized, then scaled by sigma?
            # In my code: y is normalized. pop = x + sigma * y.
            # So the step is sigma * y_w.
            # The evolution path pc tracks the direction of the mean shift.
            # pc = (1-cc)pc + cc * (1 + c1 + cmu) * (y_w / ||y_w||) * ||y_w||
            # This simplifies to: pc = (1-cc)pc + cc * (1 + c1 + cmu) * y_w
            # But y_w is a vector of directions.
            # Let's stick to the standard implementation from Hansen's website.
            
            # Correct CMA-ES Update Logic:
            # 1. y are standard normal transformed by C^(1/2).
            # 2. y_w = sum w_i y_(i)
            # 3. x = x + sigma * y_w
            # 4. pc = (1-cc)pc + cc * (1 + c1 + cmu) * (y_w / ||y_w||) * ||y_w||  -> This is just cc*(1+c1+cmu)*y_w?
            #    Actually, the term is the *direction* of the mean shift.
            #    The step is y_w. The direction is y_w / ||y_w||.
            #    The update is pc = (1-cc)pc + cc * (1 + c1 + cmu) * (y_w / ||y_w||) * ||y_w||
            #    This is equivalent to pc = (1-cc)pc + cc * (1 + c1 + cmu) * y_w
            #    BUT, pc is a vector in the *search space* (transformed by B?).
            #    No, pc is in the *C-space*.
            #    Let's use the simplified version often used in code:
            
            # Recalculating pc and ps correctly
            # y_w is the weighted mean of the *normalized* vectors? No, y are the vectors.
            # y_w = sum w_i * y_i
            # The step is sigma * y_w.
            
            # Update ps
            # ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * (B @ (C^-0.5 @ (y_w / ||y_w||)))
            # Wait, y_w is the mean of the *normalized* y's? No, y's are already unit norm.
            # So y_w is a vector.
            # The term in ps is the direction of the mean shift in the *C-space*.
            # direction = y_w / ||y_w||
            # ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * (B @ (C^-0.5 @ direction))
            
            # Update pc
            # pc = (1-cc)pc + cc * (1+c1+cmu) * (B @ (C^-0.5 @ direction)) ?
            # No, pc is updated with the *same* direction vector.
            # pc = (1-cc)pc + cc * (1+c1+cmu) * (B @ (C^-0.5 @ direction))
            
            # Let's rewrite the loop body to be safe and correct.
            pass # This block is a mess, I will replace the whole method.
>>>>>>> REPLACE

The previous replacement attempt was messy. I will provide a clean, complete replacement for the `Optimizer` class using a well-tested CMA-ES implementation structure, ensuring correctness and efficiency.

