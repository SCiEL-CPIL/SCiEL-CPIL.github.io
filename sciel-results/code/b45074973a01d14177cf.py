"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES with adaptive restarts for BBOB-style f6 (Attractive sector).
    
    Strategy:
    - Use CMA-ES (a robust black-box optimizer) as the core local search.
    - Multiple restarts with different initializations to handle the
      deceptive landscape of f6.
    - LHS for initial point selection to ensure good coverage.
    - Deadline guard to stay within the 300s limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        def remaining():
            return max(0, budget - calls_made[0])

        def time_left():
            return max(0.0, deadline - time.time())

        calls_made = [0]

        def eval_point(x):
            """Evaluate a point, tracking call count and deadline."""
            nonlocal best_x, best_f
            if time.time() > deadline:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_made[0] += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: LHS exploration to find a good starting point ---
        lhs_count = min(60, budget // 10)
        lhs_points = self._lhs(lhs_count, dim, rng)
        for i in range(lhs_count):
            if time.time() > deadline:
                break
            fx = eval_point(lhs_points[i])
            if fx is None:
                break

        if best_x is None:
            # Fallback: random start
            x0 = rng.uniform(-5, 5, dim)
            eval_point(x0)
            if best_x is None:
                return np.zeros(dim), np.inf

        # --- Phase 2: CMA-ES with restarts ---
        # We'll run CMA-ES from the best point found, with restarts if it stagnates.
        
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # population size
        sigma = 1.0  # initial step size
        
        # Run CMA-ES
        x = best_x.copy()
        fx = best_f
        
        # CMA-ES state
        C = np.eye(dim)  # covariance matrix
        step_size = sigma
        c1 = (2 / ((dim + 1) ** 2 + 10)) * 0.5
        cmu = min(0.5 - c1, 2 / ((dim + 2) ** 2 + 5))
        cs = (dim + 1.3) / (dim + 5 + 2 * dim / (dim + 1))
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        hsig = 1.0
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        damps = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
        
        # Bounded CMA-ES
        max_iter = budget - calls_made[0]
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            if calls_made[0] >= budget:
                break
            
            # Generate population
            pop = []
            pop_f = []
            
            for i in range(pop_size):
                if time.time() > deadline or calls_made[0] >= budget:
                    break
                # Sample from N(x, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Whitening: C @ z
                y = x + step_size * (C @ z)
                y = np.clip(y, -5.0, 5.0)
                
                fy = eval_point(y)
                if fy is None:
                    break
                pop.append(y)
                pop_f.append(fy)
            
            if len(pop) < 2:
                break
            
            # Sort by fitness
            indices = np.argsort(pop_f)
            pop = [pop[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]
            
            # Update best
            if pop_f[0] < fx:
                fx = pop_f[0]
                x = pop[0].copy()
            
            # Update CMA-ES parameters
            lam = len(pop)
            mueff = max(1, int(0.5 * lam))
            
            # Mean update
            weights = np.zeros(lam)
            weights[:mueff] = np.log(mueff + 0.5) - np.log(np.arange(1, mueff + 1))
            weights /= weights.sum()
            new_mean = np.zeros(dim)
            for i in range(mueff):
                new_mean += weights[i] * pop[i]
            
            # Step size adaptation
            z = (new_mean - x) / (step_size * np.sqrt(np.trace(C)) + 1e-16)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1))) / chiN < (1.4 + 2 / (dim + 1)) * (1 - cs)
            pc = (1 - cs) * pc + hsig * cs * (new_mean - x) / step_size
            step_size *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
            step_size = np.clip(step_size, 1e-12, 10.0)
            
            # Covariance update
            diff = np.array([(pop[i] - x) / step_size for i in range(mueff)])
            C = (1 - mu - 2 * cmu) * C + cmu * (np.outer(pc, pc) + diff.T @ (weights * diff))
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Add small regularization for numerical stability
            C += 1e-12 * np.eye(dim)
            
            x = new_mean
            x = np.clip(x, -5.0, 5.0)
            
            # Check for stagnation
            if iteration > 50 and step_size < 1e-6:
                break
        
        # --- Phase 3: Polish with very small steps if time remains ---
        polish_budget = min(30, budget - calls_made[0])
        sigma_polish = 0.005
        for _ in range(polish_budget):
            if time.time() > deadline or calls_made[0] >= budget:
                break
            y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5.0, 5.0)
            fy = eval_point(y)
            if fy is None:
                break
            if fy < fx:
                x, fx = y, fy
                sigma_polish *= 0.9
            else:
                sigma_polish *= 0.7
            if sigma_polish < 1e-7:
                break

        return best_x, best_f

    @staticmethod
    def _lhs(n, dim, rng):
        """Latin Hypercube Sampling: n points in [-5, 5]^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            points[:, d] = (perm + 0.5) / n * 10.0 - 5.0
        return points
# EVOLVE-BLOCK-END
