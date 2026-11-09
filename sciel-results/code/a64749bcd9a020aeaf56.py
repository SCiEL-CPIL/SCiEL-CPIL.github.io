"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts for BBOB f6 (Attractive Sector)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285.0  # safety margin before the 300s limit
        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def call(x):
            nonlocal best_x, best_f, call_count
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Coarse LHS sampling to identify promising basins ---
        lhs_count = min(60, budget // 8)
        lhs_points = self._lhs(lhs_count, dim, rng)
        for i in range(lhs_count):
            if time.time() > deadline or call_count >= budget:
                break
            call(lhs_points[i])

        if best_x is None:
            x0 = np.zeros(dim)
            call(x0)
            best_x = x0.copy()
            best_f = np.inf

        # --- Phase 2: Multi-start CMA-ES with adaptive restarts ---
        # We use CMA-ES which is more robust to the non-smooth attractive sector
        # than simple (1+1)-ES. We do multiple restarts from different starting
        # points to escape local minima.
        
        remaining_budget = budget - call_count
        n_restarts = 3
        budget_per_restart = max(10, remaining_budget // n_restarts)
        
        # Collect candidate starting points: best from LHS + a few random ones
        start_points = [best_x.copy()]
        for _ in range(n_restarts - 1):
            sp = np.clip(rng.uniform(-5, 5, dim), -5.0, 5.0)
            start_points.append(sp)

        for restart_idx in range(n_restarts):
            if call_count >= budget or time.time() > deadline:
                break
            
            x_start = start_points[restart_idx] if restart_idx < len(start_points) else best_x.copy()
            x, fx = self._cma_es(func, call, x_start, budget_per_restart, deadline, rng, dim)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # --- Phase 3: Final polish with very small CMA-ES or simple ES ---
        if call_count < budget and time.time() < deadline:
            polish_budget = min(30, budget - call_count)
            x, fx = self._cma_es(func, call, best_x.copy(), polish_budget, deadline, rng, dim, sigma_init=0.05)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f

    def _cma_es(self, func, call, x0, budget, deadline, rng, dim, sigma_init=1.0):
        """Simple CMA-ES implementation with population size based on budget."""
        x = x0.copy()
        fx = call(x)
        sigma = sigma_init
        pop_size = max(4, min(20, int(budget // 5)))
        
        # CMA-ES parameters
        cc = (4 + dim / (dim + 4)) / ((dim + 4) + 4 + dim / (dim + 4))
        cmu = min(1, (2 + 2. / (dim + 1.)) / (dim + 2. + 2 + 2. / (dim + 1.)))
        cs = (dim + 1.5 + cmu) / (dim + 5.5 + cmu)
        ds = 1 + 2 * max(0, sqrt(cc - 0.5)) + cs
        
        ps = np.zeros(dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        sqrtC = np.eye(dim)
        invsqrtC = np.eye(dim)
        
        steps = 0
        no_improve = 0
        
        while call_count_global < budget and time.time() < deadline:
            # Generate population
            Y = np.zeros((pop_size, dim))
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                Y[i] = x + sigma * (sqrtC @ z)
            
            # Evaluate
            F = np.array([call(Y[i]) for i in range(pop_size)])
            steps += pop_size
            
            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            m = pop_size // 2  # number of selected
            
            # Update mean
            w = np.log(m + 0.5) - np.log(np.arange(1, m + 1))
            w = w / w.sum()
            x_new = np.sum(w[:, None] * Y[:m], axis=0)
            
            # Update evolution path
            ps = (1 - cs) * ps + cs * sqrt(mu_eff) * invsqrtC @ (x_new - x) / sigma
            chi_n = np.sqrt(dim) * (1 - 1. / (2 * dim * dim) + 1. / (12 * dim * dim * dim))
            if np.linalg.norm(ps) / sqrt(1 - (1 - cs) ** (2 * (steps + 1) / dim)) < (1.4 + 2. / (dim + 1)) * chi_n:
                C = (1 - cmu) * C + (cmu / (1 - (1 - cs) ** 2)) * np.outer(ps, ps)
            
            # Update covariance
            C = (1 - cmu) * C + np.sum(w[i] * np.outer(Y[i] - x, Y[i] - x) for i in range(m))
            
            # Regularize
            C = (C + C.T) / 2
            
            # Update square root and inverse
            try:
                sqrtC, invsqrtC = self._update_sqrt(C, sqrtC, invsqrtC)
            except:
                break
            
            x = x_new
            fx = F[0]
            
            # Adaptive sigma
            if np.linalg.norm(ps) / chi_n > 1.0:
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            
            if F[0] < fx - 1e-10:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 10:
                    break
        
        return x, fx

    @staticmethod
    def _lhs(n, dim, rng):
        """Latin Hypercube Sampling: n points in [-5, 5]^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            points[:, d] = (perm + 0.5) / n * 10.0 - 5.0
        return points

    @staticmethod
    def _update_sqrt(C, sqrtC_old, invsqrtC_old):
        """Update square root decomposition of covariance matrix."""
        # Use eigendecomposition for stability
        evals, evecs = np.linalg.eigh(C)
        evals = np.maximum(evals, 1e-10)
        sqrtC = evecs @ np.diag(np.sqrt(evals))
        invsqrtC = evecs @ np.diag(1.0 / np.sqrt(evals))
        return sqrtC, invsqrtC
# EVOLVE-BLOCK-END
