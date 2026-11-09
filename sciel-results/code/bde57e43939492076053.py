"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def halton(index, base):
    """Generate the Halton sequence value for a given index and base."""
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r

def halton_sequence(n, dim):
    """Generate n points in [0, 1)^dim using the Halton sequence."""
    points = np.zeros((n, dim))
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
    for i in range(1, n + 1):
        for d in range(dim):
            points[i - 1, d] = halton(i, bases[d])
    return points

class Optimizer:
    """Deterministic global search with Halton sequence + CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, budget, deadline, rng):
        """CMA-ES implementation for local refinement."""
        dim = self.dim
        x = x0.copy()
        fbest = func(x)
        calls = 1
        best_x = x.copy()
        
        # Initialize CMA-ES parameters
        sigma = 0.1
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        
        # Track state
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        C = np.eye(dim)
        invC = np.eye(dim)
        
        # Compute mu_eff
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Cholesky decomposition helper
        def cholesky_inv(A):
            try:
                L = np.linalg.cholesky(A)
                return np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if not positive definite
                return np.linalg.pinv(A)
        
        while calls < budget and time.time() < deadline:
            # Generate population
            pop = np.zeros((pop_size, dim))
            fvals = np.zeros(pop_size)
            
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                xi = x + sigma * (invC @ z)
                xi = np.clip(xi, -5, 5)
                pop[i] = xi
                fvals[i] = func(xi)
                calls += 1
                if fvals[i] < fbest:
                    fbest = fvals[i]
                    best_x = xi.copy()
            
            if calls >= budget or time.time() > deadline:
                break
                
            # Sort population
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Update best
            x_new = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update step size control variable
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (invC @ (x_new - x) / sigma)
            
            # Update covariance matrix
            hc = (1 - c_c) if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim + mu_eff - 2) else 0
            pc = (1 - c_c) * pc + hc * c_c * (ps + (c_sigma - c_c) * (x_new - x) / sigma)
            
            delta_hc = (1 - hc) * (1 - (1 - c_c)**2)
            C = (1 - c1 - cmu + delta_hc) * C + c1 * (np.outer(pc, pc) - C) + cmu * np.sum(weights[:, None, None] * np.outer(pop[:mu] - x, pop[:mu] - x), axis=0)
            
            c1 = (2 / ((dim + 1)**2 + 10))
            cmu = 2 * ((mu_eff + 2) / (dim + 2) - 1) * (c1 - c1 / 2)
            
            # Regularize C
            C = (C + C.T) / 2
            invC = cholesky_inv(C)
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
            
            x = x_new
            
            # Convergence check
            if np.ptp(fvals[:mu]) < 1e-10:
                break

        return best_x, fbest, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0
        rng = np.random.default_rng(12345)

        # Phase 1: Deterministic Global Search using Halton Sequence
        # Use 70% of budget for global exploration
        global_budget = int(budget * 0.7)
        
        # Generate Halton points
        n_points = global_budget
        halton_points = halton_sequence(n_points, dim)
        
        # Map from [0, 1)^dim to [-5, 5]^dim
        halton_points = halton_points * 10 - 5
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Evaluate Halton points
        for i in range(n_points):
            if calls >= budget or time.time() > deadline:
                break
            x = halton_points[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Phase 2: CMA-ES Local Refinement
        # Use remaining budget for local refinement
        remaining_budget = budget - calls
        if remaining_budget > 10 and time.time() < deadline:
            x_ref, f_ref, c_ref = self._cma_es(func, best_x, remaining_budget, deadline, rng)
            calls += c_ref
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref.copy()
                
        # Safety check: if best_x is still None (should not happen), return a default
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
