"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid lattice-sampling + CMA-ES for rotated/shifted Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 300s limit

        calls_made = 0
        best_x = None
        best_f = np.inf

        def do_call(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Evaluate integer lattice points within [-4, 4]^5
        # This exploits Rastrigin's periodic structure to find good baselines quickly
        lattice_vals = np.arange(-4, 5, 1)  # -4, -3, -2, -1, 0, 1, 2, 3, 4
        # In 5D that's 9^5 = 59049 points, too many. Use a subset.
        # Actually, let's be smarter: evaluate a coarser grid first, then refine.
        # For 500 calls, we need to be selective.
        # Let's evaluate a subset: all points where coordinates are in {-4, -2, 0, 2, 4}
        # That's 5^5 = 3125, still too many.
        # Let's use {-4, -3, -2, -1, 0, 1, 2, 3, 4} but only a random subset or structured.
        # Better: evaluate all points with coordinates in {-4, -2, 0, 2, 4} -> 5^5 = 3125, too many.
        # Use {-4, -1, 0, 1, 4} -> 5^5 = 3125, still too many.
        # Let's just do a random sample of lattice points, prioritizing center.
        
        # Actually, for a 500-call budget, let's do:
        # Phase 1: ~50 calls on a structured lattice subset
        # Phase 2: ~450 calls on CMA-ES refinement around the best lattice point
        
        # Structured lattice: use coordinates from {-4, -3, -2, -1, 0, 1, 2, 3, 4}
        # but sample only a fraction. Let's do a random subset of size ~50.
        rng = np.random.default_rng(42)
        
        # Generate all lattice points in [-4, 4]^5
        # 9^5 = 59049, too many to generate all at once efficiently for random sampling
        # Instead, sample random lattice points
        n_lattice = 50
        lattice_points = []
        for _ in range(n_lattice):
            pt = rng.choice(lattice_vals, size=self.dim)
            lattice_points.append(pt)
        
        # Also add the origin and some structured points
        structured = [
            np.zeros(self.dim),
            np.ones(self.dim) * 4,
            np.ones(self.dim) * -4,
            np.ones(self.dim) * 2,
            np.ones(self.dim) * -2,
            np.ones(self.dim) * 1,
            np.ones(self.dim) * -1,
        ]
        for pt in structured:
            lattice_points.append(pt.astype(float))
        
        # Deduplicate
        seen = set()
        unique_lattice = []
        for pt in lattice_points:
            key = tuple(pt)
            if key not in seen:
                seen.add(key)
                unique_lattice.append(pt)
        
        # Evaluate lattice points
        for pt in unique_lattice:
            if calls_made >= self.budget or time.time() > deadline:
                break
            do_call(pt)
        
        # Phase 2: CMA-ES refinement around the best point found so far
        if best_x is not None and calls_made < self.budget:
            # Run CMA-ES with remaining budget
            remaining_budget = self.budget - calls_made
            self._run_cma_es(func, best_x, best_f, remaining_budget, deadline)
        
        return best_x, best_f

    def _run_cma_es(self, func, x0, f0, budget, deadline):
        """Run CMA-ES starting from x0."""
        calls_made = 0
        best_x = x0.copy()
        best_f = f0
        dim = self.dim
        
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # ~10 for dim=5
        sigma = 1.0  # initial step size
        
        # CMA-ES state
        mean = x0.copy()
        C = np.eye(dim)
        Cs = np.zeros((dim, dim))
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # CMA-ES step sizes
        cs = (1 + 2 / (dim + 1)) / (dim + 1 / 2)
        ds = 1 + 2 * max(0, np.sqrt((dim + 1 / 2 - 2) / (dim + 1)) - 1) + cs
        cmu = min(1 / (dim + 3), 2 / ((dim + 2) ** 2 - 3))
        c1 = (2 + 2 / (dim + 1.8)) / ((dim + 2) ** 2 + 2)
        csig = (1 + 2 / (dim + 1) + cmu) / (dim + 4)
        
        # Precompute
        C_inv = np.linalg.inv(C)
        
        for gen in range(budget // pop_size + 1):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Generate population
            W = np.exp(0.5 * np.log(0.5) * (2 * np.arange(pop_size) - pop_size + 1) / pop_size)
            W /= W.sum()
            mu_eff = 1 / W[:int(0.5 * pop_size)] @ W[:int(0.5 * pop_size)]
            
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Sample from N(mean, sigma^2 * C)
                z = np.random.default_rng(hash(mean.tobytes() + str(gen) + str(i)) % 2**32).standard_normal(dim)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                
                y = sigma * (L @ z)
                x = np.clip(mean + y, -5, 5)
                
                fx = float(func(x))
                calls_made += 1
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Store for update
                if not hasattr(self, '_cma_pop'):
                    self._cma_pop = []
                self._cma_pop.append((fx, x, y))
            
            if len(self._cma_pop) < pop_size:
                break
            
            # Sort by fitness
            self._cma_pop.sort(key=lambda t: t[0])
            selected = self._cma_pop[:int(0.5 * pop_size)]
            
            # Update mean
            mean_old = mean.copy()
            mean = sum(W[i] * x for i, (f, x, y) in enumerate(selected))
            mean = np.clip(mean, -5, 5)
            
            # Update covariance matrix
            y_w = sum(W[i] * y for i, (f, x, y) in enumerate(selected))
            
            # Success rule for sigma
            if np.linalg.norm(y_w) < 1.4 * (1 + 1 / dim) * sigma:
                sigma *= np.exp(cs / ds * (np.linalg.norm(y_w) / sigma - 1))
            sigma = np.clip(sigma, 1e-10, 50)
            
            # Update Cs and C
            Cs = (1 - cs) * Cs + cs * (y_w / sigma - Cs)
            C = (1 - c1 - cmu) * C + c1 * (y_w @ y_w.T + (cs * (2 - cs)) * Cs)
            
            # Regularize C
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            del self._cma_pop
        
        # Return best found
        return best_x, best_f
# EVOLVE-BLOCK-END
