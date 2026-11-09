"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Sobol/Latin Hypercube global sampling followed by local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sequence(self, n, dim):
        """Generate n points in [0,1]^dim using a simple Sobol-like sequence."""
        # Use Halton sequence as a simpler alternative to Sobol for low dimensions
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                # Use prime bases for Halton sequence
                base = self._primes[j]
                points[i, j] = halton(i + 1, base)
        return points

    @staticmethod
    def _primes(n=10):
        """First n primes for Halton sequence bases."""
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin: 300s limit, stop at 280s

        # Phase 1: Global exploration with deterministic space-filling design
        # Use ~70% of budget for global exploration
        global_budget = int(budget * 0.7)
        
        # Generate Halton sequence points in [-5, 5]^dim
        points = self._sobol_sequence(global_budget, dim)
        points = points * 10.0 - 5.0  # Map to [-5, 5]
        
        best_x = None
        best_f = np.inf
        
        for i in range(global_budget):
            if time.time() > deadline:
                break
            x = points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback: single random point
            x = np.array([0.0] * dim)
            best_f = func(x)
            best_x = x
        
        # Phase 2: Local refinement around the best point found
        local_budget = budget - global_budget
        if local_budget > 0 and time.time() < deadline:
            # Simple local search: try small perturbations around best_x
            sigma = 0.5
            rng = np.random.default_rng(42)
            
            for i in range(local_budget):
                if time.time() > deadline:
                    break
                
                # Try a few perturbation directions
                improved = False
                for attempt in range(3):
                    direction = rng.standard_normal(dim)
                    direction /= np.linalg.norm(direction)
                    candidate = best_x + sigma * direction
                    candidate = np.clip(candidate, -5, 5)
                    cf = func(candidate)
                    if cf < best_f:
                        best_f = cf
                        best_x = candidate
                        sigma *= 1.2  # Expand step
                        improved = True
                        break
                
                if not improved:
                    sigma *= 0.7  # Shrink step
                    if sigma < 0.01:
                        sigma = 0.01
        
        return best_x, best_f
# EVOLVE-BLOCK-END
