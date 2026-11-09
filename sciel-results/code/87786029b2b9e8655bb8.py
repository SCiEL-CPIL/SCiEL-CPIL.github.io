"""Rewritten optimizer: deterministic global search with Halton sequence + multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic global search (Halton) + multi-start Nelder-Mead for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    @staticmethod
    def _halton_sequence(n, dim, skip=1):
        """Generate n points in [0,1)^dim using the Halton sequence."""
        points = np.zeros((n, dim))
        for d in range(dim):
            # Find (d+1)-th prime
            primes = []
            candidate = 2
            while len(primes) <= d:
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
            base = primes[d]
            for i in range(n):
                idx = i + skip
                f = 1.0
                r = 0.0
                while idx > 0:
                    idx, rem = divmod(idx, base)
                    f /= base
                    r += f * rem
                points[i, d] = r
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            calls_made += 1
            return val

        def check_time():
            return time.time() < deadline

        # Phase 1: Deterministic global search using Halton sequence
        # Use 60% of budget for global exploration
        global_budget = int(budget * 0.6)
        
        # Generate Halton points
        halton_pts = self._halton_sequence(global_budget, dim, skip=1)
        # Map from [0,1)^d to [-5,5]^d
        halton_pts = halton_pts * 10.0 - 5.0
        
        best_f = np.inf
        best_x = np.zeros(dim)
        
        for i in range(global_budget):
            if not check_time():
                break
            x = halton_pts[i]
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if not check_time():
            return best_x, best_f

        # Phase 2: Multi-start Nelder-Mead local refinement
        # Use remaining 40% of budget for local optimization
        from scipy.optimize import minimize
        
        # Create multiple starting points around the best found
        # Also include some perturbed versions of top candidates
        n_restarts = min(8, max(2, (budget - calls_made) // 50))
        
        # Generate starting points: best point + perturbations
        start_points = [best_x.copy()]
        
        # Add perturbations with decreasing step sizes
        for step in [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
            if len(start_points) >= n_restarts:
                break
            for d_i in range(dim):
                perturbed = best_x.copy()
                perturbed[d_i] += step
                if np.all(perturbed >= -5.0) and np.all(perturbed <= 5.0):
                    start_points.append(perturbed)
                    break
                perturbed = best_x.copy()
                perturbed[d_i] -= step
                if np.all(perturbed >= -5.0) and np.all(perturbed <= 5.0):
                    start_points.append(perturbed)
                    break
        
        # Ensure we have enough starting points
        while len(start_points) < n_restarts:
            perturbed = best_x + np.random.default_rng(42).normal(0, 0.1, dim)
            perturbed = np.clip(perturbed, -5.0, 5.0)
            start_points.append(perturbed)
        
        # Run Nelder-Mead from each starting point
        calls_per_restart = max(10, (budget - calls_made) // n_restarts)
        
        for sp in start_points:
            if not check_time():
                break
            if calls_made >= budget - 5:
                break
            
            try:
                result = minimize(
                    call_func,
                    sp,
                    method='Nelder-Mead',
                    options={
                        'maxiter': calls_per_restart,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Final polish with very tight tolerance if we have budget
        if check_time() and calls_made < budget - 10:
            try:
                result = minimize(
                    call_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': budget - calls_made - 5,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
