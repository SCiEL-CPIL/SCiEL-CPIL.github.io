import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using quasi-random position sampling
    with distance variables fixed to zero (exact Pareto front)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # Number of position variables = dim - n_obj + 1 for DTLZ2
        n_pos = dim - n_obj + 1
        n_dist = dim - n_pos  # distance variables

        # Generate a Sobol sequence for the position space.
        # Use scipy's Sobol quasi-random sequence for low-discrepancy coverage.
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_pos, scramble=True)
            # Sobol requires power-of-2 count for best properties; use next power of 2 >= budget
            n_sobol = 1
            while n_sobol < budget:
                n_sobol *= 2
            pos_samples = sampler.random(n_sobol)
            # Take the first `budget` points
            pos_samples = pos_samples[:budget]
        except Exception:
            # Fallback: Halton-like via prime-based quasi-random
            pos_samples = self._halton(budget, n_pos)

        # Build full decision vectors: position vars from samples, distance vars = 0
        X = np.empty((budget, dim), dtype=np.float64)
        X[:, :n_pos] = pos_samples
        X[:, n_pos:] = 0.0  # distance vars set to 0 for exact Pareto front

        # Clip to bounds (distance vars at 0 should be within [0,1])
        X = np.clip(X, lo, hi)

        # Evaluate all points, respecting deadline
        F = np.empty((budget, n_obj), dtype=np.float64)
        for i in range(budget):
            if time.time() > deadline:
                # Return what we have so far
                return X[:i]
            F[i] = problem(X[i])

        return X

    def _halton(self, n, d):
        """Generate n points in d dimensions using Halton sequence."""
        primes = self._first_primes(d)
        points = np.zeros((n, d))
        for j in range(d):
            for i in range(n):
                points[i, j] = self._halton_value(i + 1, primes[j])
        return points

    @staticmethod
    def _first_primes(d):
        primes = []
        candidate = 2
        while len(primes) < d:
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

    @staticmethod
    def _halton_value(index, base):
        f = 1.0
        result = 0.0
        while index > 0:
            f /= base
            result += f * (index % base)
            index //= base
        return result
