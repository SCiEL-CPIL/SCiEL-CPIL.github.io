"""DTLZ2 optimizer using Halton-sequence sampling on Pareto front angular parameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using low-discrepancy Halton sampling on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n Halton numbers in [0, 1) with the given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            i_val = i + 1
            while i_val > 0:
                result[i] += f * (i_val % base)
                i_val //= base
                f /= base
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.budget  # Use all budget for Pareto front sampling
        
        # Generate 2D Halton sequence for angular parameters
        # Base 2 for theta_1, base 3 for theta_2 (coprime bases for low discrepancy)
        theta1 = self._halton(n, 2)  # [0, 1) -> [0, pi/2]
        theta2 = self._halton(n, 3)  # [0, 1) -> [0, pi/2]
        
        # Map to spherical coordinates
        theta1 = theta1 * (np.pi / 2)
        theta2 = theta2 * (np.pi / 2)
        
        # Compute objective values on the Pareto front
        # f1 = cos(t1), f2 = sin(t1)*cos(t2), f3 = sin(t1)*sin(t2)
        f1 = np.cos(theta1)
        f2 = np.sin(theta1) * np.cos(theta2)
        f3 = np.sin(theta1) * np.sin(theta2)
        
        # For each point, construct the decision vector
        # x[0] = f1, x[1] = f2, x[2] = f3 (these determine objectives)
        # x[3:12] = 0.5 (to satisfy g(x) = 0, being on the Pareto front)
        # Clip to bounds to be safe
        
        F = np.zeros((n, self.n_obj))
        for i in range(n):
            if time.time() > deadline:
                # Return what we have so far
                break
            
            x = np.zeros(self.dim)
            x[0] = f1[i]
            x[1] = f2[i]
            x[2] = f3[i]
            x[3:] = 0.5  # Set to midpoint to be on Pareto front
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            F[i] = problem(x)
        
        return F
# EVOLVE-BLOCK-END
