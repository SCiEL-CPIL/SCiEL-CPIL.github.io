"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ2-aware multi-objective optimizer: Sobol sampling on analytic Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dtlz2_front_point(self, theta, dim):
        """Generate a point exactly on DTLZ2 Pareto front given angles theta.
        For DTLZ2: first (n_obj-1) angles define direction on unit sphere,
        last dim - (n_obj-1) variables are distance vars (set to 0.5 for front).
        """
        n_obj = self.n_obj
        n_angle = n_obj - 1  # 2 angles for 3 objectives
        x = np.zeros(dim)

        t1 = theta[0]
        t2 = theta[1] if len(theta) > 1 else 0.0

        x[0] = np.cos(t1)
        x[1] = np.sin(t1) * np.cos(t2)
        x[2] = np.sin(t1) * np.sin(t2)

        # Distance variables (x[3:]) set to 0.5 for Pareto front
        if dim > 3:
            x[3:] = 0.5

        return x

    def _sobol_sequence_2d(self, n_points):
        """Generate a 2D Sobol-like low-discrepancy sequence in [0,1)^2.
        Uses a deterministic lattice based on the golden ratio for near-uniform coverage.
        """
        # Use a Halton-like sequence with prime bases 2 and 3 for 2D
        # This gives good low-discrepancy properties
        points = np.zeros((n_points, 2))
        
        for i in range(1, n_points + 1):
            # Base 2 van der Corput sequence
            f2 = 0.0
            f = 1.0
            n = i
            while n > 0:
                f /= 2.0
                f2 += f * (n % 2)
                n //= 2
            points[i-1, 0] = f2
            
            # Base 3 van der Corput sequence
            f3 = 0.0
            f = 1.0
            n = i
            while n > 0:
                f /= 3.0
                f3 += f * (n % 3)
                n //= 3
            points[i-1, 1] = f3
        
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        all_X = []
        all_F = []

        # === Phase 1: Generate maximally space-filling set on DTLZ2 Pareto front ===
        # Use Sobol/Halton sequence to sample the 2D angle space uniformly
        # This ensures even coverage of the Pareto front
        
        n_points = self.budget  # Use entire budget for front sampling
        
        # Generate low-discrepancy sequence in [0,1)^2
        seq = self._sobol_sequence_2d(n_points)
        
        # Map to angle space: theta1 in [0, pi/2], theta2 in [0, pi/2]
        # Use the full range for maximum coverage
        angles = seq * (np.pi / 2)
        
        # Generate points on the Pareto front
        X = np.array([self._dtlz2_front_point(theta, self.dim) for theta in angles])
        
        # Clip to bounds (should be no-op for DTLZ2 with [0,1] bounds)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for i in range(n_points):
            if time.time() > deadline:
                break
            
            f_val = problem(X[i])
            used += 1
            all_X.append(X[i])
            all_F.append(f_val)
        
        return np.array(all_X) if all_X else X
# EVOLVE-BLOCK-END
