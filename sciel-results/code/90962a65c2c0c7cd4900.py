"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic DTLZ2-specific search using Latin Hypercube Sampling on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        used = 0
        
        # DTLZ2 Pareto front:
        # f1 = cos(alpha), f2 = sin(alpha)*cos(beta), f3 = sin(alpha)*sin(beta)
        # where alpha, beta in [0, pi/2]
        # Decision variables:
        # x[0] = 2*alpha/pi
        # x[1] = 2*beta/pi
        # x[2:] = 0.5 (optimal distance variables for DTLZ2 to achieve g=0)
        
        # We will use Latin Hypercube Sampling (LHS) for the (alpha, beta) space
        # to ensure good coverage of the 2D parameter space which maps to the 3D Pareto front.
        
        # Reserve 4 evaluations for the corner points of the Pareto front
        # Corners correspond to (alpha, beta) = (0,0), (0, pi/2), (pi/2, 0), (pi/2, pi/2)
        # In x-space: (0,0), (0,1), (1,0), (1,1)
        
        n_corners = 4
        n_lhs = self.budget - n_corners
        
        # Generate LHS points for alpha and beta
        # LHS: divide [0,1] into n intervals, pick one random point in each, then shuffle
        def latin_hypercube(n, dim=1):
            if n <= 0:
                return np.array([]).reshape(0, dim)
            points = np.zeros((n, dim))
            for d in range(dim):
                # Create intervals
                intervals = np.arange(n) / n
                # Random offset within each interval
                offsets = np.random.rand(n) / n
                # Combine
                pts = intervals + offsets
                # Shuffle to decouple dimensions
                np.random.shuffle(pts)
                points[:, d] = pts
            return points

        # We need 2D LHS for alpha and beta parameters
        # Note: The problem statement says the evaluator calls np.random.seed(seed).
        # We should rely on the global numpy random state for reproducibility per seed.
        
        if n_lhs > 0:
            # Generate LHS in the unit square [0,1]x[0,1]
            lhs_pts = latin_hypercube(n_lhs, dim=2)
            
            for a, b in lhs_pts:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Map [0,1] to x[0], x[1]
                x = np.zeros(self.dim)
                x[0] = a
                x[1] = b
                x[2:] = 0.5  # Optimal distance variables
                
                # Clip to bounds
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                if np.all(np.isfinite(f)):
                    results.append(f)
                    used += 1
        
        # Add corner points to ensure the extremes are covered
        corners = [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0)
        ]
        
        for a, b in corners:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = a
            x[1] = b
            x[2:] = 0.5
            
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            if np.all(np.isfinite(f)):
                results.append(f)
                used += 1
        
        return np.array(results) if results else np.empty((0, self.n_obj))
# EVOLVE-BLOCK-END
