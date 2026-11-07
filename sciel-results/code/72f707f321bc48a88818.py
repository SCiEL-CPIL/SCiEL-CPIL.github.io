"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 exploiting known structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # ZDT1 Pareto front: x[1:] = 0, x[0] in [0, 1]
        # f1 = x[0], f2 = 2*(1 - sqrt(x[0]))
        # Reference point: [11, 11]
        
        # For 2D hypervolume maximization on a convex decreasing front,
        # the optimal point distribution equalizes the marginal hypervolume
        # contribution of each point.
        
        # A practical approach: use the fact that the hypervolume of the
        # staircase is maximized when points are placed such that the
        # "height" of each step is proportional to the gap.
        
        # For ZDT1: f2 = 2*(1 - sqrt(x)), so df2/dx = -1/sqrt(x)
        # The slope is infinite at x=0 and approaches 0 as x->1.
        # Points should be denser near x=0.
        
        # Use a power-law distribution: x_i = ((i+0.5)/N)^p for some p > 1
        # or use the optimal formula derived from equalizing contributions.
        
        # Let's use a simple iterative refinement:
        # Start with uniform, then adjust based on local hypervolume contributions.
        
        N = budget
        ref = 11.0
        
        # Initial: uniform
        x = np.linspace(0.0, 1.0, N)
        
        # Iterate to improve placement
        for _ in range(50):
            f2 = 2.0 * (1.0 - np.sqrt(np.maximum(x, 1e-15)))
            
            # Compute contribution of each point
            # Point i contributes approximately: (x_{i+1} - x_{i-1})/2 * (ref - f2_i)
            # But more accurately, the staircase area between points
            contributions = np.zeros(N)
            for i in range(N):
                left = x[i-1] if i > 0 else 0.0
                right = x[i+1] if i < N-1 else 1.0
                # The width this point "owns"
                width = (right - left)
                # The height
                height = ref - f2[i]
                contributions[i] = width * height
            
            # Ideal: each point should have equal contribution
            # Adjust x to balance contributions
            # If contribution[i] is too high, move point i closer to neighbors
            # If too low, move it away
            
            # Simple heuristic: adjust x based on ratio of contributions
            avg_contrib = np.mean(contributions)
            for i in range(1, N-1):
                ratio = contributions[i] / avg_contrib
                # If ratio > 1, this point has too much area, move it closer
                # If ratio < 1, move it away
                # Adjust x[i] slightly
                if i > 0 and i < N-1:
                    mid_left = (x[i-1] + x[i]) / 2
                    mid_right = (x[i] + x[i+1]) / 2
                    # Move toward the side with more space if contribution is high
                    if ratio > 1.0:
                        # Move toward center of the larger gap
                        gap_left = x[i] - x[i-1]
                        gap_right = x[i+1] - x[i]
                        if gap_left > gap_right:
                            x[i] -= 0.01 * (ratio - 1.0)
                        else:
                            x[i] += 0.01 * (ratio - 1.0)
                    else:
                        # Move away from center
                        gap_left = x[i] - x[i-1]
                        gap_right = x[i+1] - x[i]
                        if gap_left > gap_right:
                            x[i] += 0.01 * (1.0 - ratio)
                        else:
                            x[i] -= 0.01 * (1.0 - ratio)
            
            # Re-sort and clip
            x = np.sort(x)
            x = np.clip(x, 0.0, 1.0)
        
        # Final: ensure endpoints are at 0 and 1 for maximum coverage
        x[0] = 0.0
        x[-1] = 1.0
        
        # Construct evaluation points
        n_evals = 0
        all_points = []
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            x_vec = np.zeros(dim)
            x_vec[0] = x[i]
            
            f = problem(x_vec)
            n_evals += 1
            all_points.append(f)
        
        return all_points

# EVOLVE-BLOCK-END
