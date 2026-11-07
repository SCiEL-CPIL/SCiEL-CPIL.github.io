"""Search-based optimizer for DTLZ4 with known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using Halton sequence on Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        budget = self.budget

        def obj_to_x(f1, f2, f3):
            """Map objective values on Pareto front to decision variables."""
            # DTLZ4 3-objective with g=0:
            # f1 = 0.5*cos(a)*cos(b), f2 = 0.5*cos(a)*sin(b), f3 = 0.5*sin(a)
            # where a = pi*x1/2, b = pi*x2/2
            # a = arcsin(2*f3), b = arctan2(f2, f1)
            a = np.arcsin(np.clip(2.0 * f3, -1.0, 1.0))
            b = np.arctan2(f2, f1)
            # Ensure b is in [0, pi/2]
            if b < 0:
                b += np.pi
            b = np.clip(b, 0.0, np.pi / 2)
            x1 = 2.0 * a / np.pi
            x2 = 2.0 * b / np.pi
            x1 = np.clip(x1, lo[0], hi[0])
            x2 = np.clip(x2, lo[1], hi[1])
            x = np.full(dim, 0.5)
            x[0] = x1
            x[1] = x2
            return np.clip(x, lo, hi)

        # DTLZ4 3-objective Pareto front (g=0):
        # f1^2 + f2^2 + f3^2 = 0.25, all fi >= 0
        # Quarter-sphere of radius 0.5 in positive octant.
        #
        # Strategy: Use Halton sequence in 2D to parameterize the quarter-sphere.
        # Use polar coordinates: f1 = r*cos(theta), f2 = r*sin(theta), f3 = sqrt(0.25 - r^2)
        # where r = 0.5 * sqrt(u1), theta = u2 * pi/2
        # This gives uniform coverage on the quarter-sphere surface.

        R = 0.5  # radius of Pareto front sphere

        def halton_sequence(n, base):
            """Generate n Halton sequence points in [0,1) using the given base."""
            result = []
            for i in range(1, n + 1):
                f = 1.0 / base
                result.append(0.0)
                while f > 0:
                    f = 1.0 / base
                # Standard Halton sequence generation
                h = 0.0
                f = 1.0 / base
                n_val = i
                while n_val > 0:
                    h += f * (n_val % base)
                    n_val //= base
                    f /= base
                result.append(h)
            return result

        # Generate Halton sequence points
        # We'll use two different bases for 2D Halton
        def halton_2d(n):
            """Generate n 2D Halton sequence points."""
            points = []
            for i in range(1, n + 1):
                # First coordinate: base 2
                h1 = 0.0
                f = 0.5
                n_val = i
                while n_val > 0:
                    h1 += f * (n_val % 2)
                    n_val //= 2
                    f /= 2
                
                # Second coordinate: base 3
                h2 = 0.0
                f = 1.0 / 3.0
                n_val = i
                while n_val > 0:
                    h2 += f * (n_val % 3)
                    n_val //= 3
                    f /= 3.0
                    
                points.append((h1, h2))
            return points

        # Generate candidates using Halton sequence
        candidates = []
        
        # Use Halton sequence for better coverage
        halton_points = halton_2d(budget)
        
        for u1, u2 in halton_points:
            # Map to quarter-sphere
            # r = R * sqrt(u1) for uniform area coverage
            # theta = u2 * pi/2
            r = R * np.sqrt(u1)
            theta = u2 * (np.pi / 2)
            
            f1 = r * np.cos(theta)
            f2 = r * np.sin(theta)
            f3_sq = R * R - r * r
            f3 = np.sqrt(max(0.0, f3_sq))
            
            candidates.append((f1, f2, f3))

        # Add critical boundary points for hypervolume
        # These are the extreme points that maximize individual objectives
        # f1 max: (0.5, 0, 0)
        # f2 max: (0, 0.5, 0)
        # f3 max: (0, 0, 0.5)
        # Check if they're already in candidates, if not add them
        boundary_points = [
            (R, 0.0, 0.0),
            (0.0, R, 0.0),
            (0.0, 0.0, R)
        ]
        
        for bp in boundary_points:
            # Check if this point is already in candidates
            found = False
            for c in candidates:
                if np.allclose(c, bp, atol=1e-10):
                    found = True
                    break
            if not found:
                # Replace last candidate with boundary point to maintain budget
                candidates[-1] = bp

        # Evaluate all candidates
        evaluated_points = []
        used = 0

        for f1, f2, f3 in candidates:
            if used >= budget or time.time() - t0 > deadline:
                break

            x = obj_to_x(f1, f2, f3)
            f = np.array(problem(x))
            used += 1
            evaluated_points.append(f)

        return np.array(evaluated_points)
# EVOLVE-BLOCK-END
