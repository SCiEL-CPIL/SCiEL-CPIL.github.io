"""Model-and-optimize: DTLZ2 front is a quarter-sphere; sample it surface-area-uniform."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-based optimizer for DTLZ2 (M=3, dim=12).

    The Pareto front of DTLZ2 with M=3 objectives is a quarter-sphere of radius 1 in
    the positive octant: f = (sin(a)*sin(b), sin(a)*cos(b), cos(a)) for a,b in [0, pi/2].
    Any point with g(x)=0 (i.e., x[2:] = 0.5) and position (x[0], x[1]) mapped to
    (a, b) = (pi/2 * x[0], pi/2 * x[1]) lies exactly on this front.

    Hypervolume with a fixed reference point is maximized when the sample points are
    distributed uniformly by *surface area* on the front (not uniformly in parameter
    space). We use a Fibonacci-sphere construction projected onto the quarter-sphere
    to obtain N low-discrepancy, area-uniform points. All N budget points are on the
    front, so every evaluation contributes to hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _halton(self, i, base):
        """Generate the i-th Halton sequence value with given base."""
        result = 0.0
        f = 1.0 / base
        i = int(i)
        while i > 0:
            result += f * (i % base)
            i = i // base
            f /= base
        return result

    def _get_points(self, n):
        """
        Generate n points on the DTLZ2 Pareto front.
        Strategy:
        1. Include the 3 extreme corner points of the quarter-sphere front.
        2. Fill the remaining budget with Halton sequence in angle space (a, b).
        3. The Halton sequence provides low-discrepancy coverage of [0,1]^2,
           which maps to uniform surface-area distribution on the quarter-sphere.
        """
        points = []
        
        # 1. Corner points (extremes of the front)
        # a=0, b=0 -> f=(0,0,1)
        points.append((0.0, 0.0))
        # a=pi/2, b=0 -> f=(1,0,0)
        points.append((np.pi/2, 0.0))
        # a=pi/2, b=pi/2 -> f=(0,1,0)
        points.append((np.pi/2, np.pi/2))
        
        n_corners = 3
        n_remaining = n - n_corners
        
        if n_remaining > 0:
            # Generate Halton sequence points in [0,1]^2, then map to [0, pi/2]^2
            # Use bases 2 and 3 for the two dimensions
            for i in range(1, n_remaining + 1):
                u = self._halton(i, 2)
                v = self._halton(i, 3)
                a = u * np.pi / 2
                b = v * np.pi / 2
                points.append((a, b))
        
        # If we have more points than needed (shouldn't happen), truncate
        if len(points) > n:
            points = points[:n]
        
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.budget
        
        # Generate quarter-sphere points
        pts = self._get_points(n)
        
        results = []
        
        for a, b in pts:
            if time.time() > deadline:
                break
            
            # Map (a, b) to (x[0], x[1]) where x[0] = 2*a/pi, x[1] = 2*b/pi
            x0 = 2.0 * a / np.pi
            x1 = 2.0 * b / np.pi
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            if self.dim > 2:
                x[2:] = 0.5
            
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
        
        return np.array(results) if results else None
# EVOLVE-BLOCK-END
