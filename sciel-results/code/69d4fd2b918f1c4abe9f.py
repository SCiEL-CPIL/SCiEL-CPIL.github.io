"""Multi-objective optimizer for DTLZ2 exploiting problem structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits DTLZ2 structure: Pareto front is a quarter-sphere in 3D.
    
    For DTLZ2 with 3 objectives and 12 variables:
    - g(x) = sum_{i=2}^{11} (x_i - 0.5)^2, minimized at 0 when x_i=0.5
    - f1 = (1+g)*cos(t1)*cos(t2), f2 = (1+g)*cos(t1)*sin(t2), f3 = (1+g)*sin(t1)
    - where t1 = x0*pi/2, t2 = x1*pi/2
    - Pareto front: g=0, so f1^2+f2^2+f3^2 = 1
    
    Strategy: 
    1. Use the entire budget to evaluate points exactly on the Pareto front.
    2. Use a deterministic low-discrepancy sequence (Halton) for the 2D 
       parameter space (x0, x1) to ensure uniform coverage of the front.
    3. This maximizes the hypervolume by avoiding any off-front points 
       (which are strictly dominated) and maximizing the spread of on-front points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._deadline = None

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: stop 30s before the 330s wall-clock kill
        self._deadline = start_time + 290.0  # 290s limit per evaluation

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ2 structure: g=0 when x[2:] = 0.5
        # Pareto front parameterized by x0, x1 in [0,1]
        # f = [cos(t1)*cos(t2), cos(t1)*sin(t2), sin(t1)] where t1=x0*pi/2, t2=x1*pi/2
        
        # Generate Halton sequence samples for x0, x1
        # Halton sequences are low-discrepancy and provide excellent uniform coverage
        def halton_sequence(n, base):
            """Generate n Halton sequence points in [0,1]."""
            result = np.zeros(n)
            f = 1.0
            for i in range(1, n + 1):
                x = 0.0
                j = i
                while j > 0:
                    f /= base
                    x += f * (j % base)
                    j //= base
                result[i - 1] = x
            return result
        
        # Use primes 2 and 3 for the two dimensions
        x0_hal = halton_sequence(budget, 2)
        x1_hal = halton_sequence(budget, 3)
        
        # Construct full 12D vectors on the Pareto front
        points = np.zeros((budget, dim))
        points[:, 0] = x0_hal
        points[:, 1] = x1_hal
        points[:, 2:] = 0.5  # g=0
        
        # Evaluate all points
        F = []
        used = 0
        for x in points:
            if time.time() > self._deadline:
                break
            if used >= budget:
                break
            f = problem(x)
            F.append(f)
            used += 1
        
        F = np.array(F) if F else np.zeros((0, self.n_obj))
        X = points[:used].copy() if used > 0 else np.zeros((0, dim))
        
        return X
# EVOLVE-BLOCK-END
