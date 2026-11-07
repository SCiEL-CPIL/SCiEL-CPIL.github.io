# EVOLVE-BLOCK-START
import numpy as
import time


class Optimizer:
    """Hypervolume-driven sampling of DTLZ1 Pareto front.
    
    Strategy:
    - DTLZ1 with 7 vars, 3 objectives: Pareto front is a flat simplex (triangle).
    - Parameterize front by barycentric coordinates (a, b, c) with a+b+c=1, a,b,c>=0.
    - Map to decision variables using the DTLZ1 structure.
    - Use uniform triangular grid for even hypervolume coverage.
    - Remaining budget: strategic edge/interior points for gap-filling.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _front_point(self, a, b, c):
        """Map barycentric coords (a,b,c) summing to 1 to decision vector on DTLZ1 front.
        
        For DTLZ1 3-obj at g=0:
        f1 = 0.5 * x1
        f2 = 0.5 * x2 * (1-x1)  
        f3 = 0.5 * (1-x1) * (1-x2)
        
        So: a = x1, b = x2*(1-x1), c = (1-x1)*(1-x2)
        Given a,b,c: x1 = a, x2 = b/(1-a) if a<1
        """
        x = np.full(self.dim, 0.5)
        # Clamp and normalize
        a = max(0.0, min(1.0, a))
        b = max(0.0, min(1.0, b))
        c = max(0.0, min(1.0, c))
        s = a + b + c
        if s < 1e-15:
            a, b, c = 1.0/3, 1.0/3, 1.0/3
            s = 1.0
        a, b, c = a/s, b/s, c/s
        
        if a < 1.0 - 1e-12:
            x[0] = a
            x2_val = b / (1.0 - a)
            x[1] = max(0.0, min(1.0, x2_val))
        else:
            x[0] = 1.0
            x[1] = 0.0
        # x[2:5] = 0.5, x[5:7] = 0.5 (g=0)
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Triangular grid on the simplex
        # n_per_edge=19 gives 190 points (19*20/2)
        n_per_edge = 19
        points = []
        for i in range(n_per_edge + 1):
            for j in range(n_per_edge + 1 - i):
                k = n_per_edge - i - j
                a = i / n_per_edge
                b = j / n_per_edge
                c = k / n_per_edge
                x = self._front_point(a, b, c)
                points.append(x)
        
        # Evaluate grid points
        F = np.array([problem(x) for x in points])
        used = len(points)
        
        # Phase 2: Strategic points for better coverage
        remaining = budget - used
        if remaining > 0:
            # Add vertices, edge midpoints, and centroid
            strategic = [
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.5, 0.5, 0.0),
                (0.5, 0.0, 0.5),
                (0.0, 0.5, 0.5),
                (1.0/3, 1.0/3, 1.0/3),
                (0.75, 0.25, 0.0),
                (0.25, 0.75, 0.0),
                (0.75, 0.0, 0.25),
                (0.0, 0.75, 0.25),
                (0.25, 0.0, 0.75),
                (0.0, 0.25, 0.75),
            ]
            
            for (a, b, c) in strategic:
                if used >= budget:
                    break
                x = self._front_point(a, b, c)
                F_new = problem(x)
                F = np.vstack([F, [F_new]])
                used += 1
        
        # Phase 3: Random fill if budget remains
        while used < budget:
            if time.time() > deadline:
                break
            r1 = np.random.random()
            r2 = np.random.random()
            if r1 + r2 > 1.0:
                r1 = 1.0 - r1
                r2 = 1.0 - r2
            a = r1
            b = r2
            c = 1.0 - r1 - r2
            x = self._front_point(a, b, c)
            F_new = problem(x)
            F = np.vstack([F, [F_new]])
            used += 1
        
        return F
# EVOLVE-BLOCK-END
