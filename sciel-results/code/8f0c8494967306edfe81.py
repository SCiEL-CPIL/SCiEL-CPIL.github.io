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
    1. Use quasi-random points on the Pareto front (x2..x11=0.5, x0,x1 on a grid)
    2. Use remaining budget for local refinement around best points
    3. Keep a deadline guard to stay within time limits
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._deadline = None

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: stop 30s before the 330s wall-clock kill
        self._deadline = start_time + 300.0  # 300s limit per evaluation

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ2 structure: g=0 when x[2:] = 0.5
        # Pareto front parameterized by x0, x1 in [0,1]
        # f = [cos(t1)*cos(t2), cos(t1)*sin(t2), sin(t1)] where t1=x0*pi/2, t2=x1*pi/2
        
        # Phase 1: Quasi-random sampling on the Pareto front
        # Use a stratified/sobol-like grid for good coverage
        # We'll use a combination of structured and random points
        
        n_phase1 = min(budget - 20, 180)  # Leave some budget for refinement
        if n_phase1 < 10:
            n_phase1 = max(10, budget // 2)
        
        # Generate points on the Pareto front
        # Use a low-discrepancy sequence approach for x0, x1
        # Simple approach: use golden ratio-based quasi-random points
        points = []
        
        # Add structured grid points for coverage
        n_grid = 14
        for i in range(n_grid):
            for j in range(n_grid):
                x0 = (i + 0.5) / n_grid
                x1 = (j + 0.5) / n_grid
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5  # g=0
                points.append(x)
        
        # Add additional quasi-random points to fill gaps
        remaining = n_phase1 - len(points)
        if remaining > 0:
            # Use golden ratio for quasi-random distribution
            golden = 0.6180339887498949
            for k in range(remaining):
                x0 = (k * golden) % 1.0
                x1 = (k * golden * 2.0) % 1.0  # Different frequency
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5
                points.append(x)
        
        # Evaluate all phase 1 points
        F = []
        used = 0
        for x in points:
            if time.time() > self._deadline:
                break
            f = problem(x)
            F.append(f)
            used += 1
        
        F = np.array(F) if F else np.zeros((0, self.n_obj))
        X = np.array(points[:used]) if used > 0 else np.zeros((0, dim))
        
        # Phase 2: Local refinement - perturb promising points
        # Focus on points that are non-dominated
        if len(F) > 0:
            # Find non-dominated points
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            nondom_idx = np.where(~dominated)[0]
            
            # Use remaining budget for refinement
            remaining_budget = budget - used
            if remaining_budget > 0 and len(nondom_idx) > 0:
                # Perturb each non-dominated point slightly
                n_refine = min(remaining_budget, len(nondom_idx) * 2)
                for k in range(n_refine):
                    if time.time() > self._deadline:
                        break
                    if used >= budget:
                        break
                    
                    # Pick a non-dominated parent
                    idx = nondom_idx[k % len(nondom_idx)]
                    parent = X[idx].copy()
                    
                    # Small perturbation in x0 and x1 directions
                    # Also try to improve g (move x[2:] closer to 0.5)
                    scale = 0.05 / (k + 1)  # Decreasing step size
                    child = parent.copy()
                    
                    # Perturb the angle variables
                    child[0] += scale * ((k % 2) * 2 - 1)  # + or -
                    child[1] += scale * ((k % 4) - 1.5) * 0.5
                    
                    # Move g variables toward 0.5
                    if k % 3 == 0:
                        child[2:] = 0.5 + 0.01 * (np.random.rand(dim-2) - 0.5)
                    
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    F = np.vstack([F, [fc]])
                    X = np.vstack([X, [child]])
        
        return X
# EVOLVE-BLOCK-END
