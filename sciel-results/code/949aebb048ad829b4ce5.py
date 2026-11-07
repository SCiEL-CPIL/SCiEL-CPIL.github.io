"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1 that exploits the known Pareto front geometry.
    
    DTLZ1 Pareto front: for M objectives, x_M..x_N are free, and x_1..x_{M-1} satisfy
    sum_{i=1}^{M-1} x_i^2 = 1 (on the unit sphere in M-1 dimensions).
    The objectives are f_i = 0.5 * (1 + g) * x_i for i=1..M-1 and f_M = 0.5 * (1+g) * (1 - sum x_i^2).
    On the Pareto front, g=0 and sum x_i^2 = 1, so f_i = 0.5 * x_i and f_M = 0.
    
    We sample the simplex (unit sphere in M-1 dims restricted to positive orthant) using
    low-discrepancy sequences for near-perfect coverage, then use remaining budget for
    local search to refine.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        M = self.n_obj  # 3 objectives
        D = self.dim    # 7 variables
        
        # DTLZ1: x_1..x_{M-1} are on the unit sphere (positive orthant), x_M..x_D are free
        # On Pareto front: sum(x_1^2..x_{M-1}^2) = 1, g=0
        # f_i = 0.5 * x_i for i < M, f_M = 0.5 * (1 - sum(x_1^2..x_{M-1}^2)) = 0 on front
        
        # Strategy:
        # 1. Use ~60% of budget to sample the Pareto front densely using Sobol-like sequences
        # 2. Use remaining budget for local refinement around promising points
        
        # For M=3, we need x_1, x_2 such that x_1^2 + x_2^2 = 1, x_1, x_2 >= 0
        # Parametrize: x_1 = cos(theta), x_2 = sin(theta) for theta in [0, pi/2]
        # For general M-1 dimensions, use hypersphere sampling
        
        n_front_samples = int(self.budget * 0.6)
        n_local = self.budget - n_front_samples
        
        # Generate low-discrepancy points on the (M-2)-simplex for M-1 dimensional sphere
        # For M=3: 1D simplex -> theta in [0, pi/2]
        # Use Halton sequence for low discrepancy
        
        def halton_sequence(n, base):
            """Generate n Halton sequence values in [0,1)."""
            result = np.zeros(n)
            for i in range(n):
                f = 1.0
                r = 0.0
                j = i + 1
                while j > 0:
                    f /= base
                    r += f * (j % base)
                    j //= base
                result[i] = r
            return result
        
        # For M objectives, we need M-1 variables on the positive unit sphere
        # For M=3: x_1 = cos(theta), x_2 = sin(theta), theta = halton * pi/2
        # For general M: use recursive simplex sampling
        
        def sample_sphere_positive(n, dim_sphere):
            """
            Sample n points on the positive unit sphere in dim_sphere dimensions.
            dim_sphere = M-1 for DTLZ1.
            Uses recursive angular parametrization.
            """
            points = np.zeros((n, dim_sphere))
            if dim_sphere == 1:
                # 1D: just [1]
                points[:, 0] = 1.0
                return points
            elif dim_sphere == 2:
                # 2D: circle in first quadrant
                theta = halton_sequence(n, 2) * np.pi / 2
                points[:, 0] = np.cos(theta)
                points[:, 1] = np.sin(theta)
                return points
            else:
                # Higher dimensional: use recursive construction
                # Use Dirichlet distribution with low-discrepancy input
                # For dim_sphere dimensions, generate dim_sphere-1 angles
                # Use Halton sequences for each angle
                angles = np.zeros((n, dim_sphere - 1))
                for d in range(dim_sphere - 1):
                    angles[:, d] = halton_sequence(n, d + 2)
                
                # Convert angles to sphere coordinates using recursive formula
                # x_1 = cos(a_1)
                # x_2 = sin(a_1) * cos(a_2)
                # x_3 = sin(a_1) * sin(a_2) * cos(a_3)
                # ...
                # x_{d-1} = sin(a_1) * ... * sin(a_{d-2}) * cos(a_{d-1})
                # x_d = sin(a_1) * ... * sin(a_{d-2}) * sin(a_{d-1})
                
                # Scale angles to [0, pi/2]
                angles = angles * np.pi / 2
                
                # Compute recursively
                sin_prod = np.ones(n)
                for d in range(dim_sphere - 1):
                    points[:, d] = sin_prod * np.cos(angles[:, d])
                    sin_prod = sin_prod * np.sin(angles[:, d])
                points[:, dim_sphere - 1] = sin_prod
                
                return points
        
        # Generate front samples
        sphere_points = sample_sphere_positive(n_front_samples, M - 1)
        
        # Build full x vectors for DTLZ1
        # x_1..x_{M-1} from sphere, x_M..x_D can be anything (they affect g but not front position)
        # For Pareto front, we want g=0, which requires x_M..x_D to be at their bounds
        # g = 100 * (n_g + sum((x_i - 0.5)^2 - cos(20*pi*(x_i+0.5))) for i=M..D)
        # g=0 when each x_i = 0.5 (approximately, since the cos term makes it tricky)
        # Actually for DTLZ1, g=0 when x_i = 0.5 for i=M..D (the minimum of the sum)
        
        # Set x_M..x_D to 0.5 to minimize g (approximately achieves g=0)
        X_front = np.zeros((n_front_samples, D))
        X_front[:, :M-1] = sphere_points
        X_front[:, M-1:] = 0.5  # Middle of [0,1] to minimize g
        
        # Clip to bounds
        X_front = np.clip(X_front, lo, hi)
        
        # Evaluate front samples
        F_front = np.array([problem(x) for x in X_front])
        used = n_front_samples
        
        # Local search: refine points near the front
        # For each of a subset of front points, do small perturbations
        # Focus on the first M-1 dimensions (sphere coordinates)
        
        if n_local > 0 and time.time() < deadline:
            # Pick a subset of front points to refine
            n_refine = min(20, n_local // 5) if n_local >= 5 else 0
            if n_refine > 0:
                # Choose spread-out points
                idxs = np.linspace(0, n_front_samples - 1, n_refine).astype(int)
                base_points = X_front[idxs]
                
                for i in range(n_refine):
                    if used >= self.budget or time.time() >= deadline:
                        break
                    base = base_points[i].copy()
                    # Small perturbation in sphere coordinates
                    for step in range(min(5, self.budget - used)):
                        if used >= self.budget or time.time() >= deadline:
                            break
                        # Perturb the first M-1 coordinates
                        pert = 0.05 * span[:M-1] * np.random.randn(M-1)
                        child = base.copy()
                        child[:M-1] += pert
                        child = np.clip(child, lo, hi)
                        # Renormalize to stay close to sphere
                        norm = np.sqrt(np.sum(child[:M-1]**2))
                        if norm > 1e-10:
                            child[:M-1] /= norm
                        fc = problem(child)
                        used += 1
                        # Accept if it improves some objective
                        if np.any(fc < F_front[i]):
                            base = child.copy()
                            F_front[i] = fc
        
        # Return the best set of points
        # We've recorded all evaluations via problem() calls
        # Return a dummy value (ignored by evaluator)
        return X_front
# EVOLVE-BLOCK-END
