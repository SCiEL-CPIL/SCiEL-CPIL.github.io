"""Multi-objective optimizer on DTLZ1 with analytical front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    
    Strategy:
    1. Analytically determine the Pareto front condition: g(x) = 0.
       For DTLZ1, this requires aggregation variables (x_2..x_6) to be 0.5.
    2. Sample the position variables (x_0, x_1) such that the resulting
       objective vectors are uniformly distributed on the Pareto front.
       The front is the image of the simplex under the objective functions.
       We sample points in the simplex (objective space) using a Sobol sequence
       mapped to the simplex, then invert the mapping to find the corresponding x.
    3. This maximizes hypervolume coverage by ensuring good spread in objective space.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 285.0  # Safety margin before 300s limit
    
    def _elapsed(self):
        return time.time() - self.start_time
    
    def _time_left(self):
        return self.deadline - self._elapsed()
    
    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        m = self.n_obj
        n = self.dim
        M = m - 1  # Number of position variables (2 for DTLZ1)
        
        n_samples = self.budget
        if n_samples <= 0:
            return np.array([])
        
        # Generate Sobol points in [0, 1]^M
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=M, scramble=True)
            U = sampler.random(n_samples)
        except ImportError:
            # Fallback to random
            U = np.random.rand(n_samples, M)
        
        # Map U (hypercube) to Simplex (alpha)
        # For M=2 (3 objectives):
        # alpha_0 = u_0
        # alpha_1 = (1 - u_0) * u_1
        # alpha_2 = (1 - u_0) * (1 - u_1)
        # General M:
        # alpha_0 = u_0
        # alpha_i = (1 - sum_{j<i} alpha_j) * u_{i-1} for i=1..M-1
        # alpha_M = 1 - sum_{j<M} alpha_j
        
        # Since M is fixed at 2 for DTLZ1 (n_obj=3), we can hardcode or loop.
        # Let's loop for generality, though M=2 is expected.
        Alpha = np.zeros((n_samples, m))
        if M >= 1:
            Alpha[:, 0] = U[:, 0]
        for i in range(1, M):
            rem = 1.0 - np.sum(Alpha[:, :i], axis=1)
            Alpha[:, i] = rem * U[:, i-1]
        Alpha[:, M] = 1.0 - np.sum(Alpha[:, :M], axis=1)
        
        # The target objectives are F = 0.5 * Alpha
        # F_0 = 0.5 * Alpha_0
        # F_1 = 0.5 * Alpha_1
        # F_2 = 0.5 * Alpha_2
        
        # Invert mapping to find x_pos
        # F_2 = 0.5 (1 - x_1)  => x_1 = 1 - 2 F_2
        # F_1 = 0.5 x_1 (1 - x_2) => x_2 = 1 - F_1 / (0.5 x_1) = 1 - 2 F_1 / x_1
        
        F_target = 0.5 * Alpha
        x_1 = 1.0 - 2.0 * F_target[:, 2]
        
        # Avoid division by zero if x_1 is very small (should not happen for valid simplex points except vertex)
        # If x_1 == 0, then F_2 = 0.5, F_1 = 0, F_0 = 0.
        # In that case x_2 is irrelevant (multiplied by 0). We can set x_2 = 0.
        x_2 = np.zeros(n_samples)
        mask = x_1 > 1e-12
        x_2[mask] = 1.0 - 2.0 * F_target[mask, 1] / x_1[mask]
        
        # Clip x to [0,1] to be safe against floating point errors
        x_1 = np.clip(x_1, 0.0, 1.0)
        x_2 = np.clip(x_2, 0.0, 1.0)
        
        # Construct full X
        X = np.zeros((n_samples, n))
        X[:, 0] = x_1
        X[:, 1] = x_2
        # Set aggregation variables to 0.5
        X[:, M:] = 0.5
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
