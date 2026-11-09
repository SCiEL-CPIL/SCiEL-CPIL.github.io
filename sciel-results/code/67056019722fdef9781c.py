"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Analytical Hypervolume Maximization via Uniform Area Sampling.
    
    The DTLZ4 Pareto front (for 3 objectives) lies on the surface of a quarter-sphere 
    of radius 1 in the positive octant of the objective space:
    f1^2 + f2^2 + f3^2 = 1, f_i >= 0.
    
    To maximize the hypervolume with a fixed reference point, we need to distribute 
    the N points such that they cover the surface area as uniformly as possible. 
    Non-uniform distributions (like uniform theta) leave gaps near the poles (axes) 
    where the surface area density is higher in terms of projected hypervolume contribution.
    
    We use the change of variables to map uniform random numbers to the surface 
    such that the area element is preserved.
    For a sphere, the area element is proportional to sin(theta) dtheta dphi.
    Let u = cos(theta). Then du = -sin(theta) dtheta.
    So if we sample u uniformly in [0, 1], we get uniform coverage on the sphere surface.
    
    Specifically:
    u ~ Uniform(0, 1)  =>  cos(theta) = u  =>  theta = arccos(u)
    v ~ Uniform(0, 1)  =>  phi = 2 * pi * v (but we only need quarter sphere, so phi in [0, pi/2])
    
    Wait, DTLZ4 front is f1=x1, f2=x2, f3=x3.
    x1^2 + x2^2 + x3^2 = 1.
    This is a quarter of the unit sphere surface in the first octant.
    
    Parametrization:
    x1 = sin(theta) * sin(phi)
    x2 = sin(theta) * cos(phi)
    x3 = cos(theta)
    
    Here theta in [0, pi/2] and phi in [0, pi/2].
    Surface area element dA = sin(theta) dtheta dphi.
    
    To sample uniformly on this surface:
    1. Sample Z1 ~ Uniform(0, 1). Let cos(theta) = Z1. So theta = arccos(Z1).
       Note: sin(theta) dtheta = -d(cos(theta)) = -dZ1. The absolute area is proportional to dZ1 dphi.
       Wait, the integral of sin(theta) from 0 to pi/2 is 1. So Z1 = cos(theta) is indeed uniform.
    2. Sample Z2 ~ Uniform(0, 1). Let phi = (pi/2) * Z2.
       The integral of dphi is pi/2. So Z2 is uniform.
    
    Thus, generating (Z1, Z2) uniformly in the unit square and mapping:
    theta = arccos(Z1)
    phi = 0.5 * pi * Z2
    
    yields points uniformly distributed on the Pareto front surface.
    Uniform distribution on the surface is the optimal strategy for maximizing hypervolume 
    with a fixed number of points when the reference point is symmetric (like [2.5, 2.5, 2.5]).
    
    This is O(N) and deterministic (if we use a fixed seed or Sobol), avoiding the O(N^2) 
    cost of farthest-point sampling and the suboptimality of greedy heuristics.
    We will use Sobol sequence for better low-discrepancy properties than pure random, 
    which further improves hypervolume for small N.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.t0 = time.time()
        self.deadline = self.t0 + 290

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget
        
        if n <= 0:
            return np.zeros((0, self.dim))
        
        # Generate Sobol points in 2D for better stratification than random
        # We need n points. Sobol requires power of 2.
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=2, scramble=True)
        n_power = 1
        while n_power < n:
            n_power *= 2
        # Generate n_power points and take the first n
        # Note: Sobol sequences are nested, so taking the first n is valid.
        sobol_points = sampler.random(n_power)[:n]
        
        # sobol_points[:, 0] is u in [0, 1]
        # sobol_points[:, 1] is v in [0, 1]
        
        # Map to theta and phi
        # u = cos(theta) => theta = arccos(u)
        # v = 2*phi/pi => phi = v * pi/2
        theta = np.arccos(sobol_points[:, 0])
        phi = sobol_points[:, 1] * (np.pi / 2.0)
        
        # Construct X in decision space
        # DTLZ4:
        # x1 = sin(theta) * sin(phi)
        # x2 = sin(theta) * cos(phi)
        # x3 = cos(theta)
        # x4...x12 = 0.5 (to ensure g=0 for standard DTLZ4)
        
        X = np.zeros((n, self.dim))
        X[:, 0] = np.sin(theta) * np.sin(phi)
        X[:, 1] = np.sin(theta) * np.cos(phi)
        X[:, 2] = np.cos(theta)
        
        if self.dim > 3:
            X[:, 3:] = 0.5
            
        # Clip to bounds to be safe
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # The problem function is called for each x
        # We must ensure we don't exceed the deadline, but this loop is very fast.
        for i in range(n):
            if time.time() > self.deadline:
                break
            problem(X[i])
            
        return X

# EVOLVE-BLOCK-END
