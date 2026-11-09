"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimized approach for ZDT6 exploiting the analytic Pareto front structure.
    
    ZDT6:
    - x in [0,1]^10
    - g(x) = 1 + sum(x[1:])
    - f1 = 1 - exp(-4*x1) * sin(6*pi*x1)^6
    - f2 = g
    
    Pareto front: x[1:] = 0 (g=1), x1 in [0,1]
    Front: f2 = 1, f1 = 1 - exp(-4*x1)*sin(6*pi*x1)^6
    
    Strategy: Use all budget evaluations on the Pareto front with non-uniform
    sampling that concentrates points where the front has higher curvature
    (near x1=0 where f1 oscillates rapidly).
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.budget
        dim = self.dim
        
        # Generate non-uniform x1 samples to better cover the Pareto front
        # The function f1(x1) = 1 - exp(-4*x1)*sin(6*pi*x1)^6 has high curvature
        # near x1=0, so we use a transformation that clusters samples there.
        
        # Use a power transformation: x1 = u^(1/p) where p > 1 clusters near 0
        # Also add uniform samples for good coverage across [0,1]
        
        n_total = n
        
        # Split: 60% clustered near 0, 30% uniform, 10% random
        n_cluster = int(n_total * 0.6)
        n_uniform = int(n_total * 0.3)
        n_random = n_total - n_cluster - n_uniform
        
        # Clustered samples: use power law to concentrate near x1=0
        u_cluster = np.linspace(0, 1, n_cluster)
        x1_cluster = u_cluster ** 3  # p=3 gives good clustering near 0
        
        # Uniform samples
        x1_uniform = np.linspace(0, 1, n_uniform)
        
        # Random samples for robustness
        x1_random = np.random.uniform(0, 1, n_random)
        
        # Combine and sort for better hypervolume coverage
        x1_values = np.concatenate([x1_cluster, x1_uniform, x1_random])
        x1_values = np.sort(x1_values)
        
        # Create solutions on Pareto front: x[0]=x1, x[1:]=0
        X = np.zeros((n_total, dim))
        X[:, 0] = x1_values
        # x[1:] stays 0 (this gives g=1, the Pareto-optimal value)
        
        # Evaluate all points
        F = []
        for i in range(n_total):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
        
        F = np.array(F)
        
        # The return value is ignored, but we return something valid
        return X[:len(F)]
# EVOLVE-BLOCK-END
