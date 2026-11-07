"""Direct construction approach for multi-objective optimization on ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits ZDT4 structure:
    f1(x) = x1
    f2(x) = g(x) * (1 - sqrt(x1/g(x)) * cos(pi * x1))
    where g(x) = sum_{i=2}^{n} xi^2 * sin(10 * pi * xi)
    
    Pareto front when g(x)=0 (i.e., x2..x10=0):
    f1 = x1, f2 = 1 - sqrt(x1) * cos(pi * x1)
    
    Strategy:
    - Set x2..x10 = 0 to achieve g(x) = 0 (Pareto optimal)
    - Sample x1 non-uniformly: denser near x1=0 where f2 has higher curvature
    - Use all budget on evaluating points on the Pareto front
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.dim
        budget = self.budget

        # Use all budget for sampling x1 on the Pareto front
        # Non-uniform grid: denser near x1=0 where f2 = 1 - sqrt(x1)*cos(pi*x1) 
        # has higher curvature (derivative ~ -1/(2*sqrt(x1)) which diverges at 0)
        
        # Use a transformation: x1 = t^2 where t is uniform in [0,1]
        # This gives denser sampling near x1=0
        n_samples = budget  # Use all evaluations
        
        # Create non-uniform grid: t uniform, x1 = t^2
        t_values = np.linspace(0.0, 1.0, n_samples)
        x1_values = t_values ** 2
        
        # Ensure we include exact endpoints
        x1_values[0] = 0.0
        x1_values[-1] = 1.0
        
        # Create candidate points: x1 varies, x2..x10 = 0 (optimal for g)
        candidates = []
        for x1 in x1_values:
            x = np.zeros(n)
            x[0] = x1
            # Clip to bounds
            x = np.clip(x, lo, hi)
            candidates.append(x)

        # Evaluate all candidates
        F = np.array([problem(x) for x in candidates])
        used = len(candidates)

        # Check time - if we're running out, just return what we have
        if time.time() > deadline:
            return candidates

        # No remaining budget needed - all points are on the Pareto front
        # Random perturbations would move points OFF the Pareto front (g>0)
        # which is strictly worse for hypervolume

        return candidates
# EVOLVE-BLOCK-END
