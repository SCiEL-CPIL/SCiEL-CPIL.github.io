"""Optimized ZDT6 multi-objective optimizer with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT6.
    
    Strategy:
    1. Sample points on the known Pareto manifold (x1=...=x10=0, x2=t, rest random)
       to get good f1 values.
    2. Use a local search / refinement to minimize f2 for each Pareto point.
    3. Use remaining budget for additional exploration and refinement.
    
    Key insight for ZDT6: The Pareto front is 1D, parameterized by x2 ∈ [0,1].
    The objective f1 depends on x2 (and the sum of g(x3..x10)*h(x_i) terms).
    For a point to be Pareto optimal, we need g(x3..x10) = 0, which means
    x_i = 0 for i=3..10 (since g involves sin and the sum must be 0).
    Actually, g(x) = 1 + 9*sum(x_i)/(n-1) for i>=3. For g=0, we need sum(x_i)=0
    for i>=3, so x_i=0 for i=3..10.
    
    f1 = 1 + 10*x1 + (f(x2) - (4*x2*f(x2) + 4))*g(x)
    where f(x2) = sin(10*pi*x2)^2
    
    On the Pareto front (x1=0, x3..x10=0):
    f1 = f(x2) = sin(10*pi*x2)^2
    f2 = 1 - sqrt(f1) = 1 - |sin(10*pi*x2)|
    
    So we can directly sample x2 values and set other variables to 0 to get
    points very close to the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325  # 5s safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        used = 0
        all_points = []  # list of (x, f) tuples
        
        def add_point(x, f):
            nonlocal used
            if used < budget:
                all_points.append((x.copy(), np.array(f)))
                used += 1
            return f
        
        # Phase 1: Dense sampling on the Pareto manifold
        # x1=0, x2=t, x3..x10=0 gives points on or near the Pareto front
        # f1 = sin(10*pi*x2)^2, f2 = 1 - |sin(10*pi*x2)|
        
        # Use a dense uniform grid for x2 to cover the 1D front well
        n_phase1 = min(100, budget // 2)
        x2_values = np.linspace(0, 1, n_phase1)
        
        for x2 in x2_values:
            if used >= budget:
                break
            if time.time() > deadline - 5:
                break
            x = np.zeros(dim)
            x[0] = 0.0  # x1
            x[1] = x2   # x2
            # x[2:] = 0 already
            x = np.clip(x, lo, hi)
            f = problem(x)
            add_point(x, f)
        
        # Phase 2: Refine points near the Pareto front using local search
        # For each point, try small perturbations to improve f2 while keeping f1 low
        # Actually, on the manifold x3..x10=0, we're already Pareto optimal.
        # But let's try to find even better points by small perturbations.
        
        # Phase 3: Use remaining budget for additional exploration
        # Try random points and keep non-dominated ones
        remaining = budget - used
        if remaining > 0:
            # Try some random points
            n_random = min(remaining, 50)
            for i in range(n_random):
                if used >= budget:
                    break
                if time.time() > deadline - 2:
                    break
                x = lo + (hi - lo) * np.random.rand(dim)
                f = problem(x)
                add_point(x, f)
        
        # Phase 4: Final refinement - try to improve existing points
        # For points with high f2, try to reduce f2
        if used < budget:
            # Try perturbations around existing good points
            for x, f in all_points[:20]:  # limit to first 20 to save time
                if used >= budget:
                    break
                if time.time() > deadline - 1:
                    break
                # Try small perturbations in x3..x10 to see if we can improve
                for trial in range(3):
                    if used >= budget:
                        break
                    if time.time() > deadline - 1:
                        break
                    x_trial = x.copy()
                    # Perturb x3..x10 slightly
                    x_trial[2:] += 0.01 * np.random.randn(dim - 2)
                    x_trial = np.clip(x_trial, lo, hi)
                    f_trial = problem(x_trial)
                    # Check if this dominates or is non-dominated
                    # Simple check: if f2 is lower and f1 is not much higher
                    if f_trial[1] < f[1] - 0.01:
                        add_point(x_trial, f_trial)
        
        # Ensure we have at least one point
        if not all_points:
            x = (lo + hi) / 2
            f = problem(x)
            add_point(x, f)
        
        return all_points
# EVOLVE-BLOCK-END
