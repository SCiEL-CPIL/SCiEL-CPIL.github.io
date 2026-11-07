# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for WFG4 (3 objectives, 12 variables).
    
    Strategy:
    WFG4 is a 3-objective problem where the first objective f1 is typically
    the first decision variable (or a simple function of it). The Pareto front
    is determined by the trade-off between f1 and the remaining objectives
    f2, f3, which depend on the 'shape' variables (x[1:] for 12D).
    
    For WFG4, the shape variables are usually best set to values that minimize
    the 'distance' to the ideal point. In the standard WFG4 definition, the
    Pareto front is achieved when the shape variables are at their lower bounds
    (or a specific optimal value, often 0 for the first few variables).
    
    This strategy directly samples the Pareto front by:
    1. Uniformly sampling the position variable x[0] to cover the range of f1.
    2. Fixing the shape variables x[1:] to their lower bounds (or optimal values)
       to ensure the points lie on (or very close to) the Pareto front.
    
    This minimizes budget waste by generating non-dominated points directly,
    rather than relying on stochastic search to find them.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract lower and upper bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Create a base vector with shape variables set to their midpoints.
        # For WFG problems, the 'shape' variables (x[1:]) typically have a penalty
        # term like (x_i - 0.5)^2 or x_i^2. The midpoint of the bounds is a strong
        # heuristic for the optimal value that minimizes these penalties, placing
        # the points on or very close to the Pareto front.
        base = (lo + hi) / 2.0
        
        # We will sample x[0] uniformly to spread points along the front.
        # The remaining dimensions are fixed to 'base' (lower bounds).
        
        # Budget allocation:
        # We have `self.budget` evaluations.
        # We will use all of them to sample points along the front.
        
        # Generate unique x[0] values. We want to cover the range [lo[0], hi[0]].
        # To maximize hypervolume, a uniform distribution is good, but we might
        # want to ensure we capture the ends well. 
        # Let's use a deterministic uniform sampling for reproducibility and coverage.
        # Or use random sampling for diversity? The task says "constrained random sampling".
        # Let's use random sampling for x[0] to satisfy the "random" part, but biased
        # to the front by fixing other variables.
        
        # However, deterministic uniform might give better hypervolume coverage if we
        # know the front is 1D. Let's mix: use deterministic uniform points to ensure
        # full coverage, and if we have extra budget, add random points.
        # But 200 evaluations is enough for a 1D front. 
        # Let's use a mix: 80% deterministic uniform, 20% random?
        # Actually, just uniform random is fine, but deterministic uniform might be safer
        # for hypervolume if the front is smooth. Let's stick to random to follow the
        # "random sampling" directive, but ensure we cover the range.
        
        # Let's generate `self.budget` samples.
        x0_samples = np.random.uniform(lo[0], hi[0], self.budget)
        
        # Construct the full vectors
        X = np.tile(base, (self.budget, 1))
        X[:, 0] = x0_samples
        
        # Evaluate all points
        # Note: The problem function clips inputs, but we are already within bounds.
        F = np.array([problem(x) for x in X])
        
        # Return the set of points (or the front). The return value is ignored by the
        # evaluator, but we must return something.
        return X
# EVOLVE-BLOCK-END
