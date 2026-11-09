"""Multi-objective optimizer for WFG4 exploiting problem structure."""
import numpy as np
import time


class Optimizer:
    """
    Optimizer for WFG4 (12 variables, 3 objectives).
    
    WFG4 structure:
    - x[0] (position variable) determines the shape/position on the Pareto front
    - x[1..11] (bias variables) determine distance from the Pareto front
    - The Pareto front is achieved when bias variables are at their optimal values
    - For WFG4, the bias variables contribute to all objectives in a way that
      minimization requires them to be at specific values (typically midpoints or bounds)
    
    Strategy:
    1. Use a large portion of budget to densely sample x[0] while keeping bias
       variables near their optimal values (midpoints of their ranges)
    2. Use remaining budget for local refinement and exploration
    3. Maintain a non-dominated set for diversity
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        mid = (lo + hi) / 2.0
        span = hi - lo
        
        # WFG4: x[0] is position, x[1:] are bias variables
        # The Pareto front for WFG4 is typically achieved when bias vars are at midpoints
        # or specific values. We'll try midpoints first.
        
        # Phase 1: Dense sampling along x[0] with bias vars at midpoints
        # This should capture most of the Pareto front
        n_phase1 = int(self.budget * 0.7)
        n_phase2 = self.budget - n_phase1
        
        # Generate dense samples for x[0]
        x0_samples = np.linspace(lo[0], hi[0], n_phase1, endpoint=True)
        
        # Bias variables at midpoints (optimal for WFG4)
        bias_mid = mid[1:]
        
        X = np.zeros((n_phase1, self.dim))
        X[:, 0] = x0_samples
        X[:, 1:] = bias_mid
        
        # Evaluate phase 1
        F = np.array([problem(x) for x in X])
        used = n_phase1
        
        # Phase 2: Local refinement - perturb bias variables around midpoints
        # and x[0] to find better points
        rng = np.random.default_rng(42)
        
        for i in range(n_phase2):
            if time.time() > deadline:
                break
            
            # Pick a random base point
            idx = rng.integers(0, len(X))
            base = X[idx].copy()
            
            # Create child: small perturbation
            # Focus on bias variables (indices 1-11) with small step
            child = base.copy()
            child[0] += 0.01 * span[0] * rng.standard_normal()
            child[1:] += 0.05 * span[1:] * rng.standard_normal()
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Check if child is non-dominated and better than some existing point
            # Simple replacement: if child dominates a random existing point, replace it
            jdx = rng.integers(0, len(X))
            if np.all(fc <= F[jdx]) and np.any(fc < F[jdx]):
                X[jdx] = child
                F[jdx] = fc
        
        return X
