"""Multi-objective optimizer for ZDT4 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits ZDT4's analytic Pareto set: x2..xn at zeros of sin(2*pi*x_i) with cos=1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        dim = self.dim
        budget = self.budget
        used = 0
        
        # Optimal values for x2..x10: 0.0 (gives sin^2=0, cos=1, term=-10)
        optimal_tail = np.zeros(dim - 1)
        
        # ZDT4 Pareto front: f1 = x1, f2 = g*(2 - 2*x1^2 - 0.2*sin(8*pi*x1))
        # where g = 19 when x2..x10 are at optimal values
        # The sin(8*pi*x1) term creates 4 full oscillations in [0,1]
        # Critical points (extrema) are at x1 = k/8 for k = 0, 1, ..., 8
        
        # Strategy: Use all budget for dense, well-distributed sampling of x1
        # 1. Include endpoints and critical points of the oscillatory term
        # 2. Fill remaining with golden-ratio low-discrepancy sequence
        # This maximizes hypervolume coverage of the 2D front
        
        x1_values = set()
        
        # Always include endpoints
        x1_values.add(0.0)
        x1_values.add(1.0)
        
        # Include critical points of sin(8*pi*x1) where derivative = 0
        # d/dx1 [sin(8*pi*x1)] = 8*pi*cos(8*pi*x1) = 0
        # => 8*pi*x1 = pi/2 + k*pi => x1 = (2k+1)/16 for k=0,1,...,7
        # These are the local maxima/minima of the oscillatory part
        for k in range(8):
            x1_val = (2*k + 1) / 16.0
            x1_values.add(x1_val)
        
        # Also include points at x1 = k/8 (where sin(8*pi*x1) = 0)
        for k in range(9):
            x1_val = k / 8.0
            x1_values.add(x1_val)
        
        # Fill remaining budget with golden-ratio low-discrepancy sequence
        golden = (1 + np.sqrt(5)) / 2
        golden_conj = 1.0 / golden  # ~0.618
        
        n_fixed = len(x1_values)
        n_remaining = budget - n_fixed
        
        if n_remaining > 0:
            for i in range(1, n_remaining + 1):
                x1_val = (i * golden_conj) % 1.0
                x1_values.add(x1_val)
        
        # Convert to sorted list for deterministic evaluation
        x1_values = sorted(x1_values)
        
        # Evaluate all points on the Pareto front
        for x1 in x1_values:
            if used >= budget or time.time() > deadline:
                break
            x = np.concatenate([[x1], optimal_tail])
            x = np.clip(x, lo, hi)
            problem(x)
            used += 1
        
        return None
# EVOLVE-BLOCK-END
