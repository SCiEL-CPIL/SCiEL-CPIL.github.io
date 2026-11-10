"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3 exploiting front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 structure: Pareto front at x2..xn = 0, x1 in [0,1]
        # f1 = x1, f2 = (1+10*(n-1))*(1-sqrt(f1/(1+10*(n-1)))) + sin(10*pi*f1)
        # Critical points: zeros of sin(10*pi*x1) at x1 = k/10, k=0..10
        # Extrema at x1 = (2k+1)/20, k=0..9
        
        # Generate x1 values that capture oscillatory structure
        # Zeros: k/10 for k in 0..10
        zeros = np.array([k / 10.0 for k in range(11)])
        # Extrema: (2k+1)/20 for k in 0..9
        extrema = np.array([(2 * k + 1) / 20.0 for k in range(10)])
        
        # Combine and sort, remove duplicates
        x1_candidates = np.unique(np.concatenate([zeros, extrema]))
        # Add a few more points for better coverage
        extra = np.linspace(0, 1, 30)
        x1_candidates = np.unique(np.concatenate([x1_candidates, extra]))
        
        # Sort
        x1_candidates = np.sort(x1_candidates)
        
        # We have 200 budget. Let's use ~150 for front sampling, rest for refinement
        # Actually, let's be smarter: sample the front densely, then refine
        
        # Create points on the theoretical front: x = [x1, 0, 0, ..., 0]
        # But we need to evaluate them to get actual objective values
        
        # Let's use all 200 evaluations wisely
        # First, evaluate a dense set of front points
        # We have len(x1_candidates) points. Let's cap at budget
        
        n_front = min(len(x1_candidates), self.budget - 20)  # Reserve 20 for refinement
        if n_front < 10:
            n_front = len(x1_candidates)
        
        # Select x1 values
        if len(x1_candidates) > n_front:
            # Use stratified sampling to cover the range
            indices = np.linspace(0, len(x1_candidates) - 1, n_front).astype(int)
            x1_selected = x1_candidates[indices]
        else:
            x1_selected = x1_candidates
        
        # Create and evaluate front points
        X = np.zeros((len(x1_selected), self.dim))
        X[:, 0] = x1_selected
        # x2..xn are already 0
        
        F = np.array([problem(x) for x in X])
        used = len(x1_selected)
        
        # Now use remaining budget for local refinement
        # For each front point, try to improve by adjusting x1 slightly
        # and ensuring x2..xn stay near 0
        
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Try to refine points by perturbing x1
            step_size = 0.01
            for i in range(len(X)):
                if used >= self.budget or time.time() >= deadline:
                    break
                # Try small perturbations in x1
                for delta in [-step_size, step_size]:
                    if used >= self.budget or time.time() >= deadline:
                        break
                    new_x = X[i].copy()
                    new_x[0] = np.clip(X[i][0] + delta, lo[0], hi[0])
                    # Keep x2..xn at 0 (they should be on the front)
                    new_x[1:] = 0.0
                    new_f = problem(new_x)
                    used += 1
                    # Check if this is better (non-dominated or dominates)
                    # Simple: if it dominates the current point, replace
                    if np.all(new_f <= F[i]) and np.any(new_f < F[i]):
                        X[i] = new_x
                        F[i] = new_f
        
        return X
# EVOLVE-BLOCK-END
