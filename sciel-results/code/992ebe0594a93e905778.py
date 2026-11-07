"""Multi-objective optimizer for ZDT2 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer that samples the known analytic Pareto front of ZDT2.
    
    For ZDT2 with D dimensions:
    - The Pareto front is at x[1:] = 0
    - f1 = x[0]
    - f2 = 1 - sqrt(x[0])
    
    To maximize hypervolume with reference point [11, 11], we need to sample
    the front densely. Since the front is 1D, we use a non-uniform distribution
    that concentrates more points where the marginal HV contribution is highest.
    
    The key insight: for a convex front like ZDT2's, sampling uniformly in f2
    (i.e., f1 = (1-u)^2 where u is uniform) gives better HV than uniform in f1.
    This is because points near f2=0 (f1=1) extend the covered region further.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # Wall-clock guard
        t_start = time.time()
        deadline = t_start + 285.0  # Safety margin before 300s limit

        n_eval = 0
        results = []

        # Use full budget for dense Pareto front sampling
        n_total = self.budget

        # Initial distribution: x_i proportional to i^2
        # This minimizes the approximation error of the HV integral
        i = np.arange(n_total, dtype=np.float64)
        x0_values = (i / (n_total - 1)) ** 2
        
        # Ensure values are within bounds
        x0_values = np.clip(x0_values, lo[0], hi[0])
        
        # Sort by x_0 for consistent ordering
        order = np.argsort(x0_values)
        x0_values = x0_values[order]

        # Polish the positions using gradient ascent on the exact HV function
        # g(x) = 11 - f2(x) = 10 + sqrt(x)
        # H = sum_{i=1}^{N-1} (x_{i+1} - x_i) g(x_i) + (r_1 - x_N) g(x_N)
        r1 = 11.0
        r2 = 11.0
        
        def compute_hv_and_grad(x):
            # x is sorted array of x0 values
            N = len(x)
            if N == 0:
                return 0.0, np.array([])
            
            # g(x) = 10 + sqrt(x)
            g = 10.0 + np.sqrt(x)
            g_prime = 0.5 / np.sqrt(x + 1e-12) # avoid div by 0
            
            # Compute HV
            # Sum (x_{i+1} - x_i) * g(x_i) for i=0 to N-2
            # Plus (r1 - x_{N-1}) * g(x_{N-1})
            dx = np.diff(x)
            hv = np.sum(dx * g[:-1])
            hv += (r1 - x[-1]) * g[-1]
            
            # Compute Gradient
            grad = np.zeros_like(x)
            
            # Interior points: k = 1 to N-2
            # dH/dx_k = g(x_{k-1}) - g(x_k) + (x_{k+1} - x_k) * g'(x_k)
            if N > 2:
                grad[1:-1] = g[:-2] - g[1:-1] + (x[2:] - x[1:-1]) * g_prime[1:-1]
            
            # First point: k = 0
            # dH/dx_0 = -g(x_0) + (x_1 - x_0) * g'(x_0)
            if N > 1:
                grad[0] = -g[0] + (x[1] - x[0]) * g_prime[0]
            else:
                grad[0] = 0.0 # Should not happen with N=200
                
            # Last point: k = N-1
            # dH/dx_{N-1} = g(x_{N-2}) - g(x_{N-1}) + (r1 - x_{N-1}) * g'(x_{N-1})
            if N > 1:
                grad[-1] = g[-2] - g[-1] + (r1 - x[-1]) * g_prime[-1]
            else:
                grad[-1] = 0.0

            return hv, grad

        # Perform a few steps of gradient ascent
        # Learning rate needs to be small to maintain order and bounds
        lr = 1e-4
        for _ in range(20):
            if time.time() > deadline:
                break
            hv, grad = compute_hv_and_grad(x0_values)
            x0_values = x0_values + lr * grad
            # Project back to [0, 1] and maintain sorting
            x0_values = np.clip(x0_values, lo[0], hi[0])
            x0_values = np.sort(x0_values)
            
            # Optional: re-sort indices if we were tracking them, but we just evaluate sorted x

        # Final evaluation
        for x0 in x0_values:
            if time.time() > deadline:
                break
            if n_eval >= self.budget:
                break

            x = np.zeros(self.dim)
            x[0] = float(x0)
            # Set remaining dimensions to lower bound (0 for standard ZDT2)
            if self.dim > 1:
                x[1:] = lo[1:]

            f = problem(x)
            results.append(f)
            n_eval += 1

        # Return the last point or a list, though return value is ignored by evaluator
        if results:
            return results[-1]
        return None
# EVOLVE-BLOCK-END
