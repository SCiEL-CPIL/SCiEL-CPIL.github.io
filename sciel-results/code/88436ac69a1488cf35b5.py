"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-aware optimizer: estimate gradient via finite differences, jump to optimal vertex."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        # Phase 1: Estimate gradient using 6 evaluations (origin + 5 axis probes)
        # Use a small step h for numerical stability
        h = 0.1
        x0 = np.zeros(dim)
        f0 = func(x0)
        
        # Evaluate at x0 + h * e_i for each dimension
        f_probes = []
        for i in range(dim):
            xi = x0.copy()
            xi[i] += h
            fi = func(xi)
            f_probes.append(fi)
        
        # Compute gradient estimate: g_i = (f_i - f_0) / h
        # For linear function, this is exact
        grad = np.array([(f_probes[i] - f0) / h for i in range(dim)])
        
        # Phase 2: Determine optimal vertex
        # For linear f(x) = grad · x + c, minimum on [-5,5]^dim is at vertex where
        # x_i = -5 if grad_i > 0, x_i = +5 if grad_i < 0
        # If grad_i ≈ 0, either value works (use 0 for safety)
        x_opt = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 1e-10:
                x_opt[i] = -5.0
            elif grad[i] < -1e-10:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0  # Gradient is ~0, any value in [-5,5] is optimal
        
        f_opt_est = func(x_opt)
        
        # Phase 3: Use remaining budget for verification and polishing
        # Since we believe we've found the exact optimum, we can use remaining calls
        # to verify and potentially refine if there's any nonlinearity we missed
        
        # Check if we're within deadline
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return x_opt, f_opt_est
        
        # Use remaining calls for a simple local refinement around x_opt
        # In case there's slight nonlinearity, do a few local searches
        remaining_calls = budget - 7  # We've used 7 calls so far
        if remaining_calls > 0:
            # Try a few local perturbations to see if we can improve
            sigma = 0.5
            for _ in range(min(remaining_calls, 50)):
                if time.time() - start_time > deadline:
                    break
                y = np.clip(x_opt + sigma * np.random.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < f_opt_est:
                    x_opt = y
                    f_opt_est = fy
                    sigma *= 0.5
                else:
                    sigma *= 0.7
        
        return x_opt, f_opt_est
# EVOLVE-BLOCK-END
