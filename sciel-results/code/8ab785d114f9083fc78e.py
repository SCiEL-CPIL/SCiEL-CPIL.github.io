"""Trust-region reflective optimizer with sparse grid initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Trust-region reflective algorithm with sparse grid initialization for BBOB-style optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        
        # Phase 1: Sparse grid initialization to rapidly identify basin
        # Use a 3^5 = 243 point grid on [-4, 4]^5 (minimizer is in [-4,4]^5)
        # This uses 243 calls, leaving 257 for local refinement
        grid_levels = 3
        grid_points = []
        for i in range(grid_levels):
            for j in range(grid_levels):
                for k in range(grid_levels):
                    for l in range(grid_levels):
                        for m in range(grid_levels):
                            grid_points.append([i / (grid_levels - 1) * 8 - 4,
                                                j / (grid_levels - 1) * 8 - 4,
                                                k / (grid_levels - 1) * 8 - 4,
                                                l / (grid_levels - 1) * 8 - 4,
                                                m / (grid_levels - 1) * 8 - 4])
        
        grid_points = np.array(grid_points[:min(len(grid_points), 200)])
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Evaluate grid points
        for x in grid_points:
            if time.time() > deadline - 10.0:
                break
            if calls_made >= budget:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Trust-region reflective local refinement
        # Start from the best grid point
        if best_x is None:
            # Fallback: random point
            best_x = rng.uniform(-4, 4, dim)
            best_f = func(best_x)
            calls_made += 1
        
        x = best_x.copy()
        fx = best_f
        
        # Trust region radius
        delta = 1.0
        delta_max = 4.0
        delta_min = 1e-4
        
        # Step size for finite differences
        eps = 1e-4
        
        # Line search parameters
        rho1 = 0.1
        rho2 = 0.75
        c1 = 1e-4
        c2 = 0.9
        
        while calls_made < budget and time.time() < deadline - 5.0:
            # Compute gradient via finite differences
            grad = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xp[i] += eps
                xp = np.clip(xp, -5, 5)
                fm = x.copy()
                fm[i] -= eps
                fm = np.clip(fm, -5, 5)
                
                fp = func(xp)
                fm_val = func(fm)
                calls_made += 2
                
                if calls_made >= budget:
                    break
                
                grad[i] = (fp - fm_val) / (2 * eps)
            
            if calls_made >= budget:
                break
            
            # Approximate Hessian via finite differences of gradient
            # Use a diagonal approximation for efficiency
            H = np.eye(dim) * 1.0  # Start with identity
            
            # Better: estimate curvature from recent steps
            # For simplicity, use a fixed Hessian approximation based on gradient magnitude
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-8:
                # Estimate diagonal Hessian entries
                for i in range(dim):
                    if abs(grad[i]) > 1e-8:
                        H[i, i] = 1.0  # Default
                    else:
                        H[i, i] = 0.1  # Smaller if gradient is small in this direction
            
            # Solve trust region subproblem: min 0.5*d^T H d + grad^T d
            # subject to ||d|| <= delta
            # For diagonal H, this is separable
            # d_i = -grad_i / H_ii, clipped to trust region
            
            d = -grad / np.diag(H)
            
            # Check trust region constraint
            d_norm = np.linalg.norm(d)
            if d_norm > delta:
                d = d * (delta / d_norm)
            
            # Proposed step
            x_new = x + d
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate new point
            f_new = func(x_new)
            calls_made += 1
            
            # Compute actual reduction
            actual_reduction = fx - f_new
            
            # Compute predicted reduction
            predicted_reduction = -grad @ d - 0.5 * d @ H @ d
            
            # Compute ratio
            if predicted_reduction > 1e-8:
                rho = actual_reduction / predicted_reduction
            else:
                rho = 0.0
            
            # Update trust region
            if rho > rho2:
                # Step was good, expand trust region
                delta = min(delta_max, 2.0 * delta)
                x = x_new
                fx = f_new
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            elif rho > rho1:
                # Step was reasonable, keep trust region
                x = x_new
                fx = f_new
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Step was bad, shrink trust region
                delta = max(delta_min, 0.5 * delta)
            
            # If gradient is very small, we may be at a minimum
            if grad_norm < 1e-6:
                break
            
            # If delta is very small, we may be stuck
            if delta < delta_min * 1.1:
                # Try a random restart from current best
                if calls_made + 10 < budget:
                    x = best_x + rng.standard_normal(dim) * 0.5
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls_made += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    delta = 1.0
                else:
                    break
        
        return best_x, best_f

# EVOLVE-BLOCK-END
