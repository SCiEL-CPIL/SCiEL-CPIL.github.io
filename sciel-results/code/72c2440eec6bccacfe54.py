"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start local optimizer exploiting ZDT3's separable structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # ZDT3 structure: f1 = x[0], f2 = (1+10n)*g(x[1:])*(1 - sqrt(f1/f2_true) * sin(10*f1*pi))
        # Key insight: x[1:] should be optimized to minimize g(x[1:]) = sum(x[i] - 0.35*exp(-4x[i]-log(0.35)))*sin(6*x[i]*pi)^3 + 10
        # The minimum of g is 0, achieved when each x[i] = 0.35*exp(-4x[i]-log(0.35))
        
        # Precompute optimal values for x[1:] that minimize g
        # For ZDT3, the optimal x[i] for i>=1 is approximately 0.35 (where the sin term is small)
        # Let's use a smarter initialization: set x[1:] to values that make g small
        
        # Generate diverse starting points on the Pareto front
        # f1 ranges from 0 to 1, so x[0] should cover [0, 1]
        # We'll use a combination of uniform and log-spaced points for better coverage
        
        n_starts = min(50, self.budget // 5)
        
        # Create diverse x[0] values: mix of uniform and concentrated near 0
        x0_uniform = np.linspace(0.01, 0.99, n_starts // 2)
        x0_log = np.exp(np.linspace(np.log(0.001), np.log(0.5), n_starts - n_starts // 2))
        x0_values = np.concatenate([x0_uniform, x0_log])[:n_starts]
        
        # For x[1:], use values that minimize the penalty
        # In ZDT3, setting x[i] = 0.35 makes sin(6*0.35*pi) = sin(2.1*pi) = sin(0.1*pi) which is small
        # Actually, let's try x[i] = 0.35 for all i>=1 as a good starting point
        x_tail_base = np.full(self.dim - 1, 0.35)
        
        all_points = []
        all_fvals = []
        used = 0
        
        # Phase 1: Evaluate diverse starting points
        for x0 in x0_values:
            if used >= self.budget or time.time() > deadline:
                break
            x = np.empty(self.dim)
            x[0] = x0
            x[1:] = x_tail_base + rng.standard_normal(self.dim - 1) * 0.05
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_points.append(x.copy())
            all_fvals.append(f.copy())
            used += 1
        
        # Phase 2: Local refinement using gradient-free optimization
        # For each point, do a few steps of coordinate descent
        max_iter_per_point = min(10, (self.budget - used) // max(1, len(all_points)))
        
        for idx in range(len(all_points)):
            if used >= self.budget or time.time() > deadline:
                break
            
            current_x = all_points[idx].copy()
            current_f = all_fvals[idx].copy()
            best_x = current_x.copy()
            best_f = current_f.copy()
            
            # Coordinate descent: try perturbing each dimension
            for dim_idx in range(self.dim):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Try both directions
                for direction in [-1, 1]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    # Adaptive step size
                    step = 0.1 * (hi[dim_idx] - lo[dim_idx])
                    new_x = current_x.copy()
                    new_x[dim_idx] += direction * step
                    new_x = np.clip(new_x, lo, hi)
                    
                    if np.allclose(new_x, current_x):
                        continue
                    
                    new_f = problem(new_x)
                    used += 1
                    
                    # Check if new point is better (non-dominated)
                    if np.all(new_f <= current_f) and np.any(new_f < current_f):
                        current_x = new_x
                        current_f = new_f
                        if np.all(new_f <= best_f) and np.any(new_f < best_f):
                            best_x = new_x.copy()
                            best_f = new_f.copy()
            
            # Update the point with the best found
            all_points[idx] = best_x
            all_fvals[idx] = best_f
        
        # Phase 3: If budget remains, do more random exploration
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            n_extra = min(remaining, 20)
            for _ in range(n_extra):
                if used >= self.budget or time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(self.dim)
                f = problem(x)
                all_points.append(x.copy())
                all_fvals.append(f.copy())
                used += 1
        
        # Return the best non-dominated points
        if len(all_fvals) == 0:
            return np.array([np.mean([lo, hi], axis=0)])
        
        F = np.array(all_fvals)
        # Filter out points that are dominated
        nondominated_mask = np.ones(len(F), dtype=bool)
        for i in range(len(F)):
            if not nondominated_mask[i]:
                continue
            for j in range(len(F)):
                if i == j or not nondominated_mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nondominated_mask[i] = False
                    break
        
        if np.any(nondominated_mask):
            return np.array(all_points)[nondominated_mask]
        else:
            return np.array(all_points)
# EVOLVE-BLOCK-END
