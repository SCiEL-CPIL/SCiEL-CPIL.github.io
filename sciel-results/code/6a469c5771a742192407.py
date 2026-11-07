"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Adaptive multi-objective optimizer using informed sampling.
    Strategy:
    1. Initial uniform exploration.
    2. Adaptive sphere search centered at the lower bounds (normalized origin).
    3. Radius expands on stagnation, shrinks on success.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 290 seconds out of 300
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        used = 0
        archive = []  # List of (x, f) tuples

        def is_dominated_by_archive(f_new):
            for _, f_a in archive:
                if np.all(f_a <= f_new) and np.any(f_a < f_new):
                    return True
            return False

        def update_archive(x, f):
            # Check if dominated by any existing point
            if is_dominated_by_archive(f):
                return False

            # Remove points dominated by the new point
            new_archive = []
            for ax, af in archive:
                # If new point dominates old point, discard old
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))

            new_archive.append((x, f))
            archive[:] = new_archive
            return True

        # Phase 1: Initial Uniform Exploration
        # Use a fixed number of initial points to get a spread
        n_init = min(20, self.budget)
        for _ in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            
            # Uniform random in bounds
            x = lo + np.random.rand(self.dim) * span
            f = np.array(problem(x))
            used += 1
            update_archive(x, f)

        # Phase 2: Adaptive Sphere Search
        # Hypothesis: Pareto front is located near the lower bounds (normalized origin)
        # We sample in a sphere centered at the lower bound in normalized space.
        
        # Start with a moderate radius
        current_radius = 0.5 
        stagnation_count = 0
        max_stagnation = 10
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Mix of focused search (80%) and uniform exploration (20%)
            # This prevents getting stuck if the "origin" assumption is slightly off
            if np.random.rand() < 0.8:
                # Sample in sphere centered at 0 (normalized)
                u = np.random.randn(self.dim)
                norm_u = np.linalg.norm(u)
                if norm_u > 1e-9:
                    u = u / norm_u
                else:
                    u = np.zeros(self.dim)
                    u[0] = 1.0
                
                # Random radius within current_radius to avoid shell artifacts
                r = np.random.rand() * current_radius
                
                x_norm = r * u
                # Clip to [0, 1] to stay within bounds
                x_norm = np.clip(x_norm, 0, 1)
                x = lo + x_norm * span
            else:
                # Uniform fallback
                x = lo + np.random.rand(self.dim) * span
            
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            
            added = update_archive(x, f)
            
            if added:
                stagnation_count = 0
                # Shrink radius to refine the search near the front
                current_radius = max(0.05, current_radius * 0.95)
            else:
                stagnation_count += 1
                if stagnation_count > max_stagnation:
                    # Expand search space to escape local trap
                    current_radius = min(1.0, current_radius * 1.5)
                    stagnation_count = 0
                else:
                    # Slight shrinkage even if no new point, to tighten focus
                    current_radius = max(0.05, current_radius * 0.98)

        return archive
# EVOLVE-BLOCK-END
