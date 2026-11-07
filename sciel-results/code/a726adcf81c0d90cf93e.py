"""Optimizer for BBOB f15 (Rastrigin rotated) using adaptive lattice-based candidate generation."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Adaptive lattice-based optimizer for Rastrigin.
    
    Strategy:
    1. Initial exploration: Random sampling to find a promising basin.
    2. Adaptive Lattice Refinement: Iteratively generate a grid of candidates around the current best point,
       starting with coarse steps and refining to finer steps. This mimics the search for lattice minima
       in the rotated frame by exploiting local periodicity.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = np.inf
        self.evaluated = set()

    def _counted_func(self, x):
        """Wrapper to count function calls."""
        self.calls += 1
        return self.func(x)

    def _make_key(self, x):
        """Create a hashable key for a point."""
        return tuple(np.round(x, 8))

    def _evaluate(self, x):
        """Evaluate a point if not already evaluated and within budget."""
        if self.calls >= self.budget:
            return np.inf
        key = self._make_key(x)
        if key in self.evaluated:
            return np.inf
        self.evaluated.add(key)
        fx = self._counted_func(x)
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return fx

    def _generate_grid_candidates(self, center, step, max_radius, max_candidates):
        """
        Generate a set of grid candidates around the center point.
        """
        candidates = []
        dim = self.dim
        # Determine the range of indices for the grid
        # We want to cover a hypercube of side length 2 * max_radius around the center
        n_steps = int(max_radius / step)
        
        # To keep the number of candidates manageable, we limit the radius or use a sparse grid
        # For dim=5, a full grid of (2*n_steps+1)^5 can be huge.
        # We prioritize points closer to the center.
        
        # Generate indices for each dimension
        # We'll generate a list of all possible offsets within the radius
        # But we cap the total number of candidates.
        
        # Simple approach: Randomly sample from the grid space if it's too large,
        # or just iterate if small.
        # Let's estimate the size.
        # If n_steps is small (e.g., 2), size is 5^5 = 3125.
        # If n_steps is 1, size is 3^5 = 243.
        
        # We will generate all points in the grid if the total count is reasonable,
        # otherwise we will sample.
        
        # Calculate total points
        points_per_dim = 2 * n_steps + 1
        total_points = points_per_dim ** dim
        
        if total_points <= max_candidates:
            # Generate all points
            for i0 in range(-n_steps, n_steps + 1):
                for i1 in range(-n_steps, n_steps + 1):
                    for i2 in range(-n_steps, n_steps + 1):
                        for i3 in range(-n_steps, n_steps + 1):
                            for i4 in range(-n_steps, n_steps + 1):
                                offset = np.array([i0, i1, i2, i3, i4]) * step
                                x_cand = center + offset
                                x_cand = np.clip(x_cand, -5, 5)
                                # Check if within radius
                                if np.linalg.norm(x_cand - center) <= max_radius + 1e-8:
                                    candidates.append(x_cand)
        else:
            # Sample max_candidates points from the grid
            # We can do this by randomly picking indices
            for _ in range(max_candidates):
                offsets = []
                for _ in range(dim):
                    idx = np.random.randint(-n_steps, n_steps + 1)
                    offsets.append(idx * step)
                offset = np.array(offsets)
                x_cand = center + offset
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
                
        return candidates

    def __call__(self, func):
        self.func = func
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.evaluated = set()

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial exploration
        # Use 10% of budget for initial random search to find a good basin
        n_initial = min(50, budget // 10)
        for i in range(n_initial):
            if self.calls >= budget:
                break
            x0 = rng.uniform(-5, 5, dim)
            self._evaluate(x0)
            
        if self.best_x is None:
            self._evaluate(np.zeros(dim))

        # Phase 2: Adaptive Lattice Refinement
        # Define a sequence of step sizes to refine the search
        # Start coarse, end fine
        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05]
        
        for step in step_sizes:
            if self.calls >= budget - 5:
                break
                
            if self.best_x is None:
                break
                
            # Determine how many candidates we can evaluate in this phase
            # Leave some budget for subsequent phases
            remaining_budget = budget - self.calls
            # Allocate a portion of the remaining budget to this step size
            # Earlier steps get more budget
            phase_budget = min(remaining_budget // 2, 100)
            if phase_budget < 5:
                phase_budget = remaining_budget
            
            # Generate candidates
            # Max radius: cover a reasonable area around the best point
            # For coarse steps, we can look further
            max_radius = 4.0 if step > 1.0 else 2.0
            
            candidates = self._generate_grid_candidates(self.best_x, step, max_radius, phase_budget)
            
            # Sort candidates by distance to center (closest first) to prioritize local refinement
            if self.best_x is not None:
                candidates.sort(key=lambda x: np.linalg.norm(x - self.best_x))
            
            # Evaluate candidates
            for x_cand in candidates:
                if self.calls >= budget:
                    break
                self._evaluate(x_cand)
                
                # If we found a significantly better point, we might want to re-center
                # But for simplicity, we stick to the original center for this phase
                # The next phase will use the new best_x as center

        # Ensure we've made at least one call
        if self.calls == 0:
            x0 = np.zeros(dim)
            self._evaluate(x0)

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
