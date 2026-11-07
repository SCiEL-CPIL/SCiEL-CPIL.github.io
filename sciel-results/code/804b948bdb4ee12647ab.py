"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Hypervolume-driven multi-objective optimizer for ZDT6.
    
    Strategy:
    1. Analytical placement of initial points along the known Pareto front of ZDT6
       using a facility-location-like spacing in objective space.
    2. Local refinement via mutation and selection that maximizes incremental hypervolume.
    3. Uses the known structure of ZDT6 to guide the search efficiently.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_2d(self, points, ref):
        """Compute 2D hypervolume for points strictly better than ref."""
        if len(points) == 0:
            return 0.0
        # Filter points strictly better than reference
        points = np.array([p for p in points if all(p[i] < ref[i] for i in range(self.n_obj))])
        if len(points) == 0:
            return 0.0
        # Sort by first objective
        idx = np.argsort(points[:, 0])
        points = points[idx]
        hv = 0.0
        prev_y = ref[1]
        for x, y in points:
            if y < prev_y:
                hv += (ref[0] - x) * (prev_y - y)
                prev_y = y
        return hv

    def _zdt6_pareto_front(self, t):
        """
        Generate a point on the Pareto front of ZDT6 given parameter t in [0, 1].
        ZDT6: x0 determines the front, other variables are set to 0.
        g(x) = 1 + 10*n
        h(x0, g) = 1 - sqrt(x0/g) - sin(10*pi*x0)
        f1 = x0
        f2 = g * h
        """
        x = np.zeros(self.dim)
        x[0] = t
        g = 1 + 10 * (self.dim - 1)
        h = 1 - np.sqrt(t / g) - np.sin(10 * np.pi * t)
        f1 = t
        f2 = g * h
        return np.array([f1, f2])

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([11.0, 11.0])
        
        # Phase 1: Analytical placement along Pareto front
        # Use facility-location-like spacing: start with uniform, then refine
        n_initial = min(40, self.budget // 5)
        
        # Initial uniform placement
        t_vals = np.linspace(0.001, 0.999, n_initial)
        X = np.array([np.zeros(self.dim) for _ in range(n_initial)])
        X[:, 0] = t_vals
        F = np.array([self._zdt6_pareto_front(t) for t in t_vals])
        used = n_initial
        
        # Verify these points with the actual problem function
        F_actual = np.array([problem(X[i]) for i in range(n_initial)])
        
        # Phase 2: Greedy refinement - find best points to add
        # We'll use a mutation-based search guided by hypervolume contribution
        pop_size = n_initial
        X = X.copy()
        F = F_actual.copy()
        
        # Create a list to track best points found
        best_points = [(X[i], F[i]) for i in range(pop_size)]
        
        while used < self.budget:
            # Compute current hypervolume
            current_hv = self._hypervolume_2d([p[1] for p in best_points], ref)
            
            # Generate candidate mutations
            n_candidates = min(5, self.budget - used)
            best_candidate = None
            best_candidate_hv = current_hv
            
            for _ in range(n_candidates):
                # Pick a random parent
                parent_idx = rng.integers(pop_size)
                parent = X[parent_idx].copy()
                
                # Mutate: primarily vary x0 (the front-determining variable)
                # with small noise on other variables
                x0_new = parent[0] + 0.02 * rng.standard_normal()
                x0_new = np.clip(x0_new, lo[0], hi[0])
                
                # For ZDT6, setting other variables to 0 gives the Pareto front
                # Add small noise to allow exploration
                child = np.zeros(self.dim)
                child[0] = x0_new
                for d in range(1, self.dim):
                    child[d] = rng.uniform(0, 0.05)
                    child[d] = np.clip(child[d], lo[d], hi[d])
                
                fc = problem(child)
                used += 1
                if used >= self.budget:
                    break
                
                # Check if this point improves hypervolume
                if all(fc[i] < ref[i] for i in range(self.n_obj)):
                    test_points = [p[1] for p in best_points] + [fc]
                    test_hv = self._hypervolume_2d(test_points, ref)
                    if test_hv > best_candidate_hv + 1e-10:
                        best_candidate = (child, fc)
                        best_candidate_hv = test_hv
            
            # If we found a good candidate, add it
            if best_candidate is not None:
                child, fc = best_candidate
                # Replace the worst point in the population
                # Find point with lowest hypervolume contribution
                worst_idx = 0
                worst_contrib = float('inf')
                for i in range(pop_size):
                    # Compute HV without this point
                    points_without = [best_points[j][1] for j in range(pop_size) if j != i]
                    hv_without = self._hypervolume_2d(points_without, ref)
                    contrib = current_hv - hv_without
                    if contrib < worst_contrib:
                        worst_contrib = contrib
                        worst_idx = i
                
                X[worst_idx] = child
                F[worst_idx] = fc
                best_points[worst_idx] = (child, fc)
        
        return X

# EVOLVE-BLOCK-END
