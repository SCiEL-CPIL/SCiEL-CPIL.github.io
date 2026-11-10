# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 using structured Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # For DTLZ1, the Pareto front is a simplex in objective space.
        # We can map points on the simplex to decision variables.
        # DTLZ1: f_i = 0.5 * (1 + g) * prod_{j=1}^{i-1} x_j * (1 - x_i)
        # where g = sum_{j=n-m+1}^{n} x_j^2, m=10 (but dim=7, so m=min(10,dim)=7)
        # For dim=7, m=7, so g = sum_{j=1}^{7} x_j^2
        # The Pareto front has g=0, so all x_j = 0 for j in the g-term.
        # But for DTLZ1 with m=10 and n=7, actually m should be min(10, n-1) or similar.
        # Let me reconsider: standard DTLZ1 has m=10, n variables.
        # f_i = 0.5*(1+g)*prod_{j=1}^{i-1} x_j * (1-x_i)
        # g = sum_{j=n-m+1}^{n} x_j^2
        # For n=7, m=10: n-m+1 = 7-10+1 = -2, which is invalid.
        # Typically m is set to min(10, n) or the problem is defined with m <= n.
        # Given the problem states 7 variables and 3 objectives, and DTLZ1,
        # likely m = min(10, dim) = 7, but then g uses all variables.
        # Actually, for DTLZ1, the standard is m=10, but if n < m, we adjust.
        # Let's assume the problem is set up so that the Pareto front is accessible.
        # The key insight: on the Pareto front, g=0, which means the "tail" variables are 0.
        # The remaining variables (first n-m of them) determine the position on the simplex.
        # For 3 objectives, we need 2 degrees of freedom on the simplex.
        
        # Generate candidate points on the Pareto simplex using a grid
        # The simplex for 3 objectives: x + y + z = 1, x,y,z >= 0
        # In DTLZ1, f_i = 0.5 * prod terms (when g=0)
        # Actually, when g=0, f_i = 0.5 * prod_{j=1}^{i-1} x_j * (1-x_i)
        # For 3 objectives:
        # f1 = 0.5 * x1
        # f2 = 0.5 * x1 * x2
        # f3 = 0.5 * x1 * x2 * x3  ... wait, this doesn't sum to a simplex properly.
        
        # Let me use a different approach: directly sample the objective space simplex
        # and use an inverse mapping or just evaluate and select.
        
        # Strategy: Generate a grid of points on the unit simplex in objective space,
        # then for each candidate objective vector, try to find a decision variable x
        # that achieves it (or close to it), evaluate, and keep the best.
        
        # Simpler: Use a structured search in decision space that targets the Pareto front.
        # For DTLZ1, setting the "g" variables to 0 puts us on the Pareto front.
        # The remaining variables control the position on the simplex.
        
        # For dim=7, n_obj=3: assume m=1 (only last variable is in g), or m=7.
        # Let's try: set variables 1..dim-2 to control simplex, last 2 to 0 (for g=0).
        # Actually, let's just do a smart search:
        
        # Phase 1: Generate a set of candidate points on the Pareto simplex
        # For 3 objectives, the Pareto front is a 2D simplex.
        # Use a triangular grid of resolution ~20x20 = 400 points, but budget is 200.
        # So use ~15x15 grid = 225 points, slightly over budget. Use 14x14 = 196.
        
        # Generate grid points on the simplex
        n_grid = 14  # 14*14 = 196 points, close to budget 200
        candidates = []
        for i in range(n_grid):
            for j in range(n_grid - i):
                x1 = i / (n_grid - 1)
                x2 = j / (n_grid - 1)
                x3 = 1.0 - x1 - x2
                if x3 >= -1e-10:
                    candidates.append([x1, x2, x3])
        
        candidates = np.array(candidates)
        
        # Map simplex coordinates to decision variables for DTLZ1
        # For DTLZ1 with 3 objectives, the relationship is:
        # f1 = 0.5 * x1 (when g=0)
        # f2 = 0.5 * x1 * x2  
        # f3 = 0.5 * x1 * x2 * x3
        # This is not a standard simplex mapping. Let me use a different approach.
        
        # Actually, let's just do a structured random search with exploitation.
        # Phase 1: Random initialization with structured sampling
        # Phase 2: Local refinement around best points
        
        # Use a hybrid approach: start with a coarse grid in decision space,
        # then refine around non-dominated points.
        
        pop_size = 50
        # Generate initial population using Latin hypercube for better coverage
        rng = np.random.RandomState(42)
        
        # Latin hypercube sampling
        LHS = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            LHS[perm, d] = (np.arange(pop_size) + 0.5) / pop_size
        X = lo + (hi - lo) * LHS
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Keep track of all evaluated points
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        # Find non-dominated points
        def non_dominated_mask(F):
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return ~dominated
        
        # Iterative refinement
        while used < budget:
            if time.time() > deadline:
                break
            
            # Get non-dominated points from all evaluations
            all_F_flat = np.vstack(all_F)
            all_X_flat = np.vstack(all_X)
            nd_mask = non_dominated_mask(all_F_flat)
            nd_points = all_X_flat[nd_mask]
            
            if len(nd_points) == 0:
                break
            
            # Select a random non-dominated point and perturb it
            idx = rng.randint(len(nd_points))
            parent = nd_points[idx]
            
            # Adaptive perturbation: start large, decrease over time
            progress = used / budget
            scale = 0.3 * (1.0 - progress) + 0.05
            child = parent + scale * (hi - lo) * rng.randn(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            all_X.append(child.reshape(1, -1))
            all_F.append(fc.reshape(1, -1))
            
            # Occasionally do a pure random restart to escape local optima
            if rng.random() < 0.05 and used < budget - 1:
                rand_x = lo + (hi - lo) * rng.rand(dim)
                rand_f = problem(rand_x)
                used += 1
                all_X.append(rand_x.reshape(1, -1))
                all_F.append(rand_f.reshape(1, -1))
        
        return np.vstack(all_X)
# EVOLVE-BLOCK-END
