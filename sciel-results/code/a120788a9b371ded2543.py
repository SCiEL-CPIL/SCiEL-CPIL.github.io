"""WFG9 optimizer exploiting analytic Pareto front structure with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit WFG9's known Pareto front geometry for dense, well-distributed coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        mid = (lo + hi) / 2.0
        budget = self.budget

        def elapsed():
            return time.time() - start_time

        def time_ok():
            return elapsed() < deadline

        # --- Phase 1: Dense grid over the Pareto front parameter space ---
        # WFG9's Pareto front can be parameterized. For 3 objectives,
        # the front lies in a 2D manifold. We sample a dense grid of
        # parameter values, map them to decision space, and evaluate.
        #
        # WFG9 structure: the Pareto-optimal solutions have specific
        # relationships. For the standard WFG9 with 12 variables:
        # - First ~6 variables control position on the front
        # - Remaining variables should be at their "optimal" values
        #
        # We use a stratified sampling approach in a 2D parameter space.

        # Number of initial grid points (dense but within budget)
        n_grid = min(120, max(40, budget // 2))
        
        # Create a 2D grid of parameters
        # WFG9 front: use two parameters to span the front
        n_per_axis = int(np.ceil(np.sqrt(n_grid)))
        n_grid = n_per_axis * n_per_axis
        
        # Ensure we don't exceed budget
        if n_grid > budget - 20:
            n_grid = max(20, budget - 20)
            n_per_axis = int(np.ceil(np.sqrt(n_grid)))
            n_grid = n_per_axis * n_per_axis

        # Parameter grid: use normalized coordinates in [0,1]^2
        # These map to the Pareto front region
        param_u = np.linspace(0.05, 0.95, n_per_axis)
        param_v = np.linspace(0.05, 0.95, n_per_axis)
        
        # Generate all grid points
        grid_params = []
        for u in param_u:
            for v in param_v:
                grid_params.append((u, v))
        
        # Shuffle to avoid systematic bias in early evaluations
        np.random.shuffle(grid_params)
        
        # Map parameters to decision space
        # WFG9: the Pareto front has specific variable configurations
        # Use a mapping that places points near the front
        X_init = np.zeros((n_grid, self.dim))
        for i, (u, v) in enumerate(grid_params):
            x = self._param_to_decision(u, v, lo, hi, mid, span)
            X_init[i] = x

        # Evaluate initial grid
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = n_grid

        # Keep track of all evaluated points
        all_X = X_init.copy()
        all_F = F_init.copy()

        # --- Phase 2: Local refinement around best points ---
        # Identify the non-dominated points and refine around them
        # to improve hypervolume coverage

        # Find non-dominated points
        nd_indices = self._find_non_dominated(F_init)
        
        # Select top candidates for refinement (diverse, non-dominated)
        # Use hypervolume contribution as a proxy
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # Sort by sum of objectives (lower is better)
        sums = np.sum(F_init, axis=1)
        sorted_idx = np.argsort(sums)
        
        # Take a diverse subset of the best points for refinement
        n_refine = min(len(nd_indices), max(10, (budget - used) // 3))
        
        # Pick diverse points from non-dominated set
        refine_candidates = []
        for idx in sorted_idx:
            if idx in nd_indices and len(refine_candidates) < n_refine:
                # Check diversity: minimum distance to already selected
                if not refine_candidates:
                    refine_candidates.append(idx)
                else:
                    min_dist = min(np.linalg.norm(F_init[idx] - F_init[rc]) for rc in refine_candidates)
                    if min_dist > 0.05:  # ensure diversity
                        refine_candidates.append(idx)
        
        # If not enough diverse points, fill with best remaining
        if len(refine_candidates) < n_refine:
            for idx in sorted_idx:
                if idx not in refine_candidates and len(refine_candidates) < n_refine:
                    refine_candidates.append(idx)

        # Local refinement: small perturbations around each candidate
        remaining_budget = budget - used
        n_candidates = len(refine_candidates)
        if n_candidates > 0 and remaining_budget > 0:
            evals_per_candidate = max(1, remaining_budget // n_candidates)
            
            for cand_idx in refine_candidates:
                if used >= budget or not time_ok():
                    break
                    
                base_x = X_init[cand_idx].copy()
                base_f = F_init[cand_idx].copy()
                
                for _ in range(evals_per_candidate):
                    if used >= budget or not time_ok():
                        break
                    
                    # Small random perturbation
                    scale = 0.05 * span
                    perturbation = np.random.randn(self.dim) * scale
                    x_new = base_x + perturbation
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    used += 1
                    all_X = np.vstack([all_X, x_new.reshape(1, -1)])
                    all_F = np.vstack([all_F, f_new.reshape(1, -1)])
                    
                    # If this is better, update base
                    if np.sum(f_new) < np.sum(base_f):
                        base_x = x_new.copy()
                        base_f = f_new.copy()

        # --- Phase 3: Fill remaining budget with smart random sampling ---
        # Sample in regions where the front is likely to be
        while used < budget and time_ok():
            # Bias toward the center of the box where Pareto-optimal
            # solutions tend to be for WFG9
            x = np.random.uniform(lo, hi, size=self.dim)
            # Blend toward center
            x = 0.7 * x + 0.3 * mid
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            used += 1
            all_X = np.vstack([all_X, x.reshape(1, -1)])
            all_F = np.vstack([all_F, f.reshape(1, -1)])

        return all_X

    def _param_to_decision(self, u, v, lo, hi, mid, span):
        """Map 2D Pareto front parameters to 12-dimensional decision space.
        
        WFG9's Pareto front: the optimal solutions have specific structures.
        For the standard WFG9:
        - The front is characterized by trade-offs between 3 objectives
        - Variables interact in specific ways
        
        We use a heuristic mapping based on the known structure:
        - u controls the first trade-off dimension
        - v controls the second trade-off dimension
        """
        x = np.zeros(self.dim)
        
        # WFG9 structure: first few variables are critical
        # Use sinusoidal and linear mappings to cover the front
        x[0] = lo[0] + span[0] * (0.5 + 0.45 * np.sin(np.pi * u))
        x[1] = lo[1] + span[1] * (0.5 + 0.45 * np.cos(np.pi * u))
        
        if self.dim > 2:
            x[2] = lo[2] + span[2] * (0.5 + 0.45 * np.sin(np.pi * v))
            x[3] = lo[3] + span[3] * (0.5 + 0.45 * np.cos(np.pi * v))
        
        if self.dim > 4:
            x[4] = lo[4] + span[4] * (0.5 + 0.4 * np.sin(np.pi * (u + v) / 2))
            x[5] = lo[5] + span[5] * (0.5 + 0.4 * np.cos(np.pi * (u + v) / 2))
        
        # Remaining variables: set near their "optimal" values
        # For WFG9, these are typically at specific positions
        for i in range(6, self.dim):
            # Alternate between low and high values with some variation
            if i % 2 == 0:
                x[i] = lo[i] + span[i] * (0.3 + 0.1 * np.sin(np.pi * u * (i - 5)))
            else:
                x[i] = lo[i] + span[i] * (0.7 + 0.1 * np.cos(np.pi * v * (i - 5)))
        
        return np.clip(x, lo, hi)

    def _find_non_dominated(self, F):
        """Find indices of non-dominated points."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[i]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        
        return [i for i in range(n) if not dominated[i]]
# EVOLVE-BLOCK-END
