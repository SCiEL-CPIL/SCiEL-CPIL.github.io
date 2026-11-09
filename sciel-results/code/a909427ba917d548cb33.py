"""Multi-objective optimizer on RE34 using deterministic Latin Hypercube Sampling."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic LHS-based multi-objective search for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Generate a pool of LHS points for optimization
        # We generate more points than needed to allow for greedy maximin selection
        pool_size = max(budget * 10, 1000)
        rng = np.random.default_rng(42)
        
        # Generate LHS in [0, 1]^dim
        lhs_pool = np.zeros((pool_size, dim))
        for d in range(dim):
            intervals = np.arange(pool_size) / pool_size
            offsets = rng.random(pool_size)
            lhs_pool[:, d] = intervals + offsets * (1.0 / pool_size)
            lhs_pool[:, d] = lhs_pool[rng.permutation(pool_size), d]

        # Map to actual bounds
        X_pool = lo + lhs_pool * span
        X_pool = np.clip(X_pool, lo, hi)

        # Phase 2: Greedy Maximin Selection
        # Compute pairwise squared distances in normalized space [0,1]
        norms = np.sum(lhs_pool**2, axis=1)
        dots = lhs_pool @ lhs_pool.T
        dist2 = norms[:, None] + norms[None, :] - 2 * dots
        dist2 = np.maximum(dist2, 0)
        
        selected_indices = []
        min_dists = np.full(pool_size, np.inf)
        
        # Start with the first point
        current_idx = 0
        selected_indices.append(current_idx)
        min_dists = dist2[current_idx, :].copy()
        
        for _ in range(budget - 1):
            # Find the point with the maximum minimum distance
            next_idx = np.argmax(min_dists)
            selected_indices.append(next_idx)
            # Update minimum distances
            min_dists = np.minimum(min_dists, dist2[next_idx, :])
            
        # Select the points
        X_selected = X_pool[selected_indices]
        
        # Evaluate all selected points
        F_selected = np.array([problem(x) for x in X_selected])

        # Filter out non-finite results
        valid_mask = np.all(np.isfinite(F_selected), axis=1)
        if not np.all(valid_mask):
            F_selected = F_selected[valid_mask]
            X_selected = X_selected[valid_mask]

        return X_selected
# EVOLVE-BLOCK-END
