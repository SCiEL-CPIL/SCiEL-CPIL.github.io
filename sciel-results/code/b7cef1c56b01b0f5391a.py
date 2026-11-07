"""Multi-objective optimizer on RE37 using space-filling initialization and farthest-point sampling in objective space."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer that:
    1. Uses a deterministic Halton sequence for initial space-filling design.
    2. Iteratively selects the next point by maximizing the minimum Euclidean distance
       to all previously selected "anchor" points in the objective space.
    3. Evaluates a batch of candidates per iteration to allow selection.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, dim, skip=1):
        """Generate n points in [0,1)^dim using the Halton sequence, skipping the first `skip` points."""
        points = []
        for i in range(skip, skip + n):
            point = []
            for base in range(2, dim + 2):  # primes: 2, 3, 5, 7, ...
                f = 1.0
                r = 0.0
                n_val = i
                while n_val > 0:
                    f /= base
                    r += f * (n_val % base)
                    n_val //= base
                point.append(r)
            points.append(point)
        return np.array(points)

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Phase 1: Halton sequence initialization
        # Use a smaller initial set to leave more budget for adaptive exploration
        n_init = min(self.budget, 20)
        
        # Generate Halton points in [0, 1)^dim
        halton_unit = self._halton_sequence(n_init, self.dim, skip=1)
        
        # Map to bounds
        X_init = lo + (hi - lo) * halton_unit
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # List of "anchor" objective vectors to use for distance calculation
        # These are the points we explicitly "selected" via FPS logic
        selected_F = F_init.copy()
        
        # We don't strictly need to keep X for return, but let's keep track of all evaluated X
        # The return value is ignored, but we must not exceed budget.
        
        if used >= self.budget:
            return X_init

        # Phase 2: Iterative Farthest-Point Sampling
        # We evaluate a batch of candidates, pick the one farthest from selected_F,
        # and add it to selected_F. The other candidates are also recorded (good for coverage)
        # but do not become anchors immediately, allowing the algorithm to jump to new regions.
        
        batch_size = 10
        
        while used < self.budget:
            # Determine how many candidates we can evaluate
            n_candidates = min(batch_size, self.budget - used)
            
            # Generate random candidates
            candidates = lo + (hi - lo) * rng.random((n_candidates, self.dim))
            
            # Evaluate all candidates
            cand_F = np.array([problem(x) for x in candidates])
            used += n_candidates
            
            if len(selected_F) > 0:
                # Compute minimum distance from each candidate to any selected_F point
                # selected_F shape: (N_sel, n_obj)
                # cand_F shape: (n_candidates, n_obj)
                
                # To avoid memory issues if N_sel is large, we can compute in chunks, 
                # but N_sel grows slowly (by 1 per iteration), so it's small.
                # Distance matrix: (n_candidates, N_sel)
                # dist[i, j] = ||cand_F[i] - selected_F[j]||
                
                # Using broadcasting
                # diff = cand_F[:, np.newaxis, :] - selected_F[np.newaxis, :, :]
                # dists = np.linalg.norm(diff, axis=2)
                # min_dists = np.min(dists, axis=1)
                
                # Efficient computation for small dimensions
                min_dists = np.full(n_candidates, np.inf)
                for i in range(n_candidates):
                    dists = np.linalg.norm(selected_F - cand_F[i], axis=1)
                    min_dists[i] = np.min(dists)
                
                # Pick the candidate with the maximum minimum distance
                best_idx = np.argmax(min_dists)
                best_f = cand_F[best_idx]
            else:
                best_idx = 0
                best_f = cand_F[best_idx]

            # Add the best candidate to the anchor set
            selected_F = np.vstack([selected_F, best_f])

        return X_init # Return value is ignored, just a placeholder
# EVOLVE-BLOCK-END
