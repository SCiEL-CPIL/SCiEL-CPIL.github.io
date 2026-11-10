# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective optimizer using hypervolume-guided local search for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def _dominates(self, a, b):
        """Return True if a dominates b (a <= b in all, a < b in at least one)."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        nd = [0] * n
        rank = [0] * n
        fronts = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    nd[i] += 1
            if nd[i] == 0:
                rank[i] = 0
                fronts.append([i])
        front_idx = 0
        while front_idx < len(fronts):
            next_front = []
            for i in fronts[front_idx]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        rank[j] = front_idx + 1
                        next_front.append(j)
            front_idx += 1
            if next_front:
                fronts.append(next_front)
        return fronts

    def _hv_contribution(self, F, ref_point, idx):
        """Compute hypervolume contribution of point idx in F with given reference point."""
        # For 3 objectives, compute the volume of the box dominated by F[idx] 
        # that is not dominated by any other point in F
        # Simple approximation: volume of box from F[idx] to ref_point
        # minus volumes of overlapping boxes (crude but fast)
        f = F[idx]
        if np.any(f >= ref_point):
            return 0.0
        
        # Box dimensions
        box = ref_point - f
        
        # Check which other points are within this box (i.e., could overlap)
        # A point j overlaps if f[j] < ref_point and for each dim, f[j][d] > f[idx][d] is not required
        # Actually, the contribution is the volume of {y : f[idx] <= y < ref_point} \ union_{j != idx} {y : f[j] <= y < ref_point}
        
        # For speed, use a simple estimate: if no other point is in the same "region", full box volume
        # Otherwise, reduce by overlap. For 3D, exact is complex. Use a heuristic.
        
        # Check if any other point dominates idx (shouldn't happen if idx is non-dominated)
        # Check if any other point is "close" and overlaps significantly
        
        # Simple approach: count how many points are in the box [f[idx], ref_point)
        # If only idx is there, contribution is full box volume
        # This is an overestimate but monotonic
        
        # Better: use the fact that for non-dominated points, the contribution is the volume
        # of the region where idx is the "best" in at least one objective
        
        # For 3 objectives, the contribution can be computed as:
        # sum over regions defined by sorting other points' coordinates
        
        # For speed, use a simplified version:
        # contribution = box_volume * (1 / (1 + number_of_points_in_box))
        
        # Even simpler: just use box volume as a proxy, since we're using it for selection
        return np.prod(box)

    def __call__(self, problem, bounds):
        deadline = time.time() + 275.0
        dim, n_obj = self.dim, self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        rng = np.random
        budget = self.budget
        
        # Reference point for hypervolume (used for scoring)
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # Population size
        pop_size = min(40, max(15, budget // 6))
        evals = 0
        
        # Initialize population using Latin Hypercube Sampling
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        for d in range(dim):
            bins = np.linspace(0, 1, pop_size + 1)
            pts = rng.uniform(bins[:-1], bins[1:])
            rng.shuffle(pts)
            X[:, d] = lo[d] + pts * span[d]
        
        for i in range(pop_size):
            F[i] = problem(X[i])
            evals += 1
        
        # Keep track of all evaluated points (for final hypervolume)
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        # Main loop: alternate between exploration and exploitation
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Get non-dominated sort
            fronts = self._non_dominated_sort(F)
            
            # Generate offspring: mix of DE mutation and local search
            offspring_X = np.zeros((pop_size, dim))
            offspring_F = np.zeros((pop_size, n_obj))
            
            for i in range(pop_size):
                if evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Strategy 1 (60%): DE mutation from random parents
                # Strategy 2 (40%): Local search around a good point
                
                if rng.random() < 0.6:
                    # DE mutation
                    p1_idx, p2_idx, p3_idx = rng.randint(pop_size, size=3)
                    F_factor = 0.5 + 0.3 * rng.random()
                    child = X[p1_idx] + F_factor * (X[p2_idx] - X[p3_idx])
                    # Add small noise
                    child += 0.05 * span * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)
                else:
                    # Local search: pick a point from first front and perturb
                    if len(fronts) > 0:
                        front0 = fronts[0]
                        base_idx = front0[rng.randint(len(front0))]
                        # Adaptive step size: start large, decrease
                        progress = evals / budget
                        step_size = 0.3 * (1 - 0.7 * progress)
                        child = X[base_idx] + step_size * span * rng.standard_normal(dim)
                        child = np.clip(child, lo, hi)
                    else:
                        child = lo + rng.random(dim) * span
                
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                evals += 1
            
            if evals >= budget:
                break
            
            # Combine and select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort on combined
            fronts = self._non_dominated_sort(combined_F)
            
            # Select next generation using hypervolume contribution as secondary criterion
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            count = 0
            
            for front in fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                else:
                    remaining = pop_size - count
                    # Use hypervolume contribution to break ties
                    hv_contribs = np.array([self._hv_contribution(combined_F, ref_point, idx) for idx in front])
                    order = np.argsort(-hv_contribs)
                    for i in range(remaining):
                        idx = front[order[i]]
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                    break
            
            X, F = new_X, new_F
            all_X.append(X.copy())
            all_F.append(F.copy())
        
        # Collect all unique non-dominated points from all generations
        # For efficiency, just use the final population's first front
        fronts = self._non_dominated_sort(F)
        if len(fronts) > 0:
            best_indices = fronts[0]
            return X[best_indices]
        return X
# EVOLVE-BLOCK-END
