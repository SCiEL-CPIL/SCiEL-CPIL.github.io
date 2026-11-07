"""Optimized program for WFG9 using MOEA/D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """MOEA/D-based Multi-Objective Optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Generate reference points
        n_sub = 30 # Number of subproblems
        ref_points = self._generate_ref_points(n_sub, self.n_obj)
        
        # Build neighborhood
        # For each reference point, find the T closest neighbors
        T = 5 # Neighborhood size
        neighbors = self._build_neighbors(ref_points, T)
        
        # Initialize population
        X = np.zeros((n_sub, self.dim))
        F = np.zeros((n_sub, self.n_obj))
        
        for i in range(n_sub):
            X[i] = lo + range_ * np.random.rand(self.dim)
            F[i] = problem(X[i])
            
        evaluations = n_sub
        
        # External Archive
        archive_X = []
        archive_F = []
        
        # Add initial population to archive if non-dominated
        for i in range(n_sub):
            if self._is_non_dominated(F[i], archive_F):
                archive_X.append(X[i])
                archive_F.append(F[i])
        
        # DE Parameters
        CR = 0.9
        F_DE = 0.8
        
        # Ideal point for Tchebycheff
        ideal_point = np.min(F, axis=0)
        
        # Main Loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            # Select a subproblem index
            # Use a roulette wheel or random selection. Random is fine for small budgets.
            # To ensure exploration, we can cycle through indices or pick randomly.
            # Let's pick randomly to avoid bias towards specific order.
            idx = np.random.randint(n_sub)
            
            # Select parents from neighborhood
            # Pick 2 or 3 parents from neighbors[idx]
            # If neighborhood is too small, pick from whole population
            n_avail = len(neighbors[idx])
            if n_avail < 3:
                parents_idx = np.random.choice(n_sub, 3, replace=False)
            else:
                parents_idx = np.random.choice(neighbors[idx], 3, replace=False)
            
            p1 = X[parents_idx[0]]
            p2 = X[parents_idx[1]]
            p3 = X[parents_idx[2]]
            
            # DE Mutation
            v = p1 + F_DE * (p2 - p3)
            
            # Crossover
            current = X[idx]
            child = np.zeros_like(v)
            j_rand = np.random.randint(self.dim)
            for i in range(self.dim):
                if np.random.rand() < CR or i == j_rand:
                    child[i] = v[i]
                else:
                    child[i] = current[i]
            
            # Clip
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            evaluations += 1
            
            # Update Ideal Point
            ideal_point = np.minimum(ideal_point, fc)
            
            # Update External Archive
            if self._is_non_dominated(fc, archive_F):
                # Remove dominated points from archive
                new_archive_X = []
                new_archive_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not self._dominates(fc, af):
                        new_archive_X.append(ax)
                        new_archive_F.append(af)
                archive_X = new_archive_X
                archive_F = new_archive_F
                archive_X.append(child)
                archive_F.append(fc)
            
            # Update Subproblem Solution
            # Tchebycheff Scalarization
            w = ref_points[idx]
            curr_fit = self._tchebycheff(F[idx], w, ideal_point)
            child_fit = self._tchebycheff(fc, w, ideal_point)
            
            if child_fit < curr_fit:
                X[idx] = child
                F[idx] = fc
            else:
                # If not better, we might still want to update if it improves diversity?
                # Standard MOEA/D only updates if scalarized fitness improves.
                # However, to prevent stagnation, we could accept with small probability if it's close?
                # Let's stick to strict improvement for now.
                pass
                
        # Return the archive or the population?
        # The evaluator records ALL returned objective vectors.
        # The return value of __call__ is ignored.
        # Wait, "Every returned objective vector is recorded".
        # This implies I should return a list/array of objective vectors?
        # "The return value of __call__ is ignored."
        # "Every returned objective vector is recorded" -> This phrasing is tricky.
        # Usually, in these tasks, the `problem` function is a closure that records calls.
        # "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded" -> This likely refers to the outputs of `problem(x)` calls made by the optimizer.
        # "The return value of __call__ is ignored." -> So I don't need to return the best solutions.
        # But wait, if the return value is ignored, why does it say "Every returned objective vector is recorded"?
        # Ah, "recorded" means the evaluator keeps track of all `f(x)` results generated during the run.
        # The scoring is based on "all recorded vectors".
        # So I just need to make sure I call `problem` enough times to get a good set of points.
        # I do not need to return anything specific, but returning None or a dummy is fine.
        
        return X

    def _generate_ref_points(self, n_points, n_obj):
        """Generate n_points uniformly distributed on the (n_obj-1) simplex."""
        pts = np.random.rand(n_points, n_obj)
        sums = pts.sum(axis=1, keepdims=True)
        pts = pts / sums
        return pts

    def _build_neighbors(self, ref_points, T):
        """Build neighbor list for each reference point."""
        n = len(ref_points)
        neighbors = []
        # Compute pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dists[i, j] = np.linalg.norm(ref_points[i] - ref_points[j])
        
        for i in range(n):
            # Get indices sorted by distance
            idxs = np.argsort(dists[i])
            # Exclude self
            idxs = idxs[idxs != i]
            # Take T closest
            neighbors.append(list(idxs[:T]))
        return neighbors

    def _tchebycheff(self, f, w, z):
        """Tchebycheff scalarization."""
        return np.max(w * np.abs(f - z))

    def _dominates(self, a, b):
        """Check if a dominates b (a is better or equal in all, strictly better in at least one)."""
        if np.all(a <= b) and np.any(a < b):
            return True
        return False

    def _is_non_dominated(self, f, archive_F):
        """Check if f is non-dominated by any point in archive_F."""
        for af in archive_F:
            if self._dominates(af, f):
                return False
        return True
# EVOLVE-BLOCK-END
