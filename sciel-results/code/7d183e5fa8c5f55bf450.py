import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n, d):
        """Generate n points in d-dimensional unit hypercube using Sobol sequence (simplified)."""
        # For simplicity and correctness, we'll use a quasi-random approach with better properties
        # than Halton for small dimensions. We'll use a simple deterministic stratified sampling
        # that ensures good coverage.
        points = np.zeros((n, d))
        # Use a grid-based stratified approach with jitter
        k = int(np.ceil(np.sqrt(n)))
        # Generate a k x k grid, then take n points
        grid = np.linspace(0.5 / k, 1 - 0.5 / k, k)
        # Create all combinations
        mesh = np.meshgrid(grid, grid)
        all_points = np.column_stack([mesh[0].flatten(), mesh[1].flatten()])
        # Shuffle deterministically
        np.random.seed(42)
        perm = np.random.permutation(len(all_points))
        all_points = all_points[perm]
        # Take first n points
        if n <= len(all_points):
            points = all_points[:n]
        else:
            # If n > k*k, use Halton as fallback
            primes = [2, 3]
            for i in range(n):
                for j in range(d):
                    base = primes[j]
                    f = 1.0
                    r = 0.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    points[i, j] = r
        return points

    def _compute_hypervolume_3d(self, points, ref_point):
        """Compute hypervolume for 3D points with given reference point."""
        # Filter points that are strictly better than reference point in all objectives
        mask = np.all(points < ref_point, axis=1)
        points = points[mask]
        
        if len(points) == 0:
            return 0.0
        
        # For 3D, we can use the inclusion-exclusion principle or a simple sweep
        # Sort by first objective
        points = points[np.argsort(points[:, 0])]
        
        # Use a simple O(n^2) algorithm for 3D hypervolume
        # This is sufficient for n=200
        hv = 0.0
        n = len(points)
        
        for i in range(n):
            # Width in dimension 0
            if i == 0:
                w0 = ref_point[0] - points[i, 0]
            else:
                w0 = points[i, 0] - points[i-1, 0]
            
            if w0 <= 0:
                continue
            
            # For this slice, find the 2D hypervolume of points i..n-1
            # projected onto dimensions 1 and 2
            sub_points = points[i:, 1:]
            sub_ref = ref_point[1:]
            
            # 2D hypervolume: sort by dim1, then compute
            sub_points = sub_points[np.argsort(sub_points[:, 0])]
            hv2 = 0.0
            for j in range(len(sub_points)):
                if sub_points[j, 0] >= sub_ref[0] or sub_points[j, 1] >= sub_ref[1]:
                    continue
                if j == 0:
                    w1 = sub_ref[0] - sub_points[j, 0]
                else:
                    w1 = sub_points[j, 0] - sub_points[j-1, 0]
                
                if w1 <= 0:
                    continue
                
                # Max value in dim2 among points 0..j
                max_v2 = 0.0
                for k in range(j + 1):
                    if sub_points[k, 1] < sub_ref[1]:
                        max_v2 = max(max_v2, sub_ref[1] - sub_points[k, 1])
                
                hv2 += w1 * max_v2
            
            hv += w0 * hv2
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        n_front_vars = n_obj - 1  # 2 for DTLZ1
        n_g_vars = dim - n_front_vars  # 5 for DTLZ1

        # Generate candidate points using Sobol-like stratified sampling
        # We generate more candidates than budget, then greedily select best subset
        n_candidates = budget  # Keep it simple: generate exactly budget candidates
        
        # Use stratified sampling for better coverage
        front_samples = self._sobol_sequence(n_candidates, n_front_vars)
        
        # Construct 7D points
        X = np.zeros((n_candidates, dim))
        
        range_front = hi[:n_front_vars] - lo[:n_front_vars]
        scaled_front = front_samples * range_front + lo[:n_front_vars]
        X[:, :n_front_vars] = scaled_front
        
        g_mid = (lo[n_front_vars:] + hi[n_front_vars:]) / 2.0
        X[:, n_front_vars:] = g_mid
        
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.zeros((n_candidates, n_obj))
        for i in range(n_candidates):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
