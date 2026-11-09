"""Combined optimizer for DTLZ1: analytic front construction with dense simplex sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits DTLZ1's known Pareto front structure: fix x[2:] to minimize g,
    sample the 2D simplex densely with stratified + jittered sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        dim = self.dim
        budget = self.budget
        used = 0
        results = []

        def eval_point(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            f = problem(x)
            used += 1
            results.append(f)
            return f

        # DTLZ1 structure:
        # g(x) is minimized when x[i] = 0.5 for i >= 2 (the g-variables).
        # With g=0, the Pareto front is:
        #   f1 = 0.5 * x0
        #   f2 = 0.5 * x1
        #   f3 = 0.5 * (1 - x0 - x1)
        # where x0 >= 0, x1 >= 0, x0 + x1 <= 1.
        #
        # Strategy: Use ALL budget to densely sample the 2D simplex
        # {x0 >= 0, x1 >= 0, x0 + x1 <= 1} with x[2:] = 0.5.
        #
        # For hypervolume maximization with reference point [400, 400, 400],
        # we want good coverage of the entire front. Use a combination of:
        # 1. A structured grid on the simplex for uniform coverage
        # 2. Jittered points to break grid artifacts
        # 3. Additional random points if budget allows

        n_total = budget

        # Generate stratified samples on the simplex triangle
        # Use a triangular grid: for i, j >= 0 with i + j <= K
        # Map to x0 = (i + 0.5) / K, x1 = (j + 0.5) / K where (i+0.5)/K + (j+0.5)/K <= 1
        
        # Determine grid resolution
        # Number of points in triangular grid with parameter K is K*(K+1)/2
        # We want roughly n_total points, so K ~ sqrt(2*n_total)
        K = int(np.ceil(np.sqrt(2 * n_total)))
        K = max(2, K)
        
        grid_points = []
        for i in range(K):
            for j in range(K - i):
                # Check if point is inside the simplex
                x0 = (i + 0.5) / K
                x1 = (j + 0.5) / K
                if x0 + x1 <= 1.0:
                    grid_points.append((x0, x1))
        
        # If grid has too many points, subsample; if too few, add random
        if len(grid_points) >= n_total:
            # Shuffle and take n_total
            np.random.shuffle(grid_points)
            sample_points = grid_points[:n_total]
        else:
            # Use all grid points and fill with random points on simplex
            sample_points = list(grid_points)
            n_extra = n_total - len(sample_points)
            for _ in range(n_extra):
                if time.time() > deadline:
                    break
                # Random point on simplex: use Dirichlet(1,1,1) or rejection
                r1 = np.random.random()
                r2 = np.random.random()
                # Transform to simplex: x0 = r1, x1 = r2 * (1 - r1)
                x0 = r1
                x1 = r2 * (1.0 - x0)
                sample_points.append((x0, x1))
        
        # Add slight jitter to grid points to avoid exact degeneracies
        # and to better approximate continuous coverage
        jittered_points = []
        for x0, x1 in sample_points:
            # Small random perturbation
            j0 = np.random.uniform(-0.001, 0.001)
            j1 = np.random.uniform(-0.001, 0.001)
            x0j = np.clip(x0 + j0, 0.0, 1.0)
            x1j = np.clip(x1 + j1, 0.0, 1.0)
            # Ensure still on simplex
            if x0j + x1j > 1.0:
                scale = 1.0 / (x0j + x1j)
                x0j *= scale
                x1j *= scale
            jittered_points.append((x0j, x1j))
        
        # Evaluate all points
        for x0, x1 in jittered_points:
            if used >= budget or time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = x0
            x[1] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)
        
        # If we still have budget (shouldn't normally happen), add more random points
        while used < budget and time.time() < deadline:
            r1 = np.random.random()
            r2 = np.random.random()
            x0 = r1
            x1 = r2 * (1.0 - x0)
            x = np.full(dim, 0.5)
            x[0] = x0
            x[1] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return results
# EVOLVE-BLOCK-END
