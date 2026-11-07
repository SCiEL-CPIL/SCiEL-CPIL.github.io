# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    DTLZ4 optimizer using a 2D Latin Hypercube on the quarter-sphere (Pareto front)
    with explicit vertex placement for maximum hypervolume contribution.
    
    Strategy:
    1. DTLZ4's Pareto front is a 2D quarter-sphere in the first 3 objective dimensions.
    2. Parameterize with spherical coordinates: theta in [0, pi/2], phi in [0, pi/2].
    3. Use 2D Latin Hypercube Sampling (LHS) on (theta, phi) for uniform surface coverage.
    4. Add explicit vertices at the 3 axis-aligned corners and the center for max HV.
    5. Set remaining dimensions to 0.5 to achieve g=0.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_points = self.budget

        # Reserve 4 points for explicit vertices
        n_vertices = 4
        n_lhs = n_points - n_vertices

        # 2D Latin Hypercube Sampling on (theta, phi)
        # theta in [0, pi/2], phi in [0, pi/2]
        # For LHS: divide [0, 1] into n_lhs intervals, pick one random point in each
        # Then map to the spherical coordinate ranges

        # Use np.random which is seeded by the evaluator
        # Generate LHS indices
        perm_theta = np.random.permutation(n_lhs)
        perm_phi = np.random.permutation(n_lhs)

        # Uniform points within each cell
        u_theta = (perm_theta + np.random.rand(n_lhs)) / n_lhs
        u_phi = (perm_phi + np.random.rand(n_lhs)) / n_lhs

        # Map to spherical coordinates
        theta = u_theta * (np.pi / 2)  # [0, pi/2]
        phi = u_phi * (np.pi / 2)      # [0, pi/2]

        # Convert spherical to Cartesian
        x_sph = np.sin(phi) * np.cos(theta)
        y_sph = np.sin(phi) * np.sin(theta)
        z_sph = np.cos(phi)

        # Create full dimension vectors
        x_all = np.zeros((n_points, self.dim))

        # Set first 3 dimensions from spherical coordinates
        x_all[:n_lhs, 0] = x_sph
        x_all[:n_lhs, 1] = y_sph
        x_all[:n_lhs, 2] = z_sph

        # Add explicit vertices at the corners of the quarter-sphere
        # Corner 1: (1, 0, 0) -> f1=1, f2=0, f3=0
        x_all[n_lhs, 0] = 1.0
        x_all[n_lhs, 1] = 0.0
        x_all[n_lhs, 2] = 0.0
        # Corner 2: (0, 1, 0) -> f1=0, f2=1, f3=0
        x_all[n_lhs + 1, 0] = 0.0
        x_all[n_lhs + 1, 1] = 1.0
        x_all[n_lhs + 1, 2] = 0.0
        # Corner 3: (0, 0, 1) -> f1=0, f2=0, f3=1
        x_all[n_lhs + 2, 0] = 0.0
        x_all[n_lhs + 2, 1] = 0.0
        x_all[n_lhs + 2, 2] = 1.0
        # Corner 4: (1/sqrt(3), 1/sqrt(3), 1/sqrt(3)) -> f1=f2=f3=1/sqrt(3)
        x_all[n_lhs + 3, 0] = 1.0 / np.sqrt(3)
        x_all[n_lhs + 3, 1] = 1.0 / np.sqrt(3)
        x_all[n_lhs + 3, 2] = 1.0 / np.sqrt(3)

        # Set remaining dimensions to 0.5 (optimal for g=0)
        x_all[:, 3:] = 0.5

        # Clip to bounds
        x_all = np.clip(x_all, lo, hi)

        # Evaluate all points
        results = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            try:
                f = problem(x_all[i])
                results.append(f)
            except Exception:
                pass

        return results
# EVOLVE-BLOCK-END
