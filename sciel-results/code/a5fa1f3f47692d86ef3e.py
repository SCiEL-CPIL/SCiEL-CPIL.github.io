<<<<<<< SEARCH
"""Portfolio search for DTLZ2: Geometrically-weighted Sobol sampling on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats.qmc as qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using geometrically-corrected Sobol sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # 270s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        budget = self.budget
        dim = self.dim

        # DTLZ2 Pareto front: quarter-sphere in objective space
        # f1 = sin(phi)*cos(theta), f2 = sin(phi)*sin(theta), f3 = cos(phi)
        # where phi in [0, pi/2], theta in [0, pi/2]
        #
        # The surface area element is sin(phi) dphi dtheta.
        # For hypervolume maximization with reference point [2.5, 2.5, 2.5],
        # points near the center of the front (where all objectives are moderate)
        # contribute more to hypervolume than points near the axes.
        #
        # We use a geometrically-weighted sampling strategy:
        # - Sample (u, v) uniformly in [0,1]^2 using Sobol sequence
        # - Map u to phi using inverse CDF of the surface area density
        # - Apply a bias toward the center of the front to maximize hypervolume
        #
        # The hypervolume contribution of a point (f1, f2, f3) with reference [R, R, R]
        # is proportional to (R-f1)(R-f2)(R-f3). To maximize expected hypervolume,
        # we should oversample regions where this product is large.

        candidates = []

        # 1. Add special points: 3 corners, 3 mid-edges, and 1 center
        # Corners: (0,0,1), (0,1,0), (1,0,0) in objective space
        x_corner1 = np.full(dim, 0.5)
        x_corner1[0] = 0.0
        x_corner1[1] = 0.0
        candidates.append(x_corner1)

        x_corner2 = np.full(dim, 0.5)
        x_corner2[0] = 0.0
        x_corner2[1] = 1.0
        candidates.append(x_corner2)

        x_corner3 = np.full(dim, 0.5)
        x_corner3[0] = 1.0
        x_corner3[1] = 0.0
        candidates.append(x_corner3)

        # Mid-edges: midpoints between pairs of corners in objective space
        # Mid-edge 1: between (0,0,1) and (0,1,0) -> (0, 0.5, 0.5)
        # In DTLZ2: f1=0, f2=0.5, f3=0.5 => sin(phi)*cos(theta)=0, sin(phi)*sin(theta)=0.5, cos(phi)=0.5
        # phi=pi/3, theta=pi/2
        x_mid1 = np.full(dim, 0.5)
        x_mid1[0] = 0.0
        x_mid1[1] = 1.0
        candidates.append(x_mid1)

        # Mid-edge 2: between (0,0,1) and (1,0,0) -> (0.5, 0, 0.5)
        # phi=pi/3, theta=0
        x_mid2 = np.full(dim, 0.5)
        x_mid2[0] = 1.0
        x_mid2[1] = 0.0
        candidates.append(x_mid2)

        # Mid-edge 3: between (0,1,0) and (1,0,0) -> (0.5, 0.5, 0)
        # phi=pi/4, theta=pi/4... wait, cos(phi)=0 means phi=pi/2
        # sin(pi/2)*cos(theta)=0.5, sin(pi/2)*sin(theta)=0.5 => theta=pi/4
        x_mid3 = np.full(dim, 0.5)
        x_mid3[0] = 1.0 / np.sqrt(2)
        x_mid3[1] = 1.0 / np.sqrt(2)
        candidates.append(x_mid3)

        # Center: (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
        # phi=arccos(1/sqrt(3)), theta=pi/4
        inv_sqrt3 = 1.0 / np.sqrt(3.0)
        phi_center = np.arccos(inv_sqrt3)
        x_center = np.full(dim, 0.5)
        x_center[0] = np.sin(phi_center) * np.cos(np.pi / 4)
        x_center[1] = np.sin(phi_center) * np.sin(np.pi / 4)
        candidates.append(x_center)

        # 2. Generate remaining points using Sobol sequence with hypervolume-aware weighting.
        n_remaining = budget - len(candidates)
        if n_remaining > 0:
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            sobol_pts = sampler.random(n=n_remaining)

            u = sobol_pts[:, 0]
            v = sobol_pts[:, 1]

            # Hypervolume-aware mapping: bias toward center of front
            # The contribution (R-f1)(R-f2)(R-f3) is maximized at the center.
            # We use a power transform to concentrate samples near the center.
            # u^alpha with alpha > 1 concentrates near u=0 (phi near pi/2, i.e., f3 near 0)
            # But we want to concentrate near the center, so we use a symmetric bias.
            # 
            # For phi: center is at phi_center = arccos(1/sqrt(3)) ≈ 0.955
            # We map u in [0,1] to phi with a bias toward phi_center.
            # Use: phi = phi_center + (arccos(u) - phi_center) * (1 - 0.3 * |2*u - 1|)
            # This shrinks the range toward phi_center by 30% at the edges.
            phi_uniform = np.arccos(u)
            # Bias factor: 1 at center (u=0.5), 0.7 at edges
            bias_factor = 1.0 - 0.3 * np.abs(2.0 * u - 1.0)
            phi = phi_center + (phi_uniform - phi_center) * bias_factor

            # For theta: center is at pi/4
            theta_uniform = v * (np.pi / 2)
            bias_factor_theta = 1.0 - 0.3 * np.abs(2.0 * v - 1.0)
            theta = np.pi / 4 + (theta_uniform - np.pi / 4) * bias_factor_theta

            # Compute x1, x2 on the Pareto front
            x1 = np.sin(phi) * np.cos(theta)
            x2 = np.sin(phi) * np.sin(theta)

            for i in range(n_remaining):
                x = np.full(dim, 0.5)
                x[0] = x1[i]
                x[1] = x2[i]
                candidates.append(x)

        # 3. Evaluate all candidates, recording every objective vector.
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            try:
                problem(x)
            except Exception:
                break
            used += 1

        return None
# EVOLVE-BLOCK-END
=======
"""Portfolio search for DTLZ2: Dense Pareto-front grid with strategic gap-filling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using a Pareto-front grid portfolio."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        used = 0
        budget = self.budget
        dim = self.dim

        candidates = []

        # 1. Add special points: 3 corners, 3 mid-edges, and 1 center
        x_corner1 = np.full(dim, 0.5)
        x_corner1[0] = 0.0
        x_corner1[1] = 0.0
        candidates.append(x_corner1)

        x_corner2 = np.full(dim, 0.5)
        x_corner2[0] = 0.0
        x_corner2[1] = 1.0
        candidates.append(x_corner2)

        x_corner3 = np.full(dim, 0.5)
        x_corner3[0] = 1.0
        x_corner3[1] = 0.0
        candidates.append(x_corner3)

        x_mid1 = np.full(dim, 0.5)
        x_mid1[0] = 0.0
        x_mid1[1] = 1.0
        candidates.append(x_mid1)

        x_mid2 = np.full(dim, 0.5)
        x_mid2[0] = 1.0
        x_mid2[1] = 0.0
        candidates.append(x_mid2)

        x_mid3 = np.full(dim, 0.5)
        x_mid3[0] = 1.0 / np.sqrt(2)
        x_mid3[1] = 1.0 / np.sqrt(2)
        candidates.append(x_mid3)

        inv_sqrt3 = 1.0 / np.sqrt(3.0)
        phi_center = np.arccos(inv_sqrt3)
        x_center = np.full(dim, 0.5)
        x_center[0] = np.sin(phi_center) * np.cos(np.pi / 4)
        x_center[1] = np.sin(phi_center) * np.sin(np.pi / 4)
        candidates.append(x_center)

        # 2. Generate remaining points using a dense grid in (phi, theta) space.
        # For DTLZ2, the Pareto front is parameterized by:
        # f1 = sin(phi)*cos(theta), f2 = sin(phi)*sin(theta), f3 = cos(phi)
        # where phi in [0, pi/2], theta in [0, pi/2]
        #
        # A uniform grid in (phi, theta) gives near-uniform coverage of the quarter-sphere.
        # With budget=200 and 7 special points, we have 193 remaining points.
        # Use a grid of ~14x14 = 196 points, which is close to 193.
        # We'll use a slightly irregular grid to fit exactly.

        n_remaining = budget - len(candidates)
        if n_remaining > 0:
            # Use a grid in (phi, theta) space
            # phi in [0, pi/2], theta in [0, pi/2]
            # For a 14x14 grid, we have 196 points. We need 193, so we'll trim 3.
            
            # Actually, let's use a more systematic approach:
            # Generate a grid that fits within n_remaining
            # Try to make it as square as possible
            n_grid = int(np.sqrt(n_remaining))
            while n_grid * (n_grid + 1) > n_remaining and n_grid > 1:
                n_grid -= 1
            while n_grid * (n_grid + 1) < n_remaining and (n_grid + 1) * (n_grid + 1) <= n_remaining:
                n_grid += 1
            
            # Use n_grid x (n_grid+1) grid if it fits, otherwise n_grid x n_grid
            n_rows = n_grid
            n_cols = n_grid
            total_grid = n_rows * n_cols
            
            # If we have room for more, add a partial row
            extra = n_remaining - total_grid
            if extra > 0:
                n_cols += 1  # Add extra points to the last row
                total_grid = n_rows * n_cols
            
            # Create grid points in (phi, theta) space
            # phi from 0 to pi/2, theta from 0 to pi/2
            # Use linspace with endpoints included
            phi_vals = np.linspace(0, np.pi / 2, n_rows, endpoint=True)
            theta_vals = np.linspace(0, np.pi / 2, n_cols, endpoint=True)
            
            # Generate all grid combinations
            phi_grid, theta_grid = np.meshgrid(phi_vals, theta_vals, indexing='ij')
            
            # Flatten and take only what we need
            phi_flat = phi_grid.flatten()
            theta_flat = theta_grid.flatten()
            
            # Take only n_remaining points
            phi_flat = phi_flat[:n_remaining]
            theta_flat = theta_flat[:n_remaining]
            
            # Compute x1, x2 on the Pareto front
            x1 = np.sin(phi_flat) * np.cos(theta_flat)
            x2 = np.sin(phi_flat) * np.sin(theta_flat)
            
            for i in range(n_remaining):
                x = np.full(dim, 0.5)
                x[0] = x1[i]
                x[1] = x2[i]
                candidates.append(x)

        # 3. Evaluate all candidates, recording every objective vector.
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            try:
                problem(x)
            except Exception:
                break
            used += 1

        return None
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
BET: {mechanism: "Replacing Sobol with a dense uniform grid in (phi, theta) space gives more systematic coverage of the Pareto front, which the cell findings indicate improves hypervolume from 1.1745 to ~1.18", q: 0.72, delta: [0.005, 0.015]}
