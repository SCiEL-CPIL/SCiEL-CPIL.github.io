"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import time


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses a multi-start non-linear least-squares optimization with
    cKDTree-accelerated nearest-neighbor constraints.
    """
    best_centers = None
    best_score = 0.0
    start_time = time.time()
    time_budget = 110.0  # Leave 10s buffer for safety

    # Target radius from best known packing
    target_r = 0.0531542991469

    def score_of_centers(c):
        """Compute the maximum common radius for given centers."""
        if c.shape[0] != N:
            return 0.0
        if np.any(~np.isfinite(c)):
            return 0.0
        # Distance from each center to boundary
        dists_to_boundary = 1.0 - np.linalg.norm(c, axis=1)
        min_boundary = np.min(dists_to_boundary)
        if min_boundary <= 0:
            return 0.0
        # Minimum pairwise distance / 2
        tree = cKDTree(c)
        # Query nearest neighbor for each point (excluding self)
        dists, _ = tree.query(c, k=2)
        min_pair_dist = np.min(dists[:, 1])
        r = min(min_pair_dist / 2.0, min_boundary)
        return r if r > 0 else 0.0

    def make_residuals(c_flat, margin=0.0):
        """
        Residuals for least_squares.
        We want to maximize the minimum of:
          - pairwise distances / 2
          - distance to boundary (1 - |c_i|)
        
        We formulate this as minimizing violations of a target radius r_target.
        Residuals are:
          - For each pair (i,j) that is "close" (within some threshold): 
            max(0, r_target - dist(i,j)/2)
          - For each point i: max(0, r_target - (1 - |c_i|))
        
        But since we don't know which pairs are close, we use a soft approach:
        Instead, we minimize the sum of squared violations where violation is
        defined as how much the constraint is violated relative to a running
        estimate of the best radius.
        
        Actually, a cleaner approach: minimize the negative of the score, but
        that's not differentiable. So we use the following trick:
        
        We define residuals as:
          For each point i: r_target - (1 - |c_i|)  (if negative, it's a violation)
          For each pair (i,j): r_target - |c_i - c_j|/2  (if negative, violation)
        
        But we only include pairs that are close enough. To make it efficient,
          we use cKDTree to find neighbors within a certain distance.
        
        However, least_squares expects a fixed number of residuals or a function
        that returns an array. Let's use a dynamic approach: we'll compute
        residuals based on current positions, finding nearby pairs.
        
        For efficiency, we'll only consider pairs within a distance of, say,
        3 * r_target, since pairs farther apart won't be the binding constraint.
        """
        c = c_flat.reshape(-1, 2)
        
        # Boundary residuals
        dists_to_boundary = 1.0 - np.linalg.norm(c, axis=1)
        boundary_res = r_target - dists_to_boundary  # positive means violation
        
        # Pairwise residuals: only for nearby pairs
        tree = cKDTree(c)
        # Find all pairs within distance 4 * r_target
        pairs = tree.query_pairs(4.0 * r_target)
        
        pair_res = []
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            violation = r_target - d / 2.0
            if violation > 0:
                pair_res.append(violation)
        
        pair_res = np.array(pair_res) if pair_res else np.array([0.0])
        
        # Combine: scale boundary residuals to be comparable
        # We want to minimize the maximum violation, so we square them
        # and sum. Use a weighted combination.
        residuals = np.concatenate([boundary_res, pair_res])
        return residuals

    # Multi-start optimization
    n_starts = 8
    rng = np.random.default_rng(42)
    
    for start_idx in range(n_starts):
        if time.time() - start_time > time_budget:
            break
        
        # Generate initial configuration
        if start_idx == 0:
            # Hexagonal lattice initialization
            spacing = 2.0 / (np.sqrt(N) + 1.0)
            while True:
                k = int(1.0 / spacing) + 2
                rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                        for i in range(-k, k + 1) for j in range(-k, k + 1)]
                points = np.array(rows)
                radius = np.hypot(points[:, 0], points[:, 1])
                mask = radius <= 1.0 - spacing / 2.0
                points = points[mask]
                if len(points) >= N:
                    # Sort by radius and take N
                    idx = np.argsort(radius, kind="stable")[:N]
                    c0 = points[idx].copy()
                    break
                spacing *= 0.98
        elif start_idx == 1:
            # Random initialization inside unit circle
            angles = rng.uniform(0, 2 * np.pi, N)
            radii = rng.uniform(0, 0.95, N)
            c0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        elif start_idx == 2:
            # Concentric rings
            c0 = np.zeros((N, 2))
            ring_sizes = []
            remaining = N
            ring_radius = 0.0
            ring_idx = 0
            while remaining > 0:
                if ring_radius == 0:
                    # One point at center
                    if remaining >= 1:
                        ring_sizes.append(1)
                        remaining -= 1
                        ring_radius = 0.3
                    else:
                        break
                else:
                    # Number of points on this ring
                    circumference = 2 * np.pi * ring_radius
                    # Estimate spacing
                    n_on_ring = min(remaining, max(2, int(circumference / (2 * 0.05))))
                    ring_sizes.append(n_on_ring)
                    remaining -= n_on_ring
                    ring_radius += 0.08
            # Distribute points
            idx = 0
            for ring_idx, n_pts in enumerate(ring_sizes):
                if n_pts == 1:
                    c0[idx] = [0.0, 0.0]
                    idx += 1
                else:
                    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
                    c0[idx:idx + n_pts, 0] = ring_sizes[ring_idx] * 0.35 * np.cos(angles)
                    c0[idx:idx + n_pts, 1] = ring_sizes[ring_idx] * 0.35 * np.sin(angles)
                    idx += n_pts
        else:
            # Perturbed hexagonal
            spacing = 2.0 / (np.sqrt(N) + 1.0)
            while True:
                k = int(1.0 / spacing) + 2
                rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                        for i in range(-k, k + 1) for j in range(-k, k + 1)]
                points = np.array(rows)
                radius = np.hypot(points[:, 0], points[:, 1])
                mask = radius <= 1.0 - spacing / 2.0
                points = points[mask]
                if len(points) >= N:
                    idx = np.argsort(radius, kind="stable")[:N]
                    c0 = points[idx].copy()
                    # Add random perturbation
                    c0 += rng.normal(0, 0.01, c0.shape)
                    break
                spacing *= 0.98
        
        # Clamp to unit circle
        norms = np.linalg.norm(c0, axis=1)
        mask = norms > 0.99
        c0[mask] *= 0.99 / norms[mask, None]
        
        # Run optimization
        c_flat = c0.flatten()
        
        def residuals_func(x):
            c = x.reshape(-1, 2)
            # Boundary residuals
            dists_to_boundary = 1.0 - np.linalg.norm(c, axis=1)
            boundary_res = r_target - dists_to_boundary
            
            # Pairwise residuals using cKDTree
            tree = cKDTree(c)
            pairs = tree.query_pairs(4.0 * r_target)
            pair_res = []
            for i, j in pairs:
                d = np.linalg.norm(c[i] - c[j])
                violation = r_target - d / 2.0
                if violation > 0:
                    pair_res.append(violation)
            
            pair_res = np.array(pair_res) if pair_res else np.array([0.0])
            return np.concatenate([boundary_res, pair_res])
        
        try:
            result = least_squares(
                residuals_func,
                c_flat,
                method='trf',
                bounds=([-0.99, -0.99], [0.99, 0.99]),
                max_nfev=500,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12
            )
            c_opt = result.x.reshape(-1, 2)
        except Exception as e:
            c_opt = c0
        
        # Ensure validity
        norms = np.linalg.norm(c_opt, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            c_opt[mask] *= 0.99 / norms[mask, None]
        
        # Evaluate score
        s = score_of_centers(c_opt)
        if s > best_score:
            best_score = s
            best_centers = c_opt.copy()
        
        # Check time
        if time.time() - start_time > time_budget:
            break
        
        # If we've reached or exceeded the target, stop
        if best_score >= target_r * 0.999:
            break
    
    # Final check
    if best_centers is None or best_centers.shape != (N, 2):
        # Fallback to hexagonal
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= N:
                idx = np.argsort(radius, kind="stable")[:N]
                return points[idx]
            spacing *= 0.98
    
    return best_centers

# EVOLVE-BLOCK-END
