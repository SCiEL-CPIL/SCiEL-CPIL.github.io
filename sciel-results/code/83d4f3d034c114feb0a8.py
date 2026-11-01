"""Improved packing: hexagonal seed + gradient-based local refinement."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s (or 150s) limit
    
    # Step 1: Generate a good initial hexagonal packing
    points = _hexagonal_seed()
    
    # Step 2: Local optimization via gradient ascent on the objective
    # Objective: r = min(min_{i<j} |c_i - c_j|/2, min_i (1 - |c_i|))
    # We maximize r by pushing points apart and towards the boundary.
    
    points = _local_optimize(points, deadline)
    
    # Step 3: Final polish with tighter iterations if time remains
    if time.time() < deadline - 5.0:
        points = _local_optimize(points, deadline, tight=True)
    
    return points


def _hexagonal_seed():
    """Generate N points in a hexagonal lattice pattern inside unit circle."""
    # Try different spacings to get exactly N points
    best_points = None
    best_radius = 0.0
    
    for spacing in np.linspace(0.03, 0.12, 30):
        k = int(1.0 / spacing) + 3
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    rows.append((x, y))
        
        if len(rows) < N:
            continue
        
        pts = np.array(rows)
        # Sort by distance from origin, take closest N
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        pts = pts[idx]
        
        # Compute the radius this packing achieves
        r = _compute_radius(pts)
        if r > best_radius:
            best_radius = r
            best_points = pts.copy()
    
    if best_points is None:
        # Fallback: uniform random
        best_points = np.random.randn(N, 2) * 0.5
    
    return best_points


def _compute_radius(points):
    """Compute the packing radius: min of min pair distance/2 and min distance to boundary."""
    n = len(points)
    if n < 2:
        return 1.0 - np.max(np.hypot(points[:, 0], points[:, 1]))
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to boundary
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_pair, min_boundary)


def _local_optimize(points, deadline, tight=False):
    """
    Local optimization: gradient ascent on a smooth surrogate.
    Use soft-min to make the objective differentiable.
    """
    n = len(points)
    if n < 2:
        return points
    
    # We'll use a soft-min formulation.
    # r_soft = -log(sum(exp(-alpha * d_i)))/alpha where d_i are the constraints
    # Constraints: for each pair i<j, d_ij = |c_i - c_j|/2
    #              for each i, d_i = 1 - |c_i|
    
    # For efficiency, we'll use a simpler approach: 
    # Iteratively push points apart based on nearest neighbors.
    
    max_iter = 500 if tight else 200
    step_size = 0.001 if tight else 0.005
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find nearest neighbor for each point
        nn_idx = np.argmin(dists, axis=1)
        nn_dists = dists[np.arange(n), nn_idx]
        
        # Also consider boundary constraint
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dist = 1.0 - radii
        
        # The limiting constraint for each point is min(nn_dist/2, boundary_dist)
        # We want to increase the minimum of all these.
        
        # Gradient-like update: move each point away from its nearest neighbor
        # and towards the boundary (outward) if boundary is limiting.
        
        updates = np.zeros_like(points)
        
        for i in range(n):
            j = nn_idx[i]
            if i == j:
                continue
            
            # Direction away from nearest neighbor
            if nn_dists[i] > 1e-10:
                dir_nn = (points[i] - points[j]) / nn_dists[i]
            else:
                dir_nn = np.random.randn(2)
                dir_nn /= np.linalg.norm(dir_nn)
            
            # Direction towards boundary (radially outward)
            if radii[i] > 1e-10:
                dir_boundary = points[i] / radii[i]
            else:
                dir_boundary = np.random.randn(2)
                dir_boundary /= np.linalg.norm(dir_boundary)
            
            # Weight based on which constraint is more limiting
            w_nn = nn_dists[i] / 2.0
            w_boundary = boundary_dist[i]
            
            if w_nn < w_boundary:
                # Pair constraint is limiting: move away from neighbor
                updates[i] = dir_nn
            else:
                # Boundary constraint is limiting: move outward
                updates[i] = dir_boundary
            
            # Also move slightly towards center if too close to boundary
            if boundary_dist[i] < 0.01:
                updates[i] -= 0.5 * dir_boundary
        
        # Normalize updates
        norms = np.linalg.norm(updates, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        updates /= norms
        
        # Apply update
        points = points + step_size * updates
        
        # Ensure points stay inside the unit circle
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            points[mask] /= radii[mask, np.newaxis]
            points[mask] *= 0.999
    
    return points


# EVOLVE-BLOCK-END
