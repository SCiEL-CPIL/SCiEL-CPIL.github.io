"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

def _compute_radii(centers):
    """
    Given centers, compute radii based on Delaunay triangulation.
    For each circle, its radius is half the minimum of:
      - distances to all other centers (but we use Delaunay neighbors for efficiency,
        though for correctness with disjoint interiors, we need min distance to ALL others)
      - distance to each boundary (x=0, x=1, y=0, y=1)
    
    Actually, for disjoint interiors, the constraint is that for every pair i,j:
      r_i + r_j <= dist(i,j)
    And for each i: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    
    A common heuristic for tight packing: set r_i to half the minimum distance to any other center,
    but also capped by boundary distance. However, this might not be optimal.
    
    Better approach: For each center i, the maximum possible radius given other centers' radii is complex.
    But a simpler geometric approach: 
      r_i = min(
        min_{j != i} dist(i,j) / 2,   # but this is too conservative if we want to maximize sum
        min(x_i, 1-x_i, y_i, 1-y_i)
      )
    
    Wait, that's not right either. If we set r_i = min_dist_to_others / 2, then r_i + r_j might still exceed dist(i,j) if r_j is large.
    
    Let me think differently. The directive says: "radii are determined by the half-length of the shortest incident edge and boundary distance".
    
    So for each vertex i in the Delaunay triangulation:
      - Find the shortest edge incident to i in the Delaunay triangulation. Let this be d_min.
      - r_i = min(d_min / 2, min(x_i, 1-x_i, y_i, 1-y_i))
    
    But this might not guarantee disjoint interiors for all pairs, only for Delaunay neighbors. However, in a Delaunay triangulation, the closest point to i is likely a Delaunay neighbor, so this is a reasonable heuristic.
    
    Actually, for safety, let's compute the minimum distance to ALL other points, not just Delaunay neighbors, but use Delaunay to identify which pairs are "close". 
    For efficiency, let's just compute all pairwise distances for N=26.
    """
    n = len(centers)
    if n == 0:
        return np.array([])
    
    # Compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Minimum distance to any other center
    min_dist_to_other = np.min(dists, axis=1)
    
    # Boundary distances
    boundary_dist = np.minimum.reduce([
        centers[:, 0],
        1 - centers[:, 0],
        centers[:, 1],
        1 - centers[:, 1]
    ])
    
    # Radius is half the min distance to other, capped by boundary
    radii = np.minimum(min_dist_to_other / 2.0, boundary_dist)
    
    # Ensure non-negative
    radii = np.maximum(radii, 0)
    
    return radii


def _validate_and_fix(centers, radii):
    """
    Check if circles are disjoint and inside the unit square.
    If not, shrink radii to fix violations.
    """
    n = len(centers)
    if n == 0:
        return centers, radii
    
    # Check boundaries
    for i in range(n):
        max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        if radii[i] > max_r:
            radii[i] = max(max_r, 0)
    
    # Check pairwise disjointness
    # For each pair, if r_i + r_j > dist(i,j), we need to shrink
    # Simple iterative fix
    for iteration in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2)
                if radii[i] + radii[j] > d + 1e-9:
                    # Shrink both proportionally
                    excess = (radii[i] + radii[j]) - d
                    if excess > 0:
                        # Reduce each by excess/2
                        reduction = excess / 2.0
                        radii[i] = max(radii[i] - reduction, 0)
                        radii[j] = max(radii[j] - reduction, 0)
                        changed = True
        if not changed:
            break
    
    # Final boundary check
    for i in range(n):
        max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        if radii[i] > max_r:
            radii[i] = max(max_r, 0)
    
    return centers, radii


def _objective(centers):
    """
    Objective to minimize: negative sum of radii (since we want to maximize sum of radii).
    We also add a penalty for any violations (though we fix them in _validate_and_fix).
    """
    centers = np.array(centers).reshape(-1, 2)
    # Keep centers inside [0,1]^2 with small margin
    centers = np.clip(centers, 0.01, 0.99)
    
    radii = _compute_radii(centers)
    centers, radii = _validate_and_fix(centers, radii)
    
    # Penalty for being too close to boundary or overlapping
    # Actually, after _validate_and_fix, we should be valid.
    # Let's just return negative sum of radii
    return -np.sum(radii)


def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2, maximizing sum of radii.
    """
    np.random.seed(42)
    
    # Initial positions: random or grid
    # Let's use a grid as initial guess
    side = int(np.ceil(np.sqrt(N)))
    centers = np.array([
        [(i + 0.5) / side, (j + 0.5) / side]
        for i in range(side)
        for j in range(side)
    ][:N])
    
    # Optimize using Nelder-Mead or L-BFGS-B
    # Nelder-Mead is derivative-free and might work better here
    result = minimize(
        _objective,
        centers,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
    )
    
    centers_opt = result.x.reshape(-1, 2)
    centers_opt = np.clip(centers_opt, 0.01, 0.99)
    
    radii = _compute_radii(centers_opt)
    centers_opt, radii = _validate_and_fix(centers_opt, radii)
    
    # Combine into (x, y, radius) format
    output = np.column_stack([centers_opt, radii])
    
    return output
# EVOLVE-BLOCK-END
