# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def heuristics(distance_matrix):
    t0 = time.time()
    n = distance_matrix.shape[0]
    deadline = 330.0 - 30.0  # safety margin

    # Reconstruct 2D points via classical MDS (isometric embedding)
    # distance_matrix is the Euclidean distance matrix of 50 points in unit square
    D2 = distance_matrix ** 2
    # Double-centering: B = -0.5 * J * D2 * J
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # Eigen-decompose for coordinates
    eigvals, eigvecs = np.linalg.eigh(B)
    # Take top 2 eigenvalues (sorted descending)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Guard against negative eigenvalues (numerical noise)
    eigvals = np.clip(eigvals, 0, None)
    coords = eigvecs[:, :2] * np.sqrt(eigvals[:2])

    # Compute convex hull for hull-membership bias
    try:
        hull = ConvexHull(coords)
        hull_set = set(hull.vertices)
    except Exception:
        hull_set = set()

    # Compute per-point angle to neighbor for "straightness" heuristic
    # For each pair (i, j), compute how "straight" the path is
    # by looking at the angle between (i->j) and the local direction
    # A simpler approach: use the angle subtended at j by nearby points

    # Distance-based attraction: shorter is more attractive
    # Use a sharpened inverse power
    eps = 1e-9
    d = distance_matrix + eps

    # Base: inverse distance raised to power (sharpened)
    base = 1.0 / (d ** 2.5)

    # Angle-based continuity: for each edge (i,j), compute the "turning" quality
    # We want to prefer edges that keep the tour going in a roughly consistent direction
    # Compute for each node its nearest neighbors to define local direction
    # For edge (i, j): the quality is how well j continues from i's local direction
    # Simpler: use the angle between vectors at each node

    # Compute unit vectors from each node to all others
    # v[i, j] = (coords[j] - coords[i]) / ||coords[j] - coords[i]||
    diff = coords[:, None, :] - coords[None, :, :]  # (n, n, 2)
    norms = np.linalg.norm(diff, axis=2)  # (n, n)
    norms_safe = np.where(norms > eps, norms, 1.0)
    unit = diff / norms_safe[:, :, None]  # (n, n, 2)

    # For edge (i, j), the "straightness" at j relative to coming from i:
    # We want the outgoing direction from j to align with the incoming direction
    # But we don't know the outgoing node yet. Instead, use a simpler proxy:
    # The angle between (i->j) and the local "flow" direction at j.
    # Approximate flow direction at j as the direction to its nearest neighbor.
    # Find nearest neighbor for each node (excluding self)
    nn_dist = norms.copy()
    np.fill_diagonal(nn_dist, np.inf)
    nn_idx = np.argmin(nn_dist, axis=1)  # nearest neighbor index for each node

    # Local direction at node j: direction from j to its nearest neighbor
    local_dir = unit[np.arange(n), nn_idx]  # (n, 2)

    # For edge (i, j): incoming direction at j is unit[i, j] (from i to j)
    # Straightness = dot(incoming_dir, local_dir[j])
    # If the tour comes into j from i and j's natural flow is to nn,
    # then a good continuation would have incoming aligned with local flow
    incoming = unit  # (n, n, 2) where incoming[i, j] = direction i->j
    # straightness[i, j] = dot(incoming[i, j], local_dir[j])
    straightness = np.einsum('nij,nj->ni', incoming, local_dir)
    # straightness is in [-1, 1]; higher means more aligned
    # Convert to positive weight: prefer higher straightness
    # Use a soft scaling: map [-1,1] to [0.5, 2.0]
    straight_weight = 0.5 + 0.75 * (straightness + 1.0)  # range [0.5, 2.0]
    straight_weight = np.clip(straight_weight, 0.5, 2.0)

    # Hull bias: edges on the convex hull perimeter are "safer" (less likely to cross)
    # Boost edges where both endpoints are on the hull, especially hull-adjacent ones
    hull_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        if i in hull_set:
            for j in range(n):
                if j in hull_set:
                    hull_mask[i, j] = True
    # Compute hull perimeter order for adjacent boost
    hull_order = list(hull.vertices)
    hull_adj = np.zeros((n, n), dtype=bool)
    for k in range(len(hull_order)):
        a = hull_order[k]
        b = hull_order[(k + 1) % len(hull_order)]
        hull_adj[a, b] = True
        hull_adj[b, a] = True

    # Combine: base * straight_weight * hull_boost
    hull_boost = np.ones((n, n))
    hull_boost[hull_mask] = 1.15  # mild boost for hull-hull edges
    hull_boost[hull_adj] = 1.3     # stronger boost for adjacent hull edges

    # Also add a small penalty for very long edges that skip over many points
    # (these are likely to cause crossings)
    # Use a distance penalty that grows quadratically beyond the median
    median_d = np.median(d)
    long_penalty = np.ones((n, n))
    long_mask = d > 2.0 * median_d
    long_penalty[long_mask] = 0.7

    result = base * straight_weight * hull_boost * long_penalty

    # Ensure positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)

    # Symmetrize (TSP is symmetric, but the pheromone update is directional;
    # the heuristic should be symmetric since it's based on geometry)
    result = 0.5 * (result + result.T)

    # Final safety: ensure all positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)

    return result
# EVOLVE-BLOCK-END
