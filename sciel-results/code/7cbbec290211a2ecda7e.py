import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) with given rotation."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Base vertices of unit hexagon (flat-to-flat, angle 0): angles 0,60,...,300
    base_angles = np.deg2rad(np.arange(0, 360, 60))
    vx = np.cos(base_angles)
    vy = np.sin(base_angles)
    # Rotate
    rx = ca * vx - sa * vy
    ry = sa * vx + ca * vy
    return np.column_stack([cx + rx, cy + ry])


def _outer_hex_normals(angle_deg):
    """Return 6 outward unit normals for a flat-to-flat regular hexagon rotated by angle_deg.
    For a flat-to-flat hexagon (vertices at 0,60,...), the outward normals point at 30,90,150,210,270,330.
    With rotation angle_deg, normals are at 30+angle, 90+angle, ...
    """
    base = np.deg2rad(30 + np.arange(0, 360, 60))
    a = np.deg2rad(angle_deg)
    base = base + a
    return np.column_stack([np.cos(base), np.sin(base)])


def _max_projection(positions, angles, container_angle):
    """Compute the minimum side length s such that all hexagon vertices are inside the outer hexagon.
    For a flat-to-flat hexagon of side s centered at origin with rotation angle,
    the constraint for each normal n_i is: n_i . v <= s * cos(30°) = s * sqrt(3)/2
    Wait, let me reconsider. For a regular hexagon of side s, the apothem (distance from center to midpoint of each side) is s * sqrt(3)/2.
    The outward normals point to the midpoints of the sides. So the constraint is n_i . v <= apothem = s * sqrt(3)/2.
    Therefore s >= max_i (n_i . v) / (sqrt(3)/2) = max_i (n_i . v) * 2/sqrt(3).
    """
    normals = _outer_hex_normals(container_angle)  # shape (6,2)
    apothem_factor = SQRT3 / 2.0  # apothem = s * this
    
    max_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], angles[i])
        # Project all vertices onto each normal
        projs = verts @ normals.T  # shape (6,6): vertex x normal
        max_p = np.max(projs)
        if max_p > max_proj:
            max_proj = max_p
    
    return max_proj / apothem_factor


def _pairwise_distance_matrix(positions):
    """Compute pairwise distances between N positions."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _objective(params):
    """Objective function for Nelder-Mead.
    params: [log_weights (N*(N-1)/2), container_angle]
    Returns the minimum outer side length.
    """
    n_pairs = N * (N - 1) // 2
    log_w = params[:n_pairs]
    container_angle = params[n_pairs]
    
    # Build weight matrix from exponential of log weights
    weights = np.exp(log_w)
    W = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            W[i, j] = weights[idx]
            W[j, i] = weights[idx]
            idx += 1
    
    # Compute graph Laplacian
    degree = W.sum(axis=1)
    L = np.diag(degree) - W
    
    # Get eigenvectors
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Use the two smallest non-trivial eigenvectors (Fiedler vectors)
    # eigvals[0] should be ~0 (constant vector), so use indices 1 and 2
    v1 = eigvecs[:, 1]
    v2 = eigvecs[:, 2]
    
    positions = np.column_stack([v1, v2])
    positions -= positions.mean(axis=0)
    
    # Scale positions to have a reasonable spread
    # The eigenvectors are unit norm, so we need to scale up
    # Target: positions should span roughly 4-5 units
    scale = 4.0 / (np.max(np.abs(positions)) + 1e-10)
    positions *= scale
    
    # Try a few angle configurations for inner hexagons
    best_side = float('inf')
    for angle_set in [0.0, 30.0, 60.0]:
        angles = np.full(N, angle_set)
        side = _max_projection(positions, angles, container_angle)
        if side < best_side:
            best_side = side
    
    return best_side


def construct():
    """Construct 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
    n_pairs = N * (N - 1) // 2
    
    # Initialize with symmetric weights (all equal) and container angle 0
    init_params = np.zeros(n_pairs + 1)
    
    # Run Nelder-Mead optimization
    result = minimize(_objective, init_params, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5,
                               'adaptive': True})
    
    opt_params = result.x
    log_w = opt_params[:n_pairs]
    container_angle = opt_params[n_pairs]
    
    # Rebuild weight matrix
    weights = np.exp(log_w)
    W = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            W[i, j] = weights[idx]
            W[j, i] = weights[idx]
            idx += 1
    
    # Compute graph Laplacian and eigenvectors
    degree = W.sum(axis=1)
    L = np.diag(degree) - W
    eigvals, eigvecs = np.linalg.eigh(L)
    
    v1 = eigvecs[:, 1]
    v2 = eigvecs[:, 2]
    positions = np.column_stack([v1, v2])
    positions -= positions.mean(axis=0)
    
    # Scale positions
    scale = 4.0 / (np.max(np.abs(positions)) + 1e-10)
    positions *= scale
    
    # Find best angle configuration
    best_side = float('inf')
    best_angles = np.zeros(N)
    for angle_set in [0.0, 30.0, 60.0, 90.0]:
        angles = np.full(N, angle_set)
        side = _max_projection(positions, angles, container_angle)
        if side < best_side:
            best_side = side
            best_angles = angles
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(best_angles[i])])
    
    return inner, (0.0, 0.0), float(best_side), float(container_angle)

# EVOLVE-BLOCK-END
