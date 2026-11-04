import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3)

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_rad):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) with given rotation."""
    angles = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_side_length(positions, rotations, theta):
    """
    Compute the minimum side length s of a regular hexagon centered at origin,
    rotated by theta, that contains all vertices of the inner hexagons.
    
    A regular hexagon with side s and rotation theta has 6 edges.
    The distance from center to each edge (apothem) is s * sqrt(3)/2.
    The edge normals point at angles theta + k*pi/3 for k=0..5.
    
    For each inner vertex v, the constraint is:
        v . n_k <= s * sqrt(3)/2  for all k
    So s >= max over all vertices and all k of (v . n_k) / (sqrt(3)/2)
    """
    all_verts = []
    for i in range(N):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], rotations[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # shape (72, 2)
    
    # Container normals
    normals = np.column_stack([
        np.cos(theta + np.arange(6) * np.pi / 3.0),
        np.sin(theta + np.arange(6) * np.pi / 3.0)
    ])  # shape (6, 2)
    
    # For each vertex, compute max projection onto each normal
    # projections: (72, 6)
    projections = all_verts @ normals.T
    max_proj = np.max(projections, axis=0)  # (6,)
    # s * sqrt(3)/2 >= max_proj[k] => s >= max_proj[k] / (sqrt(3)/2)
    s = np.max(max_proj) / (SQRT3 / 2.0)
    return max(s, 0.0)

def _overlap_penalty(positions, rotations):
    """
    Estimate overlap between pairs of unit hexagons.
    Two unit hexagons (same orientation) don't overlap if their centers are
    at distance >= 2*apothem = sqrt(3) when aligned, but with different rotations
    it's more complex. We use a conservative check: if center distance < 2*inradius
    where inradius = sqrt(3)/2, they definitely overlap.
    Actually for two unit hexagons, the minimum distance between centers for
    non-overlap depends on relative rotation. A safe lower bound: if dist < sqrt(3)
    they may overlap. We'll use a soft penalty based on distance.
    
    For simplicity, use: penalty = sum over pairs of max(0, d_min - dist)^2
    where d_min is the minimum center-to-center distance for non-overlap.
    For two identical orientation unit hexagons, d_min = 2 * apothem = sqrt(3) ≈ 1.732
    (when flat-to-flat aligned). For arbitrary rotations, it can be as small as
    2 * inradius = sqrt(3) in the best case... actually the inradius of a unit hex
    is sqrt(3)/2, so two hexagons with centers at distance 2*inradius = sqrt(3) 
    just touch when their flats are parallel.
    
    We'll use a simpler approach: check if any vertex of one hex is inside the other.
    """
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            vi = _hex_vertices(positions[i,0], positions[i,1], rotations[i])
            vj = _hex_vertices(positions[j,0], positions[j,1], rotations[j])
            # Check if any vertex of i is inside j and vice versa
            # Simple point-in-convex-polygon test
            for v in vi:
                if _point_in_hex(v, positions[j,0], positions[j,1], rotations[j]):
                    penalty += 1.0
                    break
            for v in vj:
                if _point_in_hex(v, positions[i,0], positions[i,1], rotations[i]):
                    penalty += 1.0
                    break
    return penalty

def _point_in_hex(px, py, cx, cy, angle_rad):
    """Check if point (px,py) is inside unit hex centered at (cx,cy) with rotation."""
    # Transform to hex local frame
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    
    # Unit hex vertices at angles 0,60,...,300
    # Check if point is inside using half-plane tests
    # Normal to each edge points inward
    for k in range(6):
        ang = k * np.pi / 3.0
        nx = np.cos(ang + np.pi/6)  # normal direction (inward)
        ny = np.sin(ang + np.pi/6)
        # Distance from center to edge along normal = apothem = sqrt(3)/2
        if lx * nx + ly * ny > SQRT3 / 2.0 + 1e-9:
            return False
    return True

def _objective(params):
    """
    Objective: minimize container side length + penalty for overlaps.
    params = [theta, x0,y0, ..., x11,y11, r0, r1, ..., r11]
    """
    theta = params[0]
    positions = params[1:1+2*N].reshape(N, 2)
    rotations = params[1+2*N:1+3*N]
    
    s = _min_side_length(positions, rotations, theta)
    pen = _overlap_penalty(positions, rotations)
    return s + 100.0 * pen

def construct():
    """
    Construct 12 unit hexagons inside a regular hexagon, minimizing outer side length.
    Uses spectral graph embedding for initial configuration, then Nelder-Mead refinement.
    """
    # --- Step 1: Build adjacency graph for spectral embedding ---
    # Use a hexagonal lattice structure as the graph.
    # 12 hexagons: arrange as a 4x3 grid in the graph.
    # Node (i,j) for i in 0..2 (rows), j in 0..3 (cols)
    # Adjacent if |di|+|dj| == 1 in grid coords (with hex offset)
    
    # For hexagonal lattice, use offset coordinates:
    # Row 0: cols 0,1,2,3
    # Row 1: cols 0,1,2,3 (offset by 0.5 in x)
    # Row 2: cols 0,1,2,3
    # In hex lattice, neighbors of (r,c):
    #   (r, c-1), (r, c+1)  [same row]
    #   (r-1, c), (r-1, c+1) if r even... let's use axial coords
    #
    # Simpler: just use a 4x3 grid graph with 4-neighbor connectivity
    # and let the optimizer figure out the rest.
    
    # Build Laplacian of 4x3 grid
    n_nodes = 12
    L = np.zeros((n_nodes, n_nodes))
    
    def idx(r, c):
        return r * 4 + c
    
    for r in range(3):
        for c in range(4):
            i = idx(r, c)
            # Neighbors
            if c > 0:
                j = idx(r, c-1)
                L[i,j] = L[j,i] = -1.0
            if c < 3:
                j = idx(r, c+1)
                L[i,j] = L[j,i] = -1.0
            if r > 0:
                j = idx(r-1, c)
                L[i,j] = L[j,i] = -1.0
            if r < 2:
                j = idx(r+1, c)
                L[i,j] = L[j,i] = -1.0
            L[i,i] = -L[i].sum()
    
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Use the 2nd and 3rd eigenvectors (indices 1 and 2) for x and y coordinates
    # (skip the trivial 0 eigenvalue)
    x_init = eigenvectors[:, 1]
    y_init = eigenvectors[:, 2]
    
    # Scale to reasonable size
    scale = 1.5
    x_init *= scale
    y_init *= scale
    
    positions = np.column_stack([x_init, y_init])
    rotations = np.zeros(N)
    
    # --- Step 2: Nelder-Mead optimization ---
    params0 = np.concatenate([[0.0], positions.flatten(), rotations])
    
    result = minimize(
        _objective,
        params0,
        method='Nelder-Mead',
        options={
            'maxiter': 8000,
            'maxfev': 15000,
            'xatol': 1e-5,
            'fatol': 1e-5,
            'adaptive': True
        }
    )
    
    best_params = result.x
    theta = best_params[0]
    positions = best_params[1:1+2*N].reshape(N, 2)
    rotations = best_params[1+2*N:1+3*N]
    
    # Check for overlaps; if significant, try a few random restarts
    pen = _overlap_penalty(positions, rotations)
    best_s = _min_side_length(positions, rotations, theta)
    best_pen = pen
    
    # Try a few random restarts if penalty is high
    if pen > 0.1:
        for trial in range(3):
            np.random.seed(42 + trial)
            pos_rand = np.random.randn(N, 2) * 1.5
            rot_rand = np.random.uniform(-np.pi/6, np.pi/6, N)
            theta_rand = np.random.uniform(-np.pi/6, np.pi/6)
            p0 = np.concatenate([[theta_rand], pos_rand.flatten(), rot_rand])
            
            res = minimize(
                _objective, p0, method='Nelder-Mead',
                options={'maxiter': 4000, 'maxfev': 8000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
            )
            bp = res.x
            th = bp[0]
            ps = bp[1:1+2*N].reshape(N, 2)
            rs = bp[1+2*N:1+3*N]
            s_try = _min_side_length(ps, rs, th)
            pen_try = _overlap_penalty(ps, rs)
            if pen_try < best_pen or (pen_try == 0 and s_try < best_s):
                best_s = s_try
                best_pen = pen_try
                theta = th
                positions = ps
                rotations = rs
    
    # Final fallback: known valid 3x4 grid with sufficient container
    if best_pen > 1e-6:
        positions = []
        rotations = []
        for row in range(3):
            y = (row - 1) * SQRT3
            for col in range(4):
                x = (col - 1.5) * 2.0
                positions.append([x, y])
                rotations.append(0.0)
        positions = np.array(positions)
        rotations = np.array(rotations)
        theta = 0.0
        best_s = _min_side_length(positions, rotations, theta)
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(np.degrees(rotations[i]))])
    
    return inner, (0.0, 0.0), float(best_s), float(np.degrees(theta))
# EVOLVE-BLOCK-END
