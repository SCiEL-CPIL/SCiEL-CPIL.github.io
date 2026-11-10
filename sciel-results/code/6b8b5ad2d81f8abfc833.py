# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _check_valid(inner, outer, s, tol=1e-6):
    """Check if the packing is valid. Returns True if valid."""
    # Check finite
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    ocx, ocy, oang = outer
    # Outer vertices
    outer_verts = _hex_vertices(ocx, ocy, s, oang)
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Check unit edges of inner hexagons
    for row in inner:
        cx, cy, ang = row
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for i in range(6):
            edge_len = np.linalg.norm(verts[(i+1)%6] - verts[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # Check separation between inner hexagons (SAT)
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            vi = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            vj = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            # SAT: check all edge normals of both hexagons
            overlap = True
            for verts in [vi, vj]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-15)
                    proj_i = vi @ normal
                    proj_j = vj @ normal
                    min_i, max_i = proj_i.min(), proj_i.max()
                    min_j, max_j = proj_j.min(), proj_j.max()
                    if max_i < min_j - tol or max_j < min_i - tol:
                        overlap = False
                        break
                if not overlap:
                    break
            if overlap:
                return False
    
    # Check all inner vertices inside outer hexagon
    for row in inner:
        cx, cy, ang = row
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -1e-6:
                    return False
    return True

def _compute_s(inner, oang):
    """Compute minimum side length of outer hexagon (centered at origin, angle oang) containing all inner hexagons."""
    # Outer hexagon centered at origin with angle oang
    # A point (x,y) is inside if for each edge, the cross product test passes
    # Edge k: from vertex k to vertex k+1
    # For a regular hexagon centered at origin with circumradius s and angle a:
    # Vertex k: (s*cos(a+60k), s*sin(a+60k))
    # The hexagon is the intersection of 6 half-planes.
    # Half-plane k: the side containing origin.
    # Edge from V_k to V_{k+1}, normal pointing inward.
    
    a = np.radians(oang)
    # For each edge, compute the maximum "support" needed
    # Edge k has direction from V_k to V_{k+1}
    # The inward normal for edge k (where edge goes from angle a+60k to a+60(k+1))
    # The midpoint of edge k is at angle a + 60k + 30, at distance s*cos(30) = s*sqrt(3)/2 from center
    # The inward normal points toward center, so the half-plane is:
    # dot(point, n_k) <= s * cos(30 deg) where n_k is unit normal at angle a+60k+30
    
    # Actually, for a regular hexagon with circumradius s, the inradius is s*cos(30°) = s*√3/2
    # The half-plane for edge k (outward normal at angle θ_k = a + 60k + 30):
    # dot(point, [cos(θ_k), sin(θ_k)]) <= s * cos(30°) = s * √3/2
    
    # So for all points p in all inner hexagons:
    # dot(p, n_k) <= s * √3/2 for all k
    # => s >= max over all points p and all k of dot(p, n_k) / (√3/2)
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for row in inner:
        cx, cy, ang = row
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Normals
    normals = []
    for k in range(6):
        theta = a + np.radians(60*k + 30)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    normals = np.array(normals)
    
    # For each normal, find max dot product with all vertices
    # all_verts shape (72, 2), normals shape (6, 2)
    dots = all_verts @ normals.T  # (72, 6)
    max_dots = dots.max(axis=0)  # (6,)
    
    # s >= max_dots[k] / (√3/2) for all k
    s_min = np.max(max_dots) / (np.sqrt(3)/2)
    return s_min

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    # Strategy: Fix outer hexagon angle to 0 (by symmetry, we can assume this WLOG for the optimal)
    # Actually, the optimal may have a specific angle. Let's optimize over angle too.
    # But first, let's try a good initial configuration.
    
    # Initial configuration: hexagonal cluster
    # Place 12 hexagons in a tight hexagonal arrangement
    # Row-based: 
    # For a hexagonal cluster of side 3 (which has 1+2+3+4+3+2+1=16, too many)
    # Let's use a different arrangement.
    
    # Better: use the known optimal topology for 12 hexagons
    # The optimal arrangement has 12 hexagons in a pattern that minimizes the bounding hexagon.
    
    # Let's try: all hexagons with same orientation (angle 0), optimize positions
    # Use Nelder-Mead to minimize s
    
    def objective(params):
        # params: 24 values = 12*(x,y) for centers, all angles fixed to 0
        # Actually let's also optimize outer angle
        # params: 24 + 1 = 25 values
        centers = params[:24].reshape(12, 2)
        oang = params[24]
        
        # Center the configuration: shift so centroid is at origin
        centroid = centers.mean(axis=0)
        centers = centers - centroid
        
        inner = np.column_stack([centers, np.zeros(12)])
        s = _compute_s(inner, oang)
        return s
    
    # Initial guess: hexagonal arrangement
    # 12 hexagons in a pattern:
    # Let's use a 4-3-3-2 arrangement or similar
    # Actually, let's start with a rough hexagonal cluster
    
    # Hexagonal grid positions (for unit hexagons, center-to-center distance = 2)
    # In a hex grid, neighbors are at distance 2 in 6 directions
    # Direction vectors: (2,0), (1,√3), (-1,√3), (-2,0), (-1,-√3), (1,-√3)
    
    # Let's place them in a compact shape
    # Try: a 3x4 rectangular-ish arrangement in hex coordinates
    # Hex coordinate (q,r) to cartesian: x = 2*q + r, y = √3*r (for pointy-top)
    # For flat-top: x = 1.5*q, y = √3*(r + q/2)
    
    # Let's just use a simple initial guess and let Nelder-Mead work
    np.random.seed(42)
    
    # Start with a hexagonal cluster centered at origin
    # Positions in a hex grid (using axial coordinates)
    # For 12 hexagons, a good initial shape:
    hex_positions = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (-1, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2),
    ]
    # Convert to cartesian (flat-top hexagons, side=1, so center distance=2)
    # For flat-top: x = 1.5*q, y = √3*(r + q/2)
    init_centers = []
    for q, r in hex_positions:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        init_centers.append([x, y])
    init_centers = np.array(init_centers)
    
    # Center it
    init_centers = init_centers - init_centers.mean(axis=0)
    
    init_params = np.append(init_centers.flatten(), 0.0)
    
    best_params = init_params.copy()
    best_s = objective(init_params)
    
    # Nelder-Mead optimization
    n_iter = 0
    while time.time() < deadline:
        result = minimize(objective, best_params, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < best_s:
            best_s = result.fun
            best_params = result.x.copy()
        n_iter += 1
        # If converged well, break
        if result.fun < best_s + 1e-12:
            break
        # Otherwise, perturb and continue
        if time.time() > deadline - 10:
            break
    
    # Extract best solution
    centers = best_params[:24].reshape(12, 2)
    oang = best_params[24]
    
    # Center the configuration
    centroid = centers.mean(axis=0)
    centers = centers - centroid
    
    inner = np.column_stack([centers, np.zeros(12)])
    outer = np.array([0.0, 0.0, oang])
    s = _compute_s(inner, oang)
    
    # Verify validity
    if not _check_valid(inner, outer, s):
        # Fallback: use a known valid configuration
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer = np.array([0, 0, 0])
        s = 8
    
    return inner, outer, s

# EVOLVE-BLOCK-END
