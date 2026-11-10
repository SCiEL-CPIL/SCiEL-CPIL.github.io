# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 580.0  # safety margin before 630s kill


def _hex_vertices(cx, cy, s, a_deg):
    """Return (6,2) array of vertices for a regular hexagon."""
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _edge_normals(vertices):
    """Return (6,2) array of outward-pointing unit edge normals for a convex polygon."""
    n = len(vertices)
    norms = np.zeros((n, 2))
    centroid = np.mean(vertices, axis=0)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        normal /= np.linalg.norm(normal)
        mid = (p1 + p2) / 2.0
        if np.dot(normal, mid - centroid) < 0:
            normal = -normal
        norms[i] = normal
    return norms


def _min_gap(v1, v2, n1, n2):
    """
    Minimum separation between two convex hexagons using SAT.
    v1, v2: (6,2) vertex arrays
    n1, n2: (6,2) outward normal arrays
    Returns the minimum gap (positive if separated, 0 if touching/overlapping).
    """
    all_norms = np.vstack([n1, n2])  # (12, 2)
    min_gap = float('inf')
    for k in range(12):
        n = all_norms[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
        if gap < min_gap:
            min_gap = gap
    return min_gap


def _separation(h1, h2):
    """Return min gap between two unit hexagons given as (x, y, angle_deg)."""
    v1 = _hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_vertices(h2[0], h2[1], 1.0, h2[2])
    n1 = _edge_normals(v1)
    n2 = _edge_normals(v2)
    return _min_gap(v1, v2, n1, n2)


def _check_separations(inner):
    """Check all pairs. Return (min_gap, valid) where valid means all gaps > 1e-6."""
    n = len(inner)
    min_gap = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            gap = _separation(inner[i], inner[j])
            if gap < min_gap:
                min_gap = gap
    valid = min_gap > 1e-6
    return min_gap, valid


def _outer_side_for(inner, outer_center=np.array([0.0, 0.0]), outer_angle=0.0):
    """
    Find minimum outer side length such that all inner hex vertices are inside.
    Outer hexagon is centered at outer_center with given angle.
    """
    # For a regular hexagon centered at origin with angle 0, a point (px, py) is inside
    # if for each of the 6 edges, the point is on the correct side.
    # The constraint is: the maximum over all inner vertices of the "support function" value.
    # For a hexagon with side s centered at origin, angle 0:
    # Vertices at s*(cos(60k), sin(60k)) for k=0..5
    # Edge normals (outward) at angles 30+60k degrees
    # A point p is inside if p · n_k <= s * cos(30°) = s * √3/2 for all k
    # So s >= max_k (p · n_k) / (√3/2) for all points p
    # s >= 2/√3 * max_k (p · n_k)
    
    # Outward normals for hexagon with angle 0:
    # Edge i goes from vertex i to vertex i+1
    # Vertex k is at angle 60k degrees
    # Edge from vertex k to vertex k+1 has outward normal at angle 60k + 30 degrees
    normal_angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
    
    # Collect all inner vertices
    all_verts = []
    for h in inner:
        v = _hex_vertices(h[0], h[1], 1.0, h[2])
        # Shift by outer center
        v = v - outer_center
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # (66, 2)
    
    # For each normal, compute max projection
    # s must satisfy: for all k, max_p (p · n_k) <= s * √3/2
    # So s >= 2/√3 * max_k max_p (p · n_k)
    proj = all_verts @ normals.T  # (66, 6)
    max_proj_per_normal = proj.max(axis=0)  # (6,)
    required_s = (2.0 / np.sqrt(3.0)) * max_proj_per_normal
    return float(required_s.max())


def _objective(x, n=11):
    """
    Objective: minimize outer side length + penalty for overlaps.
    x: flat array of 11*3 values
    """
    inner = x.reshape(n, 3)
    
    # Penalty for overlaps
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            gap = _separation(inner[i], inner[j])
            if gap < 1e-6:
                pen += (1e-6 - gap) ** 2 * 1000.0
    
    side = _outer_side_for(inner)
    return side + pen


def _initial_packing():
    """
    Analytical initial packing: 1 center + 6 ring + 4 in the "corner" gaps.
    Unit hexagons in a hexagonal lattice.
    """
    # Hexagonal lattice: neighbors at distance 2 in 6 directions
    # For unit hexagons (side=1), the center-to-center distance for touching is 2.0
    # (the inradius is √3/2, so two hexagons touch when centers are 2*√3/2*√3 = ... 
    #  Actually, for regular hexagons with side 1, the distance between centers of touching hexagons is 2*√3/2 * 2 / √3 ... 
    #  Let me think: the "width" of a hexagon (flat-to-flat) is √3. Two hexagons side by side have centers separated by √3.
    #  Wait, no. For a regular hexagon with side length 1:
    #  - The distance from center to vertex (circumradius) = 1
    #  - The distance from center to edge midpoint (inradius) = √3/2
    #  Two hexagons sharing an edge: centers are 2 * inradius = √3 apart.
    #  So the lattice spacing is √3 in the 6 neighbor directions.
    
    # Directions for hexagonal lattice (angle 0, 60, 120, 180, 240, 300)
    d = np.sqrt(3.0)  # center-to-center distance for touching hexagons
    angles = np.deg2rad(60.0 * np.arange(6))
    dirs = np.column_stack([d * np.cos(angles), d * np.sin(angles)])
    
    inner = np.zeros((11, 3))
    # Center
    inner[0] = [0.0, 0.0, 0.0]
    # Ring of 6
    for k in range(6):
        inner[k + 1] = [dirs[k, 0], dirs[k, 1], 0.0]
    # 4 more in the "corner" positions (between ring hexagons)
    # These are at distance 2*d from center, at angles 30, 90, 150, 210, 270, 330? 
    # Actually, in a hexagonal packing, the next ring is at distance 2*d.
    # But we only need 4 more to make 11. Let's place them at alternating corners.
    # The positions between ring hexagons are at angles 30+60k, distance 2*d? 
    # Let me compute: ring hexagons at angles 0,60,120,...
    # The "gap" between two adjacent ring hexagons (at angles 0 and 60) is at angle 30.
    # The distance from center to this gap position: 
    # The two ring centers are at (d, 0) and (d/2, d√3/2).
    # The point equidistant from both and from center: it's at angle 30, distance = d * 2 * cos(30°) = d * √3? 
    # Let's just compute: the third hexagon in the triangle formed by center, ring0, ring1.
    # Center at (0,0), ring0 at (d,0), ring1 at (d*cos60, d*sin60) = (d/2, d√3/2)
    # The equilateral triangle has side d. The third vertex (for a hexagon touching both) 
    # would be at the centroid + d in the direction... 
    # Actually, for three mutually touching hexagons, the centers form an equilateral triangle of side d.
    # So the 7th hexagon (touching center, ring0, ring1) is at:
    # midpoint of ring0-ring1 = (3d/4, d√3/4), then add d * (perpendicular direction)
    # The direction from center to midpoint is at angle 30°.
    # The third vertex of the equilateral triangle is at distance d from each of the other two.
    # Position: (d*cos(30°), d*sin(30°)) * 2? No.
    # Let's compute directly: we want point P such that |P - (0,0)| = d, |P - (d,0)| = d, |P - (d/2, d√3/2)| = d.
    # From first two: P is at (d/2, d√3/2) or (d/2, -d√3/2). The first one is already ring1.
    # So actually, the three centers (0,0), (d,0), (d/2, d√3/2) already form an equilateral triangle!
    # So the "gap" hexagon would overlap with ring1. That's not right.
    # 
    # I think I'm confusing myself. In a hexagonal close packing of circles/hexagons:
    # Ring 1: 6 hexagons at distance d from center.
    # Ring 2: 12 hexagons at distance 2d from center, at angles 0,30,60,90,...
    # So for 11 hexagons: 1 + 6 + 4 from ring 2.
    # Ring 2 positions: at angles 30k for k=0..11, distance 2d.
    # Let's pick 4 of them, say at angles 30, 90, 150, 210 (or 0, 60, 120, 180 for symmetry).
    
    r2 = 2.0 * d
    # Place 4 hexagons at ring 2, at angles 30, 90, 150, 270 for symmetry
    ring2_angles = np.deg2rad([30.0, 90.0, 150.0, 270.0])
    for k in range(4):
        idx = 7 + k
        inner[idx, 0] = r2 * np.cos(ring2_angles[k])
        inner[idx, 1] = r2 * np.sin(ring2_angles[k])
        inner[idx, 2] = 0.0
    
    return inner


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    n = 11
    inner0 = _initial_packing()
    
    # Check initial packing validity
    min_gap, valid = _check_separations(inner0)
    if not valid:
        # If initial packing has overlaps, try a different arrangement
        # Fallback: use the seed's arrangement
        inner0 = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.165, 0],
            [1.25, 2.165, 0],
            [-1.25, -2.165, 0],
            [1.25, -2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, -2.165, 0],
            [3.75, -2.165, 0],
        ])
        min_gap, valid = _check_separations(inner0)
    
    best_x = inner0.copy()
    best_side = _outer_side_for(inner0)
    best_obj = best_side + (0 if valid else 1e6)
    
    # Local optimization
    rng = np.random.default_rng(42)
    
    # Try multiple starts
    starts = [inner0.copy()]
    for _ in range(8):
        jitter = rng.normal(0, 0.3, inner0.shape)
        # Keep angles at 0 for simplicity
        jitter[:, 2] = 0
        starts.append(inner0 + jitter)
    
    for s0 in starts:
        if time.time() - _START_TIME > _TIME_LIMIT - 60:
            break
        try:
            res = minimize(_objective, s0.flatten(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7,
                                    'adaptive': True})
            x = res.x.reshape(n, 3)
            # Force angles to 0 for simplicity (rotations don't help for this symmetric problem)
            x[:, 2] = 0.0
            
            min_gap, valid = _check_separations(x)
            if valid:
                side = _outer_side_for(x)
                obj = side
                if obj < best_obj:
                    best_obj = obj
                    best_side = side
                    best_x = x.copy()
        except Exception:
            continue
    
    # Final verification
    min_gap, valid = _check_separations(best_x)
    if not valid:
        # Fallback to seed
        best_x = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.165, 0],
            [1.25, 2.165, 0],
            [-1.25, -2.165, 0],
            [1.25, -2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, -2.165, 0],
            [3.75, -2.165, 0],
        ])
        best_side = 8.0
    
    side = best_side
    outer = np.array([0.0, 0.0, 0.0])
    return best_x, outer, float(side)


# EVOLVE-BLOCK-END
