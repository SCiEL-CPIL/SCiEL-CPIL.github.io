"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(angle_deg=0.0):
    """Return 6 vertices of a unit regular hexagon rotated by angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append([math.cos(theta), math.sin(theta)])
    return np.array(verts)

def _point_in_polygon(pt, polygon):
    """Ray casting point-in-polygon test. polygon is (n,2)."""
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside

def _segment_distance(p1, p2, p3, p4):
    """Minimum distance between segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-10 and e < 1e-10:
        return np.linalg.norm(r)
    elif a < 1e-10:
        t2 = f / e
        t2 = max(0, min(1, t2))
        return np.linalg.norm(r + d2 * t2)
    elif e < 1e-10:
        t1 = -np.dot(d1, r) / a
        t1 = max(0, min(1, t1))
        return np.linalg.norm(r + d1 * t1)
    
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom < 1e-10:
        t1 = 0.0
    else:
        t1 = (b * f - c * e) / denom
        t1 = max(0, min(1, t1))
    
    t2 = (b * t1 + f) / e
    t2 = max(0, min(1, t2))
    if t2 < 0:
        t2 = 0
        t1 = max(0, min(1, -c / a))
    elif t2 > 1:
        t2 = 1
        t1 = max(0, min(1, (b - c) / a))
    
    return np.linalg.norm(r + d1 * t1 - d2 * t2)

def _poly_distance(poly1, poly2):
    """Minimum distance between two convex polygons. Returns 0 if overlapping."""
    # Check if any vertex of poly1 is inside poly2 or vice versa
    for v in poly1:
        if _point_in_polygon(v, poly2):
            return 0.0
    for v in poly2:
        if _point_in_polygon(v, poly1):
            return 0.0
    
    # Check all edge pairs
    n1 = len(poly1)
    n2 = len(poly2)
    min_dist = float('inf')
    for i in range(n1):
        for j in range(n2):
            d = _segment_distance(poly1[i], poly1[(i+1)%n1], poly2[j], poly2[(j+1)%n2])
            if d < min_dist:
                min_dist = d
    return min_dist

def _outer_hex_vertices(cx, cy, side, angle_deg=0.0):
    """Vertices of outer regular hexagon with given center, side length, rotation."""
    # For a regular hexagon with side length S, the distance from center to vertex is S
    # Vertices at angles angle_deg + i*60
    verts = []
    a = math.radians(angle_deg)
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append([cx + side * math.cos(theta), cy + side * math.sin(theta)])
    return np.array(verts)

def _compute_side_length(centers, angles, outer_angle_deg=0.0):
    """
    Compute the minimum side length of an outer hexagon (at given rotation)
    that contains all inner hexagons.
    """
    # The outer hexagon with side S and rotation phi has its edges at distance
    # S*sqrt(3)/2 from center. The normals to the edges are at angles phi + i*60.
    # A point (x,y) is inside if for all k: (x-cx)*cos(nk) + (y-cy)*sin(nk) <= S*sqrt(3)/2
    # where nk = phi + k*60 degrees, k=0..5
    
    # We want to find S such that all vertices of all inner hexagons are inside.
    # S >= max over all vertices v of (v . n_k) / (sqrt(3)/2) for all k
    # But we need to consider both positive and negative directions.
    # Actually the constraint is: for all k in 0..5, |v . n_k| <= S * sqrt(3)/2
    # So S >= max_k (max_v |v . n_k|) / (sqrt(3)/2)
    
    outer_normals = []
    a = math.radians(outer_angle_deg)
    for k in range(6):
        theta = a + k * math.pi / 3.0
        outer_normals.append([math.cos(theta), math.sin(theta)])
    outer_normals = np.array(outer_normals)  # (6, 2)
    
    max_extent = 0.0
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        # verts: (6, 2)
        projections = verts @ outer_normals.T  # (6, 6)
        abs_proj = np.abs(projections)
        max_extent = max(max_extent, np.max(abs_proj))
    
    S = max_extent / (math.sqrt(3) / 2.0)
    return S

def _overlap_penalty(centers, angles):
    """
    Compute a penalty for overlapping inner hexagons.
    Uses actual polygon distance for accuracy.
    """
    penalty = 0.0
    polys = []
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        polys.append(verts)
    
    for i in range(N):
        for j in range(i + 1, N):
            d = _poly_distance(polys[i], polys[j])
            if d < 0.01:
                penalty += (0.01 - d) ** 2
    return penalty

def _is_valid(centers, angles, outer_side, outer_angle_deg=0.0):
    """Check if the configuration is valid (no overlaps, all inside outer)."""
    # Check overlaps
    polys = []
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        polys.append(verts)
    
    for i in range(N):
        for j in range(i + 1, N):
            if _poly_distance(polys[i], polys[j]) < 1e-6:
                return False
    
    # Check containment
    outer_verts = _outer_hex_vertices(0, 0, outer_side, outer_angle_deg)
    for i in range(N):
        for v in polys[i]:
            if not _point_in_polygon(v, outer_verts):
                return False
    return True

def _seed_solution():
    """Generate a valid seed solution using a hexagonal lattice."""
    d = math.sqrt(3)
    centers = []
    angles = []
    
    # Central hexagon
    centers.append([0, 0])
    angles.append(0.0)
    
    # 6 surrounding
    for i in range(6):
        theta = i * math.pi / 3.0
        centers.append([d * math.cos(theta), d * math.sin(theta)])
        angles.append(0.0)
    
    # 4 more from second ring
    # Second ring positions at distance 2*sqrt(3) in certain directions
    # Use positions that are valid
    ring2_positions = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_positions.append([x, y])
    
    # Pick 4 that are well-separated
    # Use every 3rd from the 12 positions
    selected = [ring2_positions[i] for i in [0, 3, 6, 9]]
    for pos in selected:
        centers.append(pos)
        angles.append(0.0)
    
    centers = np.array(centers)
    angles = np.array(angles)
    
    # Compute required outer side
    S = _compute_side_length(centers, angles)
    # Add small margin
    S *= 1.01
    
    return centers, angles, S

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    # Get seed solution
    centers0, angles0, S0 = _seed_solution()
    best_S = S0
    best_centers = centers0.copy()
    best_angles = angles0.copy()
    
    # Verify seed is valid
    if not _is_valid(best_centers, best_angles, best_S):
        # Fallback: spread out more
        d = math.sqrt(3) * 1.5
        centers_fb = []
        angles_fb = []
        centers_fb.append([0, 0])
        angles_fb.append(0.0)
        for i in range(6):
            theta = i * math.pi / 3.0
            centers_fb.append([d * math.cos(theta), d * math.sin(theta)])
            angles_fb.append(0.0)
        for i in range(4):
            theta = (2*i + 1) * math.pi / 12.0
            centers_fb.append([2*d * math.cos(theta), 2*d * math.sin(theta)])
            angles_fb.append(0.0)
        best_centers = np.array(centers_fb)
        best_angles = np.array(angles_fb)
        best_S = _compute_side_length(best_centers, best_angles) * 1.05
    
    # Objective for differential evolution
    def objective(params):
        if time.time() > deadline:
            return 1e10
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Compute outer side
        S = _compute_side_length(centers, angles)
        
        # Compute overlap penalty
        pen = _overlap_penalty(centers, angles)
        
        return S + 100.0 * pen
    
    # Define bounds for differential evolution
    # Centers: roughly in [-3, 3] x [-3, 3]
    # Angles: [0, 360) but we'll use [0, 180] since hexagon has 6-fold symmetry
    bounds = []
    for i in range(N):
        bounds.append((-4.0, 4.0))  # x
        bounds.append((-4.0, 4.0))  # y
    for i in range(N):
        bounds.append((0.0, 180.0))  # angle
    
    # Initial population
    init_pop = [centers0.flatten().tolist() + angles0.tolist()]
    
    # Add some random valid-ish starting points
    np.random.seed(42)
    for _ in range(10):
        c = np.random.uniform(-2.5, 2.5, size=(N, 2))
        a = np.random.uniform(0, 180, size=N)
        init_pop.append(c.flatten().tolist() + a.tolist())
    
    # Run differential evolution
    try:
        res = differential_evolution(
            objective, bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            mutation=(0.5, 1.5),
            recombination=0.8,
            polish=False,
            init=init_pop,
            updating='deferred'
        )
        
        if res.fun < best_S:
            params = res.x
            c = params[:2*N].reshape(N, 2)
            a = params[2*N:]
            S_val = _compute_side_length(c, a)
            pen_val = _overlap_penalty(c, a)
            if pen_val < 1e-4 and S_val < best_S:
                best_S = S_val
                best_centers = c
                best_angles = a
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead if time permits
    if time.time() < deadline - 20:
        from scipy.optimize import minimize
        
        def nm_objective(params):
            if time.time() > deadline:
                return 1e10
            centers = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            S = _compute_side_length(centers, angles)
            pen = _overlap_penalty(centers, angles)
            return S + 500.0 * pen
        
        x0 = best_centers.flatten().tolist() + best_angles.tolist()
        try:
            res = minimize(nm_objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000, 'maxfev': 5000})
            if res.fun < best_S:
                params = res.x
                c = params[:2*N].reshape(N, 2)
                a = params[2*N:]
                S_val = _compute_side_length(c, a)
                pen_val = _overlap_penalty(c, a)
                if pen_val < 1e-5 and S_val < best_S:
                    best_S = S_val
                    best_centers = c
                    best_angles = a
        except Exception:
            pass
    
    # Final verification and adjustment
    if not _is_valid(best_centers, best_angles, best_S):
        # Recompute S with margin
        S_val = _compute_side_length(best_centers, best_angles)
        best_S = S_val * 1.02
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i][0]), float(best_centers[i][1]), float(best_angles[i])])
    
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
