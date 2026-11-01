"""Improved packing: differential evolution + local polish for 12 unit hexagons."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12
DEADLINE = 115.0  # safety margin before 120s kill

def _hex_verts(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon (flat-top orientation at angle 0)."""
    a = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    angles = base_angles + a
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _sat_disjoint(v1, v2, eps=1e-9):
    """Check if two convex polygons are disjoint using SAT."""
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        # Project both polygons onto this normal
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 + eps or max2 < min1 + eps:
            return True  # Separating axis found -> disjoint
    for i in range(n2):
        p1 = v2[i]
        p2 = v2[(i + 1) % n2]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 + eps or max2 < min1 + eps:
            return True
    return False

def _hexagons_disjoint(inner, eps=1e-9):
    """Check all pairs of inner hexagons are disjoint."""
    verts = []
    for (cx, cy, ang) in inner:
        verts.append(_hex_verts(cx, cy, ang))
    for i in range(N):
        for j in range(i + 1, N):
            if not _sat_disjoint(verts[i], verts[j], eps):
                return False
    return True

def _outer_hex_contains(outer_center, outer_side, outer_angle, inner, eps=1e-9):
    """Check that all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = _hex_verts(ox, oy, outer_angle, outer_side)
    # For each inner hexagon, check all its vertices are inside outer hexagon
    for (cx, cy, ang) in inner:
        iv = _hex_verts(cx, cy, ang)
        for v in iv:
            if not _point_in_convex(v, outer_verts, eps):
                return False
    return True

def _point_in_convex(pt, poly, eps=1e-9):
    """Check if point is inside convex polygon (all cross products same sign)."""
    n = len(poly)
    signs = []
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        cross = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
        signs.append(cross)
    # All should be >= -eps or all <= eps (consistent winding)
    if all(s >= -eps for s in signs) or all(s <= eps for s in signs):
        return True
    return False

def _min_outer_side(inner, outer_center, outer_angle=0.0):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    ox, oy = outer_center
    # The outer hexagon has vertices at distance = side (circumradius = side for regular hex)
    # We need the smallest side such that all inner vertices are inside.
    # For a regular hexagon centered at (ox,oy) with rotation, a point (x,y) is inside
    # iff for each of the 6 edge normals, the dot product with the normal <= side * (apothem_factor)
    # Apothem = side * sqrt(3)/2. The support function in direction theta is side * cos(pi/6 - |theta mod pi/3 - pi/6|)
    # Actually for a regular hexagon with circumradius R=side, the width in direction theta is:
    # 2 * R * cos(min over k of |theta - k*pi/3| adjusted) ... 
    # Simpler: the hexagon with circumradius R has inradius R*sqrt(3)/2.
    # A point at distance d from center in direction phi is inside iff d <= R * cos(phi') where phi' is the angle to nearest vertex direction.
    # Vertices at angles: outer_angle + k*60 deg, k=0..5
    # For direction phi, the boundary distance is R * cos(delta) where delta = min_k |phi - (outer_angle + k*pi/3)| normalized to [0, pi/6]
    
    a = np.radians(outer_angle)
    vertex_angles = a + np.arange(6) * np.pi / 3
    
    # Collect all inner vertices
    all_pts = []
    for (cx, cy, ang) in inner:
        iv = _hex_verts(cx, cy, ang)
        all_pts.append(iv)
    all_pts = np.vstack(all_pts)
    
    dx = all_pts[:, 0] - ox
    dy = all_pts[:, 1] - oy
    dists = np.sqrt(dx**2 + dy**2)
    phis = np.arctan2(dy, dx)
    
    # For each point, find the required circumradius
    # The hexagon boundary in direction phi: R_req = dist / cos(delta) where delta is angular distance to nearest vertex
    # Actually the support: the hexagon with circumradius R contains point at (d, phi) iff
    # d <= R * h(phi) where h(phi) = max over edges... 
    # Let me use the half-plane formulation.
    # Edge i is perpendicular to direction (vertex_angles[i] + pi/6) ... no.
    # The 6 edges of a regular hexagon with vertices at angles theta_k = a + k*pi/3:
    # Edge between vertex k and k+1 has outward normal at angle a + (k+0.5)*pi/3
    # The half-plane: dot(p, n_k) <= R * cos(pi/6) = R * sqrt(3)/2  where n_k is unit normal
    
    normals = np.column_stack([np.cos(a + (np.arange(6) + 0.5) * np.pi / 3),
                                np.sin(a + (np.arange(6) + 0.5) * np.pi / 3)])
    # For each point, the constraint is: dot(p - center, n_k) <= R * sqrt(3)/2 for all k
    # So R >= max over all points and all k of dot(p-center, n_k) / (sqrt(3)/2)
    
    dots = all_pts[:, :2]  # dx, dy
    dots = np.column_stack([dx, dy])
    # dots shape (M, 2), normals shape (6, 2)
    proj = dots @ normals.T  # (M, 6)
    max_proj = proj.max(axis=1)  # (M,)
    R_req = (max_proj / (np.sqrt(3) / 2)).max()
    
    return max(R_req, 0.0)

def _objective(params):
    """Objective: minimize outer side length. params = [x0,y0,a0, x1,y1,a1, ...]"""
    inner = []
    for i in range(N):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        inner.append((x, y, a))
    
    # Check disjointness
    if not _hexagons_disjoint(inner, eps=1e-8):
        return 1e6  # penalty
    
    # Compute outer center as centroid of inner centers
    cx = np.mean([p[0] for p in inner])
    cy = np.mean([p[1] for p in inner])
    
    side = _min_outer_side(inner, (cx, cy), 0.0)
    
    # Verify containment
    if not _outer_hex_contains((cx, cy), side, 0.0, inner, eps=1e-8):
        return 1e6
    
    return side

def _local_polish(params, time_budget=30.0):
    """Local optimization to refine the packing."""
    start = time.time()
    best_params = params.copy()
    best_val = _objective(params)
    
    for _ in range(50):
        if time.time() - start > time_budget:
            break
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            break
        if best_val < 4.0:
            break
    return best_params, best_val

def construct():
    t0 = time.time()
    
    # Initial guess: honeycomb-like arrangement
    # 12 hexagons in a compact cluster
    # Try a 3-4-3-2 arrangement or similar
    # Honeycomb: center + 6 around + 5 more
    # Let's start with a grid-like pattern and let DE optimize
    
    # Simple initial: 4x3 grid with hexagonal offset
    init_x = []
    init_y = []
    init_a = []
    # 3 rows: 4, 4, 4
    for row in range(3):
        for col in range(4):
            x = col * 1.74  # ~sqrt(3)
            y = row * 1.5
            if row % 2 == 1:
                x += 0.87
            init_x.append(x)
            init_y.append(y)
            init_a.append(0.0)
    
    init_params = np.array(init_x + init_y + init_a)
    
    # Bounds for DE
    bounds = [(-5, 5)] * N + [(-5, 5)] * N + [(-180, 180)] * N
    
    def de_objective(params):
        return _objective(params)
    
    # Run differential evolution
    best_val = 1e6
    best_params = init_params.copy()
    
    if time.time() - t0 < DEADLINE - 60:
        try:
            de_res = differential_evolution(
                de_objective, bounds,
                maxiter=100, popsize=15,
                seed=42, tol=1e-8,
                mutation=(0.3, 0.8), recombination=0.9,
                polish=False, init='latinhypercube',
                updating='deferred'
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception as e:
            pass
    
    # Local polish
    if time.time() - t0 < DEADLINE - 20:
        try:
            polished_params, polished_val = _local_polish(best_params, time_budget=min(30, DEADLINE - (time.time() - t0) - 15))
            if polished_val < best_val:
                best_val = polished_val
                best_params = polished_params
        except:
            pass
    
    # Build result
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    cx = float(np.mean([p[0] for p in inner]))
    cy = float(np.mean([p[1] for p in inner]))
    side = float(best_val)
    
    # Final verification and fix
    if not _hexagons_disjoint(inner, eps=1e-7):
        # Fallback: use a known valid configuration
        # Simple valid: 12 hexagons in a line won't work, use compact
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        cx, cy = 0.0, 0.0
        side = 40.0
    
    if not _outer_hex_contains((cx, cy), side, 0.0, inner, eps=1e-7):
        # Recompute side
        side = _min_outer_side(inner, (cx, cy), 0.0) + 0.01
    
    return inner, (cx, cy), float(side), 0.0
