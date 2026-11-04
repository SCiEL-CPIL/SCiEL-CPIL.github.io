"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

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
    """Generate a valid seed solution using a hexagonal lattice with correct spacing."""
    # For unit regular hexagons, the distance between centers of touching hexagons
    # is 2*sqrt(3) (the distance between opposite edges is sqrt(3), so center-to-center is 2*sqrt(3))
    d = 2 * math.sqrt(3)
    centers = []
    angles = []
    
    # Central hexagon
    centers.append([0, 0])
    angles.append(0.0)
    
    # 6 surrounding at distance d in hexagonal directions
    for i in range(6):
        theta = i * math.pi / 3.0
        centers.append([d * math.cos(theta), d * math.sin(theta)])
        angles.append(0.0)
    
    # 4 more from second ring at distance 2*d
    ring2_positions = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_positions.append([x, y])
    
    # Pick 4 that are well-separated
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

def _seed_solution2():
    """Alternative seed: compact cluster with some rotation and correct spacing."""
    d = 2 * math.sqrt(3)
    centers = []
    angles = []
    
    # Central
    centers.append([0, 0])
    angles.append(0.0)
    
    # 6 surrounding with slight rotation
    for i in range(6):
        theta = i * math.pi / 3.0
        centers.append([d * math.cos(theta), d * math.sin(theta)])
        angles.append(30.0)  # rotated 30 degrees
    
    # 4 more: place them in gaps at slightly compressed distance
    gap_angles = [math.pi/6, math.pi/6 + math.pi/3, math.pi/6 + 2*math.pi/3, math.pi/6 + 3*math.pi/3]
    for ga in gap_angles:
        centers.append([2*d*0.9 * math.cos(ga), 2*d*0.9 * math.sin(ga)])
        angles.append(0.0)
    
    centers = np.array(centers)
    angles = np.array(angles)
    
    S = _compute_side_length(centers, angles) * 1.02
    return centers, angles, S

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin
    
    # Get multiple seed solutions
    seeds = []
    for seed_fn in [_seed_solution, _seed_solution2]:
        try:
            c, a, S = seed_fn()
            if _is_valid(c, a, S):
                seeds.append((c, a, S))
        except Exception:
            pass
    
    # Add a third seed with different rotation pattern
    try:
        d = 2 * math.sqrt(3)
        c3 = []
        a3 = []
        c3.append([0, 0])
        a3.append(0.0)
        for i in range(6):
            theta = i * math.pi / 3.0
            c3.append([d * math.cos(theta), d * math.sin(theta)])
            a3.append(0.0)
        # Place 4 more with alternating rotations
        for i in range(4):
            theta = (2*i + 1) * math.pi / 12.0
            c3.append([2*d * 0.95 * math.cos(theta), 2*d * 0.95 * math.sin(theta)])
            a3.append(15.0 if i % 2 == 0 else 45.0)
        c3 = np.array(c3)
        a3 = np.array(a3)
        S3 = _compute_side_length(c3, a3) * 1.03
        if _is_valid(c3, a3, S3):
            seeds.append((c3, a3, S3))
    except Exception:
        pass
    
    if not seeds:
        # Fallback
        d = 2 * math.sqrt(3) * 1.1
        c_fb = []
        a_fb = []
        c_fb.append([0, 0])
        a_fb.append(0.0)
        for i in range(6):
            theta = i * math.pi / 3.0
            c_fb.append([d * math.cos(theta), d * math.sin(theta)])
            a_fb.append(0.0)
        for i in range(4):
            theta = (2*i + 1) * math.pi / 12.0
            c_fb.append([2*d * math.cos(theta), 2*d * math.sin(theta)])
            a_fb.append(0.0)
        c_fb = np.array(c_fb)
        a_fb = np.array(a_fb)
        S_fb = _compute_side_length(c_fb, a_fb) * 1.05
        seeds.append((c_fb, a_fb, S_fb))
    
    best_S = min(s[2] for s in seeds)
    best_centers = seeds[0][0].copy()
    best_angles = seeds[0][1].copy()
    for s in seeds:
        if s[2] < best_S:
            best_S = s[2]
            best_centers = s[0].copy()
            best_angles = s[1].copy()
    
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
    
    # Define tighter bounds based on expected packing size
    max_extent = 8.0
    bounds = []
    for i in range(N):
        bounds.append((-max_extent, max_extent))  # x
        bounds.append((-max_extent, max_extent))  # y
    for i in range(N):
        bounds.append((0.0, 180.0))  # angle
    
    # Run differential evolution from best seed
    try:
        init_pop = [best_centers.flatten().tolist() + best_angles.tolist()]
        
        # Add other seeds
        for c, a, S in seeds:
            init_pop.append(c.flatten().tolist() + a.tolist())
        
        # Add some random starting points with better spread
        np.random.seed(42)
        for _ in range(10):
            # Generate random hexagonal lattice perturbations
            base_d = 2 * math.sqrt(3)
            c = []
            a = []
            c.append([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])
            a.append(np.random.uniform(0, 180))
            for i in range(6):
                theta = i * math.pi / 3.0 + np.random.uniform(-0.2, 0.2)
                dist = base_d * np.random.uniform(0.9, 1.1)
                c.append([dist * math.cos(theta), dist * math.sin(theta)])
                a.append(np.random.uniform(0, 180))
            for i in range(4):
                theta = (2*i + 1) * math.pi / 12.0 + np.random.uniform(-0.3, 0.3)
                dist = 2 * base_d * np.random.uniform(0.85, 1.05)
                c.append([dist * math.cos(theta), dist * math.sin(theta)])
                a.append(np.random.uniform(0, 180))
            init_pop.append(np.array(c).flatten().tolist() + a)
        
        res = differential_evolution(
            objective, bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.4, 1.2),
            recombination=0.7,
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
    if time.time() < deadline - 30:
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
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000, 'maxfev': 10000, 'initial_simplex': None})
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
    
    # Second round of DE from the improved solution if time permits
    if time.time() < deadline - 45:
        try:
            init_pop2 = [best_centers.flatten().tolist() + best_angles.tolist()]
            np.random.seed(123)
            for _ in range(8):
                # More diverse perturbations
                scale = np.random.uniform(0.2, 0.6)
                c = best_centers + np.random.normal(0, scale, size=(N, 2))
                a = np.mod(best_angles + np.random.normal(0, 20, size=N), 180)
                init_pop2.append(c.flatten().tolist() + a.tolist())
            
            res = differential_evolution(
                objective, bounds,
                seed=123,
                maxiter=60,
                popsize=12,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init=init_pop2,
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
    
    # Final refinement
    if time.time() < deadline - 15:
        def nm_objective2(params):
            if time.time() > deadline:
                return 1e10
            centers = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            S = _compute_side_length(centers, angles)
            pen = _overlap_penalty(centers, angles)
            return S + 1000.0 * pen
        
        x0 = best_centers.flatten().tolist() + best_angles.tolist()
        try:
            res = minimize(nm_objective2, x0, method='Nelder-Mead',
                           options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 3000, 'maxfev': 8000})
            if res.fun < best_S:
                params = res.x
                c = params[:2*N].reshape(N, 2)
                a = params[2*N:]
                S_val = _compute_side_length(c, a)
                pen_val = _overlap_penalty(c, a)
                if pen_val < 1e-6 and S_val < best_S:
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
    else:
        # Try to reduce S slightly if valid
        S_val = _compute_side_length(best_centers, best_angles)
        if S_val < best_S * 0.995:
            best_S = S_val * 1.001
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i][0]), float(best_centers[i][1]), float(best_angles[i])])
    
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
