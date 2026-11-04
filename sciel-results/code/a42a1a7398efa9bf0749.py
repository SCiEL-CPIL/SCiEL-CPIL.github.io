import numpy as np
import time
import math
import scipy.optimize

N = 11
D = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    
    if a < 1e-12:
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
        s = c / a
        s = max(0, min(1, s))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """Compute the minimum distance between two convex polygons."""
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _compute_side_fast(inner_params, outer_angle_deg):
    """
    Fast computation of minimum outer hexagon side length using support function.
    inner_params: array of shape (n, 3) with [x, y, angle_deg]
    """
    n = len(inner_params)
    # Compute all vertices
    all_verts = np.zeros((n * 6, 2))
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        all_verts[i*6:(i+1)*6] = _hex_vertices(x, y, a)
    
    # Outer hexagon normals
    phi_rad = np.deg2rad(outer_angle_deg)
    normals = np.array([
        [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
        for k in range(6)
    ])
    
    projs = all_verts @ normals.T  # (66, 6)
    max_proj = np.max(projs)
    return 2.0 * max_proj / np.sqrt(3)

def _overlap_penalty_fast(inner_params, penalty_weight=50.0):
    """Compute penalty for overlapping inner hexagons using center distances."""
    n = len(inner_params)
    penalty = 0.0
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation(v1, v2)
            if sep < 0.005:
                penalty += penalty_weight * (0.005 - sep) ** 2
    
    return penalty

def _seed_ring():
    """Concentric ring seed solution."""
    eps = 1e-8
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (D + eps) * np.cos(angle)
        y = (D + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    return inner

def _seed_lattice():
    """Triangular lattice cluster seed."""
    d = np.sqrt(3)
    u = np.array([1.5, d/2])
    v = np.array([0.0, d])
    
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]
    
    cluster_centers = np.array([i * u + j * v for i, j in lattice_coords])
    cluster_centers *= 1.002
    
    inner = []
    for i in range(N):
        inner.append([float(cluster_centers[i, 0]), float(cluster_centers[i, 1]), 0.0])
    return inner

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Start with the ring seed (known to score ~10.0)
    best_inner = _seed_ring()
    best_side = _compute_side_fast(np.array(best_inner), 30.0)
    best_outer_angle = 30.0
    
    # Also try lattice seed
    lattice_inner = _seed_lattice()
    lat_side = _compute_side_fast(np.array(lattice_inner), 0.0)
    if lat_side < best_side:
        best_inner = lattice_inner
        best_side = lat_side
        best_outer_angle = 0.0
    
    # Objective function for optimization
    def objective(params, outer_angle):
        inner_params = params.reshape(N, 3)
        side = _compute_side_fast(inner_params, outer_angle)
        pen = _overlap_penalty_fast(inner_params)
        return side + pen
    
    # Phase 1: Nelder-Mead optimization of inner positions/angles
    init_params = np.array([v for hex_ in best_inner for v in hex_])
    
    try:
        if time.time() < deadline - 30:
            res = scipy.optimize.minimize(
                objective, init_params, args=(best_outer_angle,),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s = _compute_side_fast(res.x.reshape(N, 3), best_outer_angle)
            if s < best_side:
                best_inner = [[res.x[i*3], res.x[i*3+1], res.x[i*3+2]] for i in range(N)]
                best_side = s
    except Exception:
        pass
    
    # Phase 2: Optimize outer angle (1D)
    if time.time() < deadline - 20:
        def obj_outer(phi):
            return _compute_side_fast(np.array(best_inner), phi)
        
        try:
            # Grid search
            phi_vals = np.linspace(0, 60, 1201, endpoint=False)
            sides = np.array([obj_outer(p) for p in phi_vals])
            best_idx = np.argmin(sides)
            best_outer_angle = phi_vals[best_idx]
            best_side = sides[best_idx]
            
            # Fine refinement
            phi_fine = np.linspace(best_outer_angle - 0.5, best_outer_angle + 0.5, 1001)
            for p in phi_fine:
                s = obj_outer(p)
                if s < best_side:
                    best_side = s
                    best_outer_angle = p
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B refinement
    if time.time() < deadline - 15:
        init_params2 = np.array([v for hex_ in best_inner for v in hex_])
        try:
            res2 = scipy.optimize.minimize(
                objective, init_params2, args=(best_outer_angle,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            s = _compute_side_fast(res2.x.reshape(N, 3), best_outer_angle)
            if s < best_side:
                best_inner = [[res2.x[i*3], res2.x[i*3+1], res2.x[i*3+2]] for i in range(N)]
                best_side = s
        except Exception:
            pass
    
    # Phase 4: Joint refinement - optimize inner and outer together
    if time.time() < deadline - 10:
        def joint_obj(params):
            inner_p = params[:-1].reshape(N, 3)
            outer_a = params[-1]
            side = _compute_side_fast(inner_p, outer_a)
            pen = _overlap_penalty_fast(inner_p)
            return side + pen
        
        init_joint = np.concatenate([np.array([v for hex_ in best_inner for v in hex_]), [best_outer_angle]])
        try:
            res3 = scipy.optimize.minimize(
                joint_obj, init_joint,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            s = _compute_side_fast(res3.x[:-1].reshape(N, 3), res3.x[-1])
            if s < best_side:
                best_inner = [[res3.x[i*3], res3.x[i*3+1], res3.x[i*3+2]] for i in range(N)]
                best_outer_angle = float(res3.x[-1]) % 60.0
                best_side = s
        except Exception:
            pass
    
    # Phase 5: Final polish - independent scalar refinements
    if time.time() < deadline - 5:
        # Polish outer angle
        def obj_o(p):
            return _compute_side_fast(np.array(best_inner), p)
        try:
            res_o = scipy.optimize.minimize_scalar(
                obj_o, bounds=(best_outer_angle - 0.1, best_outer_angle + 0.1),
                method='bounded', options={'xatol': 1e-12}
            )
            s = obj_o(res_o.x)
            if s < best_side:
                best_side = float(s)
                best_outer_angle = float(res_o.x)
        except Exception:
            pass
        
        # Polish each hexagon's angle independently
        for i in range(N):
            if time.time() > deadline - 3:
                break
            def obj_ai(ai, idx=i):
                inner_copy = [list(h) for h in best_inner]
                inner_copy[idx][2] = ai
                return _compute_side_fast(np.array(inner_copy), best_outer_angle)
            try:
                cur_a = best_inner[i][2]
                res_ai = scipy.optimize.minimize_scalar(
                    obj_ai, bounds=(cur_a - 5, cur_a + 5),
                    method='bounded', options={'xatol': 1e-10}
                )
                s = obj_ai(res_ai.x)
                if s < best_side:
                    best_inner[i][2] = float(res_ai.x)
                    best_side = float(s)
            except Exception:
                pass
    
    # Phase 6: Small perturbations to explore nearby configurations
    if time.time() < deadline - 5:
        rng = np.random.default_rng(42)
        for trial in range(5):
            if time.time() > deadline - 3:
                break
            # Perturb positions slightly
            perturbed = [list(h) for h in best_inner]
            for i in range(N):
                perturbed[i][0] += rng.normal(0, 0.02)
                perturbed[i][1] += rng.normal(0, 0.02)
            
            init_p = np.array([v for h in perturbed for v in h])
            try:
                res_p = scipy.optimize.minimize(
                    objective, init_p, args=(best_outer_angle,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                s = _compute_side_fast(res_p.x.reshape(N, 3), best_outer_angle)
                if s < best_side:
                    best_inner = [[res_p.x[i*3], res_p.x[i*3+1], res_p.x[i*3+2]] for i in range(N)]
                    best_side = s
            except Exception:
                pass
    
    # Final verification and safety margin
    final_side = float(best_side) + 1e-10
    
    # Ensure inner list is properly formatted
    inner = []
    for h in best_inner:
        inner.append([float(h[0]), float(h[1]), float(h[2])])
    
    return inner, (0.0, 0.0), final_side, float(best_outer_angle) % 60.0
