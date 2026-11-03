import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 12
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check disjointness via segment intersection
    def _seg_seg_dist(p1, p2, p3, p4):
        d1 = p2 - p1
        d2 = p4 - p3
        r = p1 - p3
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        
        if a < 1e-12 and e < 1e-12:
            return np.linalg.norm(r)
        elif a < 1e-12:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
            return np.linalg.norm(r + t * d2)
        elif e < 1e-12:
            s = -np.dot(d1, r) / a
            s = max(0.0, min(1.0, s))
            return np.linalg.norm(s * d1 - r)
        
        c = np.dot(d1, r)
        b = np.dot(d1, d2)
        denom = a * e - b * b
        
        if denom < 1e-12:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
            s = max(0.0, min(1.0, (b * t - c) / a))
        else:
            s = (b * f - c * e) / denom
            if s < 0.0 or s > 1.0:
                s = 0.0
                t = f / e
                t = max(0.0, min(1.0, t))
            else:
                t = (b * s + f) / e
                t = max(0.0, min(1.0, t))
                if t < 0.0:
                    t = 0.0
                    s = max(0.0, min(1.0, -c / a))
                elif t > 1.0:
                    t = 1.0
                    s = max(0.0, min(1.0, (b - c) / a))
        
        return np.linalg.norm(s * d1 - r - t * d2)

    def _polygons_overlap(v1, v2, eps=1e-9):
        n1, n2 = len(v1), len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i + 1) % n1]
            for j in range(n2):
                q1 = v2[j]
                q2 = v2[(j + 1) % j if False else (j + 1) % n2]
                if _seg_seg_dist(p1, p2, q1, q2) < eps:
                    return True
        return False

    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _polygons_overlap(inner_verts[i], inner_verts[j], eps=1e-9):
                return False, "overlap"
            
            # Check containment
            is_inside = True
            for p in inner_verts[i]:
                inside = True
                for k in range(6):
                    a = outer_verts[k]
                    b = outer_verts[(k+1)%6]
                    cross_ab = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                    if cross_ab < -eps:
                        inside = False
                        break
                if not inside:
                    is_inside = False
                    break
            
            if not is_inside:
                return False, "outside"
                
    return True, "valid"

def _objective(x, outer_angle):
    """
    Objective function for optimization.
    x: 2N vector of inner hexagon centers (x1, y1, x2, y2, ..., xN, yN)
    outer_angle: angle of the outer hexagon
    Returns: (value, grad) where value is the required outer side length
    """
    centers = x.reshape(N, 2)
    
    # Compute the maximum projection of all inner hexagon vertices onto the outer normals
    # For a regular hexagon with side S, the apothem is S * sqrt(3) / 2
    # The distance from center to each side is the apothem.
    # We need to find the maximum distance from the outer center to any point in the union of inner hexagons,
    # projected onto the 6 normals of the outer hexagon.
    
    # Actually, it's simpler: the condition for an inner hexagon to be inside the outer hexagon
    # is that all its vertices are inside.
    # For a convex outer hexagon defined by 6 half-planes, a point P is inside if
    # dot(P - C, n_k) <= apothem for all k, where n_k are the outward normals.
    
    # So we need to compute max over all inner vertices v and all normals n_k of dot(v - C, n_k).
    # Then S * sqrt(3)/2 >= max_val, so S >= max_val * 2 / sqrt(3).
    
    max_proj = 0.0
    for i in range(N):
        c = centers[i]
        # Vertices of inner hexagon i
        # Inner hexagons have angle 0 (fixed in this strategy for simplicity, or we could optimize)
        # Let's fix inner angle to 0 for now to reduce variables.
        # Vertices are at c + (cos(60k), sin(60k))
        verts = c + np.array([[np.cos(np.pi/3*k), np.sin(np.pi/3*k)] for k in range(6)])
        
        for k in range(6):
            angle = np.deg2rad(outer_angle + k * 60)
            normal = np.array([np.cos(angle), np.sin(angle)])
            for v in verts:
                proj = np.dot(v, normal)
                if proj > max_proj:
                    max_proj = proj
    
    side = max_proj * 2.0 / np.sqrt(3.0)
    
    # Gradient
    # side = (2/sqrt(3)) * max_proj
    # max_proj = max_{i,k,v} dot(c_i + v_rel, n_k)
    # This is a max of linear functions, so it's convex.
    # The gradient is (2/sqrt(3)) * n_k* where k* is the active constraint.
    
    # Find the active constraint
    best_proj = -np.inf
    best_normal = None
    best_idx = None
    for i in range(N):
        c = centers[i]
        verts = c + np.array([[np.cos(np.pi/3*k), np.sin(np.pi/3*k)] for k in range(6)])
        for k in range(6):
            angle = np.deg2rad(outer_angle + k * 60)
            normal = np.array([np.cos(angle), np.sin(angle)])
            for v_idx, v in enumerate(verts):
                proj = np.dot(v, normal)
                if proj > best_proj:
                    best_proj = proj
                    best_normal = normal
                    best_idx = i
    
    grad = np.zeros(2 * N)
    if best_normal is not None:
        grad[2*best_idx] = (2.0 / np.sqrt(3.0)) * best_normal[0]
        grad[2*best_idx + 1] = (2.0 / np.sqrt(3.0)) * best_normal[1]
    
    return side, grad

def _collision_penalty(x, centers, threshold=1.8):
    """
    Penalty for overlapping inner hexagons.
    We want the distance between centers to be at least 2*sqrt(3) ~ 3.464?
    No, for two unit hexagons with same orientation, the minimum distance between centers
    to be disjoint is the sum of their apothems? No.
    The distance between parallel sides is sqrt(3).
    If they are aligned, the center distance must be at least 2 * (sqrt(3)/2) = sqrt(3) ~ 1.732?
    Wait, if two hexagons are side-by-side, sharing a side, the distance between centers is 2 * apothem = sqrt(3).
    So if dist < sqrt(3), they overlap.
    So we need dist >= sqrt(3).
    """
    penalty = 0.0
    grad = np.zeros(2 * N)
    for i in range(N):
        for j in range(i + 1, N):
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < threshold:
                # Quadratic penalty
                # Penalty = (threshold - dist)^2
                # Grad with respect to center i: -2*(threshold - dist) * (d / dist)
                # Grad with respect to center j: +2*(threshold - dist) * (d / dist)
                if dist > 1e-6:
                    diff = threshold - dist
                    factor = 2.0 * diff / dist
                    grad[2*i] -= factor * d[0]
                    grad[2*i + 1] -= factor * d[1]
                    grad[2*j] += factor * d[0]
                    grad[2*j + 1] += factor * d[1]
                penalty += diff * diff
    return penalty, grad

def _combined_objective(x, outer_angle, penalty_weight=10.0):
    """
    Combined objective: Side length + Penalty for collisions.
    """
    centers = x.reshape(N, 2)
    
    obj1, grad1 = _objective(x, outer_angle)
    pen, grad_pen = _collision_penalty(x, centers, threshold=np.sqrt(3) + 0.01)
    
    value = obj1 + penalty_weight * pen
    grad = grad1 + penalty_weight * grad_pen
    
    return value, grad

def construct():
    start_time = time.time()
    
    # Strategy:
    # 1. Generate a set of candidate initial configurations.
    # 2. For each, use L-BFGS-B to minimize the outer side length, subject to collision penalties.
    # 3. Verify the final result.
    
    # We will fix the outer angle to 0 and 30 degrees and try both.
    # We will fix the inner angles to 0 degrees.
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Generate initial guesses
    # Guess 1: Triangular lattice cluster
    # Guess 2: Random
    # Guess 3: Hexagonal ring
    
    initial_guesses = []
    
    # Guess 1: Compact cluster
    # Points on triangular lattice
    lattice_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            lattice_pts.append([x, y])
    lattice_pts = np.array(lattice_pts)
    
    # Greedy pick 12 points with min distance
    picked = []
    sorted_idx = np.argsort(np.sum(lattice_pts**2, axis=1))
    for idx in sorted_idx:
        p = lattice_pts[idx]
        valid = True
        for q in picked:
            if np.linalg.norm(p - q) < np.sqrt(3) * 0.99:
                valid = False
                break
        if valid:
            picked.append(p)
            if len(picked) == 12:
                break
    if len(picked) == 12:
        initial_guesses.append(np.array(picked).flatten())
    
    # Guess 2: Random
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.uniform(-3, 3, (N, 2))
        # Simple rejection sampling to avoid obvious overlaps
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(pts[i] - pts[j]) < np.sqrt(3):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            initial_guesses.append(pts.flatten())
    
    # Guess 3: Hexagonal arrangement
    # Center + 6 neighbors + 5 more
    pts = [[0,0]]
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)])
    # Add 5 more from next ring
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle)])
        if len(pts) == 12:
            break
    initial_guesses.append(np.array(pts).flatten())
    
    # Optimize
    for guess in initial_guesses:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        for outer_angle in [0.0, 30.0]:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            try:
                # Use L-BFGS-B
                # We need to handle the non-smoothness of the max function.
                # L-BFGS-B might struggle. Let's try SLSQP or just use the gradient from the active constraint.
                # The gradient calculation in _objective is for the active constraint, which is correct for the max function.
                
                res = minimize(
                    _combined_objective,
                    guess,
                    args=(outer_angle, 10.0),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                x_opt = res.x
                centers_opt = x_opt.reshape(N, 2)
                
                # Compute the actual side length and check validity
                # We need to recompute the side length carefully.
                # The optimization minimized side + penalty.
                # We want the side length such that all hexagons are inside.
                
                # Let's compute the required side length for the optimized positions.
                max_proj = 0.0
                for i in range(N):
                    c = centers_opt[i]
                    verts = c + np.array([[np.cos(np.pi/3*k), np.sin(np.pi/3*k)] for k in range(6)])
                    for k in range(6):
                        angle = np.deg2rad(outer_angle + k * 60)
                        normal = np.array([np.cos(angle), np.sin(angle)])
                        for v in verts:
                            proj = np.dot(v, normal)
                            if proj > max_proj:
                                max_proj = proj
                
                side = max_proj * 2.0 / np.sqrt(3.0)
                
                # Check validity
                inner_hexes = [[centers_opt[i][0], centers_opt[i][1], 0.0] for i in range(N)]
                # Center of outer hexagon is (0,0) in our optimization frame.
                # But we can translate the whole thing.
                # The optimization was done with outer center at (0,0).
                # So outer center is (0,0).
                
                is_valid, reason = _check_validity(inner_hexes, side, (0.0, 0.0), outer_angle)
                
                if is_valid and side < best_side:
                    best_side = side
                    best_inner = inner_hexes
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = outer_angle
                    
            except Exception as e:
                pass
    
    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
