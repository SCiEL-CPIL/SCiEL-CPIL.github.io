"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side.

Strategy:
  - Use a dense, rigid honeycomb cluster (or brick variants) as the inner shape.
  - The cluster is treated as a single rigid body with global translation and rotation.
  - The outer hexagon is also a rigid body (translation + rotation).
  - This reduces the problem to optimizing 5 parameters: (cx, cy, theta_inner, cx_outer, cy_outer, theta_outer)
    Actually, since we can fix the outer hexagon's rotation to 0 WLOG (by rotating the whole system),
    and fix the outer center to (0,0) WLOG (by translating), the problem reduces to:
    Minimize side length S such that there exists a rotation theta and translation (tx, ty) of the cluster
    that fits inside the outer hexagon of side S centered at (0,0) with rotation 0.
    
    Wait, the outer hexagon can rotate too. But due to symmetry, the optimal solution likely has
    the outer hexagon aligned with the cluster or at a specific angle.
    Let's parameterize:
    - Cluster shape: fixed set of 12 points (centers) and angles (all 0 relative to cluster frame).
    - Cluster transform: (tx, ty, theta_c)
    - Outer hexagon: side S, center (0,0) (WLOG), rotation theta_o.
    
    Actually, let's keep it general but efficient.
    We will search over:
    1. Cluster structure (a few candidates: 6x2, 3x4 brick, etc.)
    2. Global rotation of the cluster relative to the outer hexagon.
    3. Translation of the cluster.
    4. The outer hexagon side length is determined by the max projection of the cluster vertices
       onto the outer hexagon normals.
    
    Since the outer hexagon can also rotate, we have 2 rotational degrees of freedom.
    However, the relative rotation is what matters. Let's fix the outer hexagon at 0 rotation
    and search over the cluster rotation. This is sufficient because any configuration
    (cluster_rot, outer_rot) is equivalent to (cluster_rot - outer_rot, 0).
    
    So, for a fixed cluster shape and fixed outer rotation (0), we need to find the best
    (tx, ty, theta_cluster) that minimizes the required outer side length.
    
    The required side length for a given (tx, ty, theta_cluster) is:
    S = max_{v in cluster_verts} (max_{k=0..5} dot(v - center_outer, n_k)) * 2/sqrt(3)
    where n_k are the normals of the outer hexagon (fixed at 0 rotation).
    Since center_outer is (0,0), S = max_{v} (max_k dot(v, n_k)) * 2/sqrt(3).
    Wait, this formula assumes the hexagon is centered at the origin.
    The condition for a point p to be inside a regular hexagon of side S centered at origin with normals n_k is:
    dot(p, n_k) <= S * sqrt(3)/2 for all k.
    So S >= max_k dot(p, n_k) * 2/sqrt(3) for all p.
    Thus S_min = max_{p in cluster} (max_k dot(p, n_k)) * 2/sqrt(3).
    
    This is a function of (tx, ty, theta_cluster).
    f(tx, ty, theta) = max_{i=0..11} ( max_{k=0..5} dot(R(theta) * p_i + t, n_k) ) * 2/sqrt(3)
    
    We want to minimize f(tx, ty, theta).
    
    This is a 3D optimization problem for each cluster candidate.
    We can use Nelder-Mead or Powell.
    We can also do a coarse grid search over theta, and then optimize (tx, ty) analytically or via 2D optimization.
    For a fixed theta, f(tx, ty) is the maximum of a set of linear functions in tx, ty.
    Minimizing the max of linear functions is a linear programming problem (or can be solved by finding the center of the intersection of half-planes, but here it's simpler).
    Actually, for fixed theta, the vertices of the cluster are fixed.
    The function g(tx, ty) = max_{i,k} dot(R(theta) p_i + (tx, ty), n_k)
    is convex in (tx, ty).
    We can minimize it using scipy.optimize.minimize.
    
    Plan:
    1. Define cluster candidates.
    2. For each candidate:
       a. Coarse scan over theta_cluster (0 to 60 deg).
       b. For each theta, minimize over (tx, ty) using Nelder-Mead or similar.
       c. Keep the best (theta, tx, ty, S).
    3. Return the best configuration.
"""

import math
import time
import numpy as np

N = 12
DEADLINE = 140.0

def construct():
    t0 = time.time()

    def elapsed():
        return time.time() - t0

    def hex_vertices(cx, cy, side, angle_rad):
        angs = angle_rad + np.array([0.0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3])
        return np.column_stack([cx + side * np.cos(angs),
                                cy + side * np.sin(angs)])

    def point_in_hex(px, py, cx, cy, side, angle_rad):
        dx, dy = px - cx, py - cy
        c, s = math.cos(-angle_rad), math.sin(-angle_rad)
        lx = dx * c - dy * s
        ly = dx * s + dy * c
        apothem = side * math.sqrt(3) / 2
        for k in range(6):
            na = math.radians(30 + 60 * k)
            if lx * math.cos(na) + ly * math.sin(na) > apothem + 1e-9:
                return False
        return True

    def all_inside(inner, oc, os, oa):
        for x, y, a in inner:
            vs = hex_vertices(x, y, 1.0, a)
            for vx, vy in vs:
                if not point_in_hex(vx, vy, oc[0], oc[1], os, oa):
                    return False
        return True

    def sat_min_sep_vec(v1, v2):
        e1 = np.roll(v1, -1, axis=0) - v1
        n1 = np.column_stack([-e1[:, 1], e1[:, 0]])
        n1 /= (np.linalg.norm(n1, axis=1, keepdims=True) + 1e-12)
        
        e2 = np.roll(v2, -1, axis=0) - v2
        n2 = np.column_stack([-e2[:, 1], e2[:, 0]])
        n2 /= (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-12)
        
        normals = np.vstack([n1, n2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min_p1 = p1.min(axis=0)
        max_p1 = p1.max(axis=0)
        min_p2 = p2.min(axis=0)
        max_p2 = p2.max(axis=0)
        
        sep1 = min_p2 - max_p1
        sep2 = min_p1 - max_p2
        seps = np.maximum(sep1, sep2)
        
        return seps.min()

    def pairwise_disjoint(inner):
        verts = [hex_vertices(x, y, 1.0, a) for x, y, a in inner]
        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                if sat_min_sep_vec(verts[i], verts[j]) < -1e-9:
                    return False
        return True

    def is_valid(inner, oc, os, oa):
        return all_inside(inner, oc, os, oa) and pairwise_disjoint(inner)

    # Cluster generators
    def make_6x2():
        pts = []
        for r in range(2):
            for c in range(6):
                x = (c - 2.5) * 3.0
                y = (r - 0.5) * math.sqrt(3)
                pts.append((x, y))
        return pts

    def make_3x4_brick():
        pts = []
        for r in range(3):
            for c in range(4):
                x = (c - 1.5) * math.sqrt(3) + (math.sqrt(3) / 2 if r % 2 else 0)
                y = (r - 1.0) * 1.5
                pts.append((x, y))
        return pts

    def make_4x3_brick():
        pts = []
        for r in range(4):
            for c in range(3):
                x = (c - 1.0) * math.sqrt(3) + (math.sqrt(3) / 2 if r % 2 else 0)
                y = (r - 1.5) * 1.5
                pts.append((x, y))
        return pts

    # Hexagonal cluster: 1 center, 6 ring, 5 ring2 (subset)
    def make_hex_cluster():
        pts = [(0, 0)]
        for k in range(6):
            a = math.pi / 3 * k
            pts.append((math.sqrt(3) * math.cos(a), math.sqrt(3) * math.sin(a)))
        # Ring 2: dist 3? No, dist between centers in hex packing is sqrt(3) for touching.
        # Center to ring1 is sqrt(3). Ring1 to ring2 is sqrt(3).
        # So ring2 centers are at distance 2*sqrt(3)? No.
        # In a hex lattice with spacing d=sqrt(3), the neighbors of a neighbor are at distance d or 2d?
        # Let's stick to the brick patterns which are simpler and known to be compact.
        return make_3x4_brick()

    candidates = [
        ("6x2", make_6x2()),
        ("3x4", make_3x4_brick()),
        ("4x3", make_4x3_brick()),
    ]

    best_result = None

    for name, pts in candidates:
        if elapsed() > DEADLINE - 10:
            break
            
        pts = np.array(pts)
        
        # We want to minimize S(tx, ty, theta)
        # S = max_{i,k} dot(R(theta) pts[i] + t, n_k) * 2/sqrt(3)
        # n_k are fixed for outer hexagon at 0 rotation.
        # n_k = (cos(30+60k), sin(30+60k))
        
        normals = np.array([
            [math.cos(math.radians(30 + 60*k)), math.sin(math.radians(30 + 60*k))]
            for k in range(6)
        ])
        
        def objective(params):
            tx, ty, theta = params
            R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
            # Rotate points
            rotated_pts = pts @ R.T
            # Translate
            translated_pts = rotated_pts + np.array([tx, ty])
            
            # Project onto normals
            # (12, 2) @ (2, 6) -> (12, 6)
            projs = translated_pts @ normals.T
            # Max over points for each normal
            max_projs = projs.max(axis=0) # (6,)
            # Max over normals
            max_proj = max_projs.max()
            
            # Side length
            return max_proj * 2.0 / math.sqrt(3)

        best_s = float('inf')
        best_params = None
        
        # Coarse scan over theta
        for deg in range(0, 60, 5):
            if elapsed() > DEADLINE - 5:
                break
            theta = math.radians(deg)
            
            # Initial guess for tx, ty: center of mass
            cm = pts.mean(axis=0)
            R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
            cm_rot = R @ cm
            # We want to center the cluster in the hexagon, so t should be -cm_rot?
            # The hexagon is centered at (0,0). So we want the cluster center to be near (0,0).
            tx0 = -cm_rot[0]
            ty0 = -cm_rot[1]
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, [tx0, ty0, theta], method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_s:
                    best_s = res.fun
                    best_params = res.x
            except Exception:
                pass

        if best_params is not None and best_s < best_result[0] if best_result else True:
            tx, ty, theta = best_params
            R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
            inner = []
            for px, py in pts:
                rx, ry = R @ np.array([px, py])
                inner.append([rx + tx, ry + ty, theta])
            
            # Verify validity
            if is_valid(inner, (0,0), best_s, 0.0):
                best_result = (best_s, inner, [0.0, 0.0], best_s, 0.0)
            else:
                # If invalid due to floating point or subtle overlap, try to fix by slightly increasing S
                # But we should have ensured disjointness in the cluster generation.
                # The issue might be containment.
                # Let's recalculate S precisely from the vertices.
                all_verts = np.vstack([hex_vertices(x, y, 1.0, a) for x, y, a in inner])
                projs = all_verts @ normals.T
                max_projs = projs.max(axis=0)
                max_proj = max_projs.max()
                calc_s = max_proj * 2.0 / math.sqrt(3)
                if calc_s < best_s + 1e-6:
                    best_s = calc_s
                    best_result = (best_s, inner, [0.0, 0.0], best_s, 0.0)

    if best_result is None:
        # Fallback
        inner = []
        for r in range(2):
            for c in range(6):
                x = (c - 2.5) * 3.0
                y = (r - 0.5) * math.sqrt(3)
                inner.append([x, y, 0.0])
        best_result = (9.0, inner, [0.0, 0.0], 9.0, 0.0)

    best_os, best_inner, best_oc, best_oa = best_result[0], best_result[1], best_result[2], best_result[4]

    # Local polish: try to shrink S slightly
    if elapsed() < DEADLINE - 2:
        step = 0.005
        for _ in range(100):
            if elapsed() > DEADLINE - 1:
                break
            new_os = best_os - step
            if new_os < 2.0:
                break
            if is_valid(best_inner, best_oc, new_os, best_oa):
                best_os = new_os
            else:
                step *= 0.5
                if step < 1e-6:
                    break

    return best_inner, tuple(best_oc), best_os, best_oa
