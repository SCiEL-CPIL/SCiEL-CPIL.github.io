import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    sqrt3 = math.sqrt(3)

    # Unit hexagon vertices (flat-top): vertices at angles 0, 60, ..., 300
    unit_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    # Outward normals for the 6 sides of a unit hexagon
    # Side k connects vertex k to vertex k+1. Outward normal angle = 30 + 60*k degrees
    unit_normals = np.array([[math.cos(math.radians(30 + 60*k)), math.sin(math.radians(30 + 60*k))] for k in range(6)])

    def rot_matrix(angle_rad):
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return np.array([[c, -s], [s, c]])

    def hex_verts(cx, cy, angle_deg):
        """Vertices of unit hexagon at (cx,cy) rotated by angle_deg."""
        R = rot_matrix(math.radians(angle_deg))
        return unit_verts @ R.T + np.array([cx, cy])

    def hex_normals(angle_deg):
        """Outward normals of unit hexagon rotated by angle_deg."""
        R = rot_matrix(math.radians(angle_deg))
        return unit_normals @ R.T

    def outer_side_len(positions, angles, outer_angle_deg):
        """
        Correct minimum outer side length using Minkowski sum.
        The outer hexagon (centered at origin, rotated by outer_angle) must contain
        all inner hexagons. By Minkowski sum, the required apothem in direction n is:
            max over all inner hexagons of (support of (hex_i - hex_outer_shape))
        For a regular hexagon centered at origin with side s, the support in direction n
        (unit) is s * max_k (n . n_k) where n_k are its unit normals... 
        Actually for a regular hexagon of side s centered at origin, the support function
        in direction u (unit) is: s * (sqrt(3)/2) * max over the 6 normal directions of (u . n_k)
        Wait, let me think more carefully.
        
        A regular hexagon with side length s, centered at origin, has inradius r = s*sqrt(3)/2.
        The support function h(u) = max_{x in hex} u.x.
        For a regular hexagon with normals at angles 30+60k, the support in direction u is:
            h(u) = r * max_k (u . n_k)  ... no, that's not right either.
        
        Actually, for a regular hexagon with side s (flat-top, vertices at radius s),
        the vertices are at distance s from center. The inradius is s*sqrt(3)/2.
        The support function in direction u is the max over vertices of u.v.
        
        For containment: hex_outer (side S, rotated by theta) contains hex_inner (side 1, at pos p, rotated by phi)
        iff for all directions, the support of hex_inner in that direction <= support of hex_outer.
        
        But since we want the MINIMUM S such that containment holds, and the outer hexagon
        is centered at origin, we need:
            For each of the 6 side-normals n_k of the outer hexagon:
                max over all inner hex vertices of (n_k . v) <= S * sqrt(3)/2
            (since the outer hexagon's side in direction n_k is at distance S*sqrt(3)/2 from center)
        
        Wait, that's not quite right. The outer hexagon with side S has its sides at distance
        S*sqrt(3)/2 from the center (the inradius). The normal n_k points outward.
        The constraint is: for each outer normal direction n_k, all inner hex vertices must
        satisfy n_k . v <= S*sqrt(3)/2.
        
        But also, the inner hexagons extend in directions between the outer normals.
        The correct condition is: the Minkowski sum of the outer hexagon and (-inner hexagon)
        must contain the origin... 
        
        Let me use the correct formulation:
        Outer hexagon H_out (centered at origin, side S, rotation theta) contains inner hexagon H_in
        (centered at p, side 1, rotation phi) iff:
            H_in - H_out is contained in... no.
        
        H_out contains H_in iff for every direction u, h_{H_in}(u) - p.u <= h_{H_out}(u).
        Where h is the support function.
        
        For a regular hexagon centered at origin with side s and rotation alpha:
        h(u) = s * max_{k=0..5} (u . v_k(alpha)) where v_k are the unit vertex directions.
        Actually, h(u) = max over vertices. The vertices are at s*(cos(60k+alpha), sin(60k+alpha)).
        So h(u) = s * max_k (u_x cos(60k+alpha) + u_y sin(60k+alpha))
                = s * max_k (|u| cos(angle(u) - 60k - alpha))
        
        For unit |u|: h(u) = s * max_k cos(angle(u) - 60k - alpha)
        
        The maximum of cos over 6 equally spaced angles is always >= cos(30°) = sqrt(3)/2,
        and equals 1 when u aligns with a vertex.
        
        So: S must satisfy for all directions u:
            1 * max_k cos(angle(u) - 60k - phi) - p.u <= S * max_k cos(angle(u) - 60k - theta)
        
        This is complex. Let me use a simpler but correct approach:
        The outer hexagon contains the inner hexagon iff all 6 vertices of the inner hexagon
        are inside the outer hexagon AND all 6 sides of the inner hexagon don't cross the outer.
        
        Actually, the simplest correct check: the outer hexagon is the intersection of 6 half-planes.
        Each half-plane is defined by an outer normal n_k and distance d = S*sqrt(3)/2.
        Point v is inside iff n_k . v <= d for all k.
        
        So the constraint is: for each inner hex vertex v, for each outer normal n_k:
            n_k . v <= S * sqrt(3)/2
        
        Therefore: S >= (2/sqrt(3)) * max over all inner vertices v, all outer normals n_k of (n_k . v)
        
        This is CORRECT because a convex polygon is the intersection of its half-planes.
        """
        theta = math.radians(outer_angle_deg)
        # Outer hexagon normals (6 directions)
        outer_normals = np.array([[math.cos(theta + math.pi/6 + k*math.pi/3), 
                                   math.sin(theta + math.pi/6 + k*math.pi/3)] for k in range(6)])
        
        max_proj = 0.0
        for i in range(N):
            verts = hex_verts(positions[i][0], positions[i][1], angles[i])
            projs = verts @ outer_normals.T  # shape (6, 6)
            max_proj = max(max_proj, np.max(projs))
        
        return max_proj * 2.0 / sqrt3

    def min_gap_smooth(poly1, poly2, beta=50.0):
        """
        Smooth approximation of the minimum gap between two convex polygons.
        Uses log-sum-exp to create a smooth minimum of the 12 SAT axis gaps.
        Returns a smooth value that is negative when overlapping.
        """
        gaps = []
        # Axes from poly1
        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                continue
            axis /= norm
            p1 = poly1 @ axis
            p2 = poly2 @ axis
            gap = max(np.min(p1) - np.max(p2), np.min(p2) - np.max(p1))
            gaps.append(gap)
        # Axes from poly2
        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                continue
            axis /= norm
            p1 = poly1 @ axis
            p2 = poly2 @ axis
            gap = max(np.min(p1) - np.max(p2), np.min(p2) - np.max(p1))
            gaps.append(gap)
        
        if not gaps:
            return 0.0
        
        gaps = np.array(gaps)
        # Soft minimum: -logsumexp(-gaps) / beta
        # This approximates min(gaps) smoothly
        max_g = np.max(gaps)
        if max_g > 1e10:
            return max_g  # well separated
        shifted = -gaps / beta
        shifted -= np.max(shifted)
        logsumexp = np.max(shifted) + np.log(np.sum(np.exp(shifted)))
        return -beta * logsumexp

    def min_gap_exact(poly1, poly2):
        """Exact minimum gap (negative if overlapping)."""
        min_gap = float('inf')
        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                continue
            axis /= norm
            p1 = poly1 @ axis
            p2 = poly2 @ axis
            gap = max(np.min(p1) - np.max(p2), np.min(p2) - np.max(p1))
            if gap >= 0:
                return gap  # separated
            min_gap = min(min_gap, gap)
        
        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                continue
            axis /= norm
            p1 = poly1 @ axis
            p2 = poly2 @ axis
            gap = max(np.min(p1) - np.max(p2), np.min(p2) - np.max(p1))
            if gap >= 0:
                return gap
            min_gap = min(min_gap, gap)
        
        return min_gap

    def is_valid(positions, angles, tol=1e-6):
        polys = [hex_verts(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if min_gap_exact(polys[i], polys[j]) < -tol:
                    return False
        return True

    # --- PHASE 1: Generate good initial configurations ---
    
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0
    best_score = float('inf')
    
    # Honeycomb lattice: v1 = (1.5, sqrt3/2), v2 = (0, sqrt3)
    v1 = np.array([1.5, sqrt3/2])
    v2 = np.array([0.0, sqrt3])
    
    # Generate lattice points and pick 12 closest to origin
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * v1 + j * v2
            lattice_points.append(p)
    
    lattice_points.sort(key=lambda p: np.dot(p, p))
    
    # Try multiple initial configurations
    configs = []
    
    # Config 1: 12 closest lattice points, all angle 0
    pts = np.array(lattice_points[:12])
    centroid = np.mean(pts, axis=0)
    pts -= centroid
    configs.append((pts.copy(), np.zeros(N), 0.0))
    
    # Config 2: 12 closest, all angle 30
    configs.append((pts.copy(), np.full(N, 30.0), 0.0))
    
    # Config 3: 12 closest, all angle 60
    configs.append((pts.copy(), np.full(N, 60.0), 0.0))
    
    # Config 4: Rotate the cluster
    for rot_deg in [30, 60, 90]:
        R = rot_matrix(math.radians(rot_deg))
        pts_rot = pts @ R.T
        configs.append((pts_rot.copy(), np.zeros(N), rot_deg))
        configs.append((pts_rot.copy(), np.full(N, rot_deg), rot_deg))
    
    # Config 5: Slightly compressed lattice
    for scale in [0.95, 0.98, 1.02]:
        v1s = v1 * scale
        v2s = v2 * scale
        pts_s = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                pts_s.append(i * v1s + j * v2s)
        pts_s.sort(key=lambda p: np.dot(p, p))
        pts_s = np.array(pts_s[:12])
        pts_s -= np.mean(pts_s, axis=0)
        configs.append((pts_s.copy(), np.zeros(N), 0.0))
    
    # Evaluate all configs
    for pos, ang, oa in configs:
        if not is_valid(pos, ang, tol=1e-5):
            continue
        score = outer_side_len(pos, ang, oa)
        if score < best_score:
            best_score = score
            best_positions = pos.copy()
            best_angles = ang.copy()
            best_outer_angle = oa
    
    # Fallback: if no valid config found, use the honeycomb
    if best_positions is None:
        best_positions = pts.copy()
        best_angles = np.zeros(N)
        best_outer_angle = 0.0
        best_score = outer_side_len(best_positions, best_angles, best_outer_angle)
    
    # --- PHASE 2: Optimize with smooth objective ---
    
    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        outer_angle = params[3*N]
        
        # Outer side length
        side = outer_side_len(positions, angles, outer_angle)
        
        # Smooth penalty for overlaps
        polys = [hex_verts(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = min_gap_smooth(polys[i], polys[j], beta=30.0)
                if gap < 0:
                    penalty += 100.0 * gap * gap
        return side + penalty

    def constraint_func(params, i, j):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        poly_i = hex_verts(positions[i][0], positions[i][1], angles[i])
        poly_j = hex_verts(positions[j][0], positions[j][1], angles[j])
        return min_gap_smooth(poly_i, poly_j, beta=30.0)

    x0 = np.concatenate([best_positions.flatten(), best_angles, [best_outer_angle]])
    
    bounds = []
    for i in range(N):
        bounds.append((-8, 8))
        bounds.append((-8, 8))
    for i in range(N):
        bounds.append((0, 360))
    bounds.append((0, 60))  # outer angle: by symmetry, 0-60 is sufficient
    
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: constraint_func(p, i, j)
            })
    
    # Run SLSQP optimization
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
        )
        
        if res.fun < best_score:
            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:3*N]
            oa = res.x[3*N]
            if is_valid(pos, ang, tol=1e-5):
                score = outer_side_len(pos, ang, oa)
                if score < best_score:
                    best_score = score
                    best_positions = pos.copy()
                    best_angles = ang.copy()
                    best_outer_angle = oa
    except Exception as e:
        pass
    
    # --- PHASE 3: Polish with tighter optimization ---
    if time.time() < deadline - 15:
        # Use a more precise objective for polishing
        def obj_precise(params):
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            outer_angle = params[3*N]
            
            side = outer_side_len(positions, angles, outer_angle)
            
            # Exact overlap penalty
            polys = [hex_verts(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    gap = min_gap_exact(polys[i], polys[j])
                    if gap < 0:
                        penalty += 500.0 * gap * gap
            return side + penalty

        x0 = np.concatenate([best_positions.flatten(), best_angles, [best_outer_angle]])
        
        try:
            res2 = minimize(
                obj_precise,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            )
            
            if res2.fun < best_score:
                pos = res2.x[:2*N].reshape(N, 2)
                ang = res2.x[2*N:3*N]
                oa = res2.x[3*N]
                if is_valid(pos, ang, tol=1e-6):
                    score = outer_side_len(pos, ang, oa)
                    if score < best_score:
                        best_score = score
                        best_positions = pos.copy()
                        best_angles = ang.copy()
                        best_outer_angle = oa
        except Exception:
            pass
    
    # --- PHASE 4: Try Nelder-Mead polish ---
    if time.time() < deadline - 10:
        def obj_nm(params):
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            outer_angle = params[3*N]
            
            side = outer_side_len(positions, angles, outer_angle)
            
            polys = [hex_verts(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    gap = min_gap_exact(polys[i], polys[j])
                    if gap < 0:
                        penalty += 1000.0 * gap * gap
            return side + penalty

        x0 = np.concatenate([best_positions.flatten(), best_angles, [best_outer_angle]])
        
        try:
            res3 = minimize(
                obj_nm,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True, 'disp': False}
            )
            
            pos = res3.x[:2*N].reshape(N, 2)
            ang = res3.x[2*N:3*N]
            oa = res3.x[3*N]
            if is_valid(pos, ang, tol=1e-5):
                score = outer_side_len(pos, ang, oa)
                if score < best_score:
                    best_score = score
                    best_positions = pos.copy()
                    best_angles = ang.copy()
                    best_outer_angle = oa
        except Exception:
            pass
    
    # Final computation
    final_side = outer_side_len(best_positions, best_angles, best_outer_angle)
    
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (0.0, 0.0), final_side, best_outer_angle

if __name__ == "__main__":
    construct()
