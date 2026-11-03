import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.spatial import ConvexHull
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def get_hexagon_vertices(cx, cy, angle_deg, side=1.0):
        angle = math.radians(angle_deg)
        verts = []
        for k in range(6):
            ang = angle + k * math.pi / 3.0
            verts.append((cx + side * math.cos(ang), cy + side * math.sin(ang)))
        return np.array(verts)

    def check_non_overlap(positions, angles_deg, margin=1e-7):
        """Check if all hexagons are non-overlapping using support functions."""
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                # For two convex polygons to be separated, there must exist a direction
                # where the support function of one minus the other is > 0.
                # Equivalently, for all directions, h_A(d) + h_B(-d) <= distance projection.
                # For regular hexagons, the critical directions are the 6 normals.
                # Actually, a simpler check for hexagons:
                # The Minkowski difference of two unit hexagons is a hexagon of side 2.
                # Its apothem is 2 * APOTHEM = sqrt(3).
                # So the centers must be separated by at least sqrt(3) in all 6 normal directions
                # of the hexagon pair. But since they can rotate, the normals are the union
                # of normals of both.
                
                # A robust check: use the support function.
                # For a regular hexagon centered at origin with angle a, the support function
                # in direction theta is APOTHEM / cos(theta - a - pi/6 + k*pi/3) for the appropriate k.
                # Actually, for a regular hexagon with vertices at angles a + k*pi/3,
                # the apothem is APOTHEM, and the support function h(d) for unit vector d
                # is APOTHEM / cos(delta) where delta is the angle between d and the nearest normal.
                # The normals are at angles a + pi/6 + k*pi/3.
                
                ang_i = math.radians(angles_deg[i])
                ang_j = math.radians(angles_deg[j])
                
                d_vec = positions[i] - positions[j]
                d_norm = np.linalg.norm(d_vec)
                if d_norm < 1e-12:
                    return False
                
                d_unit = d_vec / d_norm
                theta = math.atan2(d_unit[1], d_unit[0])
                
                # Support function of hex i in direction d_unit:
                # The normals of hex i are at angles ang_i + pi/6 + k*pi/3 for k=0..5.
                # The support function is APOTHEM / cos(theta - normal_angle) for the normal
                # closest to theta. But actually, h(d) = max over vertices v of d . v.
                # For a regular hexagon, this equals APOTHEM / cos(phi) where phi is the
                # angle between d and the nearest normal, but only if |phi| <= pi/6.
                
                def support_hex(theta, angle):
                    # Find the normal closest to theta
                    # Normals are at angle + pi/6 + k*pi/3
                    delta = (theta - angle - math.pi/6) % (math.pi/3)
                    if delta > math.pi/6:
                        delta = delta - math.pi/3
                    return APOTHEM / math.cos(delta)
                
                # The condition for non-overlap is:
                # h_i(d_unit) + h_j(-d_unit) <= d_norm
                # -d_unit is at angle theta + pi
                h_i = support_hex(theta, ang_i)
                h_j = support_hex(theta + math.pi, ang_j)
                
                if h_i + h_j > d_norm + margin:
                    return False
        
        return True

    def calc_S(positions, angles_deg, center, outer_angle_deg):
        """Calculate the minimum outer side length containing all hexagons."""
        n = len(positions)
        outer_angle = math.radians(outer_angle_deg)
        
        # For each of the 6 directions of the outer hexagon,
        # compute the maximum extent of all inner hexagons.
        # The outer hexagon must contain all vertices of all inner hexagons.
        # For a direction d, the constraint is: for all inner vertices v,
        # (v - center) . d <= S * APOTHEM and (center - v) . d <= S * APOTHEM
        # Actually, the outer hexagon has normals at outer_angle + pi/6 + k*pi/3.
        # The constraint is that for each normal n_k, for all inner vertices v:
        # (v - center) . n_k <= S * APOTHEM
        
        # The maximum of (v - center) . n_k over all vertices v of all inner hexagons
        # equals max over inner hexagons of [ (pos_i - center) . n_k + h_i(n_k) ]
        # where h_i(n_k) is the support function of inner hex i in direction n_k.
        
        S_min = 0.0
        for k in range(6):
            ang_n = outer_angle + math.pi/6 + k * math.pi/3
            n_k = np.array([math.cos(ang_n), math.sin(ang_n)])
            
            max_proj = 0.0
            for i in range(n):
                ang_i = math.radians(angles_deg[i])
                rel = positions[i] - center
                base = rel @ n_k
                h_i = support_hex(ang_n, ang_i)
                max_proj = max(max_proj, base + h_i)
            
            S_candidate = max_proj / APOTHEM
            S_min = max(S_min, S_candidate)
        
        return max(S_min, 1.0)  # Side length must be at least 1

    def support_hex(theta, angle):
        """Support function of a unit regular hexagon with rotation angle in direction theta."""
        delta = (theta - angle - math.pi/6) % (math.pi/3)
        if delta > math.pi/6:
            delta = delta - math.pi/3
        return APOTHEM / math.cos(delta)

    # Candidate initial configurations
    candidates = []
    
    # Configuration 1: Compact hexagonal cluster
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_1)
    
    # Configuration 2: Linear arrangement
    axial_2 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)
    ]
    candidates.append(axial_2)
    
    # Configuration 3: 3x3 block
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (3, 1)
    ]
    candidates.append(axial_3)
    
    # Configuration 4: Diamond
    axial_4 = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1),
        (0, 2), (1, 2), (-1, 2),
        (0, -1)
    ]
    candidates.append(axial_4)
    
    # Configuration 5: T-shape
    axial_5 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3),
        (0, -1), (1, -1), (2, -1)
    ]
    candidates.append(axial_5)
    
    # Configuration 6: Cross
    axial_6 = [
        (1, 1),
        (0, 1), (2, 1),
        (1, 0), (1, 2),
        (0, 0), (2, 0),
        (0, 2), (2, 2),
        (0, -1), (2, -1)
    ]
    candidates.append(axial_6)
    
    # Configuration 7: L-shape
    axial_7 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_7)
    
    # Configuration 8: 2 rows
    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)
    ]
    candidates.append(axial_8)
    
    # Configuration 9: Clustered
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, 1), (1, 2)
    ]
    candidates.append(axial_9)
    
    # Configuration 10: Staggered
    axial_10 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    best_S = float('inf')
    best_positions = None
    best_angles = None
    best_center = None
    best_outer_angle = 0.0

    def optimize_configuration(init_pts, init_angles=None):
        """Optimize a configuration using sequential convex optimization."""
        nonlocal best_S, best_positions, best_angles, best_center, best_outer_angle
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return
        
        n = len(init_pts)
        
        if init_angles is None:
            angles = np.zeros(n)
        else:
            angles = init_angles.copy()
        
        positions = init_pts.copy()
        
        # Try different outer angles
        for outer_angle_deg in range(0, 30, 3):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return
            
            # Sequential optimization: fix angles, optimize positions
            # The problem is: minimize S subject to non-overlap and containment.
            # For fixed angles, the non-overlap constraints are convex in positions,
            # and the containment constraints are also convex.
            
            # We'll use a penalty method with Nelder-Mead for joint optimization.
            def objective(params):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    return 1e10
                
                pos = params[:2*n].reshape(n, 2)
                angs = params[2*n:3*n]
                cx = params[3*n]
                cy = params[3*n+1]
                oa = params[3*n+2]
                
                # Check non-overlap
                if not check_non_overlap(pos, angs):
                    # Return penalty
                    penalty = 0.0
                    for i in range(n):
                        for j in range(i+1, n):
                            d_vec = pos[i] - pos[j]
                            d_norm = np.linalg.norm(d_vec)
                            if d_norm < 1e-12:
                                penalty += 100.0
                                continue
                            ang_i = math.radians(angs[i])
                            ang_j = math.radians(angs[j])
                            theta = math.atan2(d_vec[1], d_vec[0])
                            h_i = support_hex(theta, ang_i)
                            h_j = support_hex(theta + math.pi, ang_j)
                            if h_i + h_j > d_norm:
                                penalty += 10.0 * (h_i + h_j - d_norm)
                    return 1e6 + penalty
                
                S = calc_S(pos, angs, (cx, cy), oa)
                return S
            
            # Initial guess
            init_params = np.concatenate([
                positions.ravel(),
                angles,
                [0.0, 0.0, float(outer_angle_deg)]
            ])
            
            try:
                res = minimize(objective, init_params, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if res.fun < best_S:
                    new_pos = res.x[:2*n].reshape(n, 2)
                    new_angs = res.x[2*n:3*n]
                    new_cx = res.x[3*n]
                    new_cy = res.x[3*n+1]
                    new_oa = res.x[3*n+2]
                    
                    # Verify
                    if check_non_overlap(new_pos, new_angs):
                        S_exact = calc_S(new_pos, new_angs, (new_cx, new_cy), new_oa)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_positions = new_pos
                            best_angles = new_angs
                            best_center = (new_cx, new_cy)
                            best_outer_angle = new_oa
            except Exception:
                pass
        
        # Try to improve angles for the best found positions
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return
        
        if best_positions is not None and best_positions.shape[0] == n:
            # Local search on angles only
            def angle_objective(angs):
                if not check_non_overlap(best_positions, angs):
                    return 1e6
                return calc_S(best_positions, angs, best_center, best_outer_angle)
            
            try:
                res = minimize(angle_objective, angles, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if res.fun < best_S:
                    if check_non_overlap(best_positions, res.x):
                        S_exact = calc_S(best_positions, res.x, best_center, best_outer_angle)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_angles = res.x
            except Exception:
                pass

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = axial_to_cartesian(axial)
        # Center the configuration
        center_pt = pts.mean(axis=0)
        pts_centered = pts - center_pt
        
        optimize_configuration(pts_centered)

    # If no valid solution found, use a simple fallback
    if best_positions is None or not np.isfinite(best_S):
        # Use the first candidate with a simple estimate
        pts = axial_to_cartesian(candidates[0])
        center_pt = pts.mean(axis=0)
        best_positions = pts
        best_angles = np.zeros(N)
        best_center = (0.0, 0.0)
        best_outer_angle = 0.0
        
        # Calculate S
        S = 0.0
        for k in range(6):
            ang_n = math.pi/6 + k * math.pi/3
            n_k = np.array([math.cos(ang_n), math.sin(ang_n)])
            max_proj = 0.0
            for i in range(N):
                rel = best_positions[i]
                base = rel @ n_k
                h_i = support_hex(ang_n, 0.0)
                max_proj = max(max_proj, base + h_i)
            S = max(S, max_proj / APOTHEM)
        best_S = max(S, 1.0)

    # Prepare output
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    center = [float(best_center[0]), float(best_center[1])]
    side = float(best_S)
    rotation = float(best_outer_angle)
    
    return inner, center, side, rotation

if __name__ == "__main__":
    res = construct()
    print(res)
