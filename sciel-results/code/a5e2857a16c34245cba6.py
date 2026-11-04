import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s limit
    
    # Centers of the 11 hexagons in honeycomb arrangement
    # 1 center
    centers = np.array([
        [0.0, 0.0]
    ])
    
    # 6 neighbors at dist sqrt(3)
    r1 = np.sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        
    # 4 more at dist 3
    # Angles 30, 90, 150, 210, 270, 330 - but only 4 in the hexagonal close packing
    # Actually in the standard honeycomb with 11 hexagons:
    # center + 6 at r1 + 4 at r2 = 11
    r2 = 3.0
    for k in range(4):
        angle = np.deg2rad(30 + k * 120)  # 30, 150, 270, 390=30? No
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
    # Wait, let me reconsider. The 11 hexagons in a hexagonal arrangement:
    # Layer 0: 1 (center)
    # Layer 1: 6 (at distance sqrt(3))
    # Layer 2: should have 12, but we only need 4 more to make 11
    # Actually, for 11 hexagons: 1 + 6 + 4 = 11
    # The 4 additional ones are at the "gaps" of layer 2, specifically at angles 30, 150, 270, 390
    # But 390 = 30, so that's only 3 unique. Let me think again.
    
    # Standard 11-hexagon cluster: center + ring of 6 + ring of 4 (not full ring of 12)
    # The 4 are at angles 30°, 150°, 270°, 30°? No.
    
    # Let me just use the original: 4 at angles 30, 90, 150, 210
    # Wait, that's asymmetric. Let me reconsider.
    
    # Actually, a better arrangement: the 4 outer hexagons should be placed to maintain
    # hexagonal symmetry as much as possible. Let me use angles 30, 150, 270, 30 - no.
    
    # Let me just keep the original setup which gave score 10.
    # 1 + 6 + 4 = 11
    
    # Recompute properly
    centers = np.array([[0.0, 0.0]])
    r1 = np.sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        
    r2 = 3.0
    # Place 4 more hexagons. In the honeycomb, the next ring would be at distance 3.
    # The angles for the 12 positions in ring 2 are at 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
    # We pick 4 that give good symmetry: 30, 150, 270, 30 is wrong.
    # Let's pick 30, 150, 270, and... we need 4. Let's do 30, 90, 150, 210, 270, 330 minus some.
    # Actually, for maximum symmetry with 4, we could do 0, 90, 180, 270 or 30, 120, 210, 300.
    # Let's try 30, 150, 270, 30 - no, 30 appears twice.
    # Let's just use 4 evenly spaced: 0, 90, 180, 270? Or match the original.
    
    # The original used 30, 90, 150, 210 which is 4 consecutive in 60-degree steps starting at 30.
    # That breaks symmetry. Let me use 0, 120, 240 (3-fold) + one more? No, we need 4.
    # Let me just use the original and focus on optimization.
    
    centers = np.array([[0.0, 0.0]])
    r1 = np.sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        
    r2 = 3.0
    for k in range(4):
        angle = np.deg2rad(30 + k * 90)  # 30, 120, 210, 300
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
    n_hex = len(centers)  # should be 11
    
    # Base vertices of a unit regular hexagon (side 1)
    # Pointy-right orientation: vertices at angles 0, 60, 120, 180, 240, 300
    base_vertices = np.array([
        [np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))] for k in range(6)
    ])
    
    def get_all_vertices(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        Returns: Array of shape (n_hex * 6, 2) containing all vertices of all inner hexagons
        """
        theta_out = params[0]
        phis = params[1:]
        
        all_vertices = []
        for i in range(n_hex):
            phi = phis[i]
            c = np.cos(phi)
            s = np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            all_vertices.append(verts)
            
        return np.vstack(all_vertices)
    
    def outer_hexagon_vertices(S, theta_out, center=np.array([0.0, 0.0])):
        """
        Returns vertices of outer hexagon with side length S, rotation theta_out, at center.
        A regular hexagon with side S has circumradius R = S.
        Vertices at angles theta_out + 60k from center.
        """
        R = S  # circumradius = side length for regular hexagon
        verts = []
        for k in range(6):
            angle = np.deg2rad(60 * k) + theta_out
            verts.append([R * np.cos(angle), R * np.sin(angle)] + center)
        return np.array(verts)
    
    def point_in_polygon(pt, poly):
        """
        Check if point pt is inside polygon poly (CCW or CW).
        Returns True if inside.
        """
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            if ((poly[i][1] > pt[1]) != (poly[j][1] > pt[1])) and \
               (pt[0] < (poly[j][0] - poly[i][0]) * (pt[1] - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0]):
                inside = not inside
            j = i
        return inside
    
    def min_distance_to_edges(pt, poly):
        """
        Returns minimum distance from point pt to any edge of polygon poly.
        """
        min_d = float('inf')
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i+1) % n]
            # Distance from pt to line segment ab
            ab = b - a
            ap = pt - a
            t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
            t = max(0, min(1, t))
            closest = a + t * ab
            d = np.linalg.norm(pt - closest)
            if d < min_d:
                min_d = d
        return min_d
    
    def hexagons_overlap(verts1, verts2):
        """
        Check if two convex polygons (hexagons) overlap using SAT.
        verts1, verts2 are arrays of shape (6, 2).
        Returns True if they overlap.
        """
        # SAT: for each edge normal of both polygons, project both polygons and check for gap
        for verts in [verts1, verts2]:
            for i in range(6):
                a = verts[i]
                b = verts[(i+1) % 6]
                edge = b - a
                normal = np.array([-edge[1], edge[0]])
                normal = normal / (np.linalg.norm(normal) + 1e-12)
                
                # Project both polygons onto this normal
                proj1 = verts1 @ normal
                proj2 = verts2 @ normal
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Check for gap
                if max1 < min2 - 1e-10 or max2 < min1 - 1e-10:
                    return False  # Found a separating axis, no overlap
        return True  # No separating axis found, they overlap
    
    def overlap_penalty(params, penalty_weight=100.0):
        """
        Compute penalty for overlapping inner hexagons.
        """
        verts_all = get_all_vertices(params)
        penalty = 0.0
        for i in range(n_hex):
            v1 = verts_all[i*6:(i+1)*6]
            for j in range(i+1, n_hex):
                v2 = verts_all[j*6:(j+1)*6]
                # Quick check: center distance
                d = np.linalg.norm(centers[i] - centers[j])
                if d < np.sqrt(3) + 0.5:  # Only check if potentially close
                    if hexagons_overlap(v1, v2):
                        penalty += penalty_weight
        return penalty
    
    def objective(params):
        """
        Minimizes the side length S of the outer hexagon.
        Outer hexagon is centered at (0,0), rotated by theta_out.
        """
        theta_out = params[0]
        
        # Get all vertices
        verts = get_all_vertices(params)
        
        # Normals of the outer hexagon
        # A regular hexagon with side S and rotation theta has sides with normals at angles theta + 30 + k*60
        # The apothem (distance from center to side) is h = S * sqrt(3) / 2
        # Condition for point P to be inside: P . n_k <= h for all k
        # So S >= (P . n_k) / (sqrt(3)/2)
        
        normals = []
        for k in range(6):
            angle = np.deg2rad(30 + k * 60) + theta_out
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)  # shape (6, 2)
        
        # Project all vertices onto all normals
        dots = verts @ normals.T
        
        # Max projection
        max_proj = np.max(dots)
        
        # Required S
        S = max_proj / (np.sqrt(3) / 2)
        
        return S
    
    def full_objective(params):
        """
        Objective with overlap penalty.
        """
        S = objective(params)
        pen = overlap_penalty(params)
        return S + pen
    
    # Initial guess
    # theta_out = 0
    # phis = [0] * 11
    x0 = np.zeros(n_hex + 1)
    
    # Bounds
    # theta_out in [0, pi/3)
    # phis in [0, pi/3)
    bounds = [(0, np.pi/3)] * (n_hex + 1)
    
    best_score = float('inf')
    best_params = None
    
    # Multi-start with Powell
    n_starts = 8
    for i in range(n_starts):
        if time.time() > deadline - 5:  # Leave 5s for final polish
            break
            
        if i == 0:
            x_start = x0.copy()
        else:
            x_start = np.random.uniform(0, np.pi/3, n_hex + 1)
            # Ensure some structure: keep theta_out small
            x_start[0] = np.random.uniform(0, np.pi/6)
            
        try:
            from scipy.optimize import minimize
            res = minimize(full_objective, x_start, method='Powell', bounds=bounds,
                          options={'maxiter': 300, 'maxfev': 5000, 'disp': False})
            
            # Check if the result is valid (no overlaps)
            final_S = objective(res.x)
            final_pen = overlap_penalty(res.x)
            
            if final_pen < 1e-6:  # No significant overlap
                if final_S < best_score:
                    best_score = final_S
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish if we have a good candidate
    if best_params is not None and time.time() < deadline - 3:
        try:
            from scipy.optimize import minimize
            # Use Nelder-Mead for polish
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
            
            final_S = objective(res.x)
            final_pen = overlap_penalty(res.x)
            
            if final_pen < 1e-6 and final_S < best_score:
                best_score = final_S
                best_params = res.x.copy()
        except Exception:
            pass
    
    # If optimization didn't find anything good, use fallback
    if best_params is None or best_score > 10.5:
        return _fallback_solution()
        
    # Extract best solution
    theta_out = best_params[0]
    phis = best_params[1:]
    
    inner = []
    for i in range(n_hex):
        inner.append([float(centers[i][0]), float(centers[i][1]), float(np.rad2deg(phis[i]))])
        
    # The side length is the objective value
    side = float(best_score)
    
    # Add tiny margin for safety
    side += 1e-9
    
    return inner, (0.0, 0.0), side, float(np.rad2deg(theta_out))

def _fallback_solution():
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 90)  # 30, 120, 210, 300
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
        
    # Compute the required side length
    # All hexagons at 30 degrees rotation
    base_vertices = np.array([
        [np.cos(np.deg2rad(60*k + 30)), np.sin(np.deg2rad(60*k + 30))] for k in range(6)
    ])
    
    all_verts = []
    for cx, cy in centers:
        all_verts.extend(base_vertices + np.array([cx, cy]))
    all_verts = np.array(all_verts)
    
    # Find max projection onto hexagon normals
    # For outer hexagon at 30 degrees, normals at 30+30+60k = 60+60k
    S_best = 0
    for theta_out_deg in [0, 30]:
        normals = []
        for k in range(6):
            angle = np.deg2rad(30 + k * 60) + np.deg2rad(theta_out_deg)
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        dots = all_verts @ normals.T
        max_proj = np.max(dots)
        S = max_proj / (np.sqrt(3) / 2)
        S_best = max(S_best, S)
    
    side = S_best + 1e-9
    return inner, (0.0, 0.0), side, 30.0

if __name__ == "__main__":
    construct()
