import math
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

# Pre-compute vertices of a unit hexagon (side 1)
# Centered at origin, one vertex at (1, 0)
HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, math.sqrt(3)/2],
    [-0.5, math.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -math.sqrt(3)/2],
    [0.5, -math.sqrt(3)/2]
])

def get_hex_verts(cx, cy, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return HEX_VERTS @ R.T + np.array([cx, cy])

def check_containment(verts, S):
    # Outer hexagon centered at (0,0), side S, orientation 0 (vertex at angle 0)
    # Normals to sides are at angles pi/6 + k*pi/3
    # Inequality: |dot(p, n_k)| <= S * APOTHEM
    max_viol = 0.0
    bound = S * APOTHEM
    for k in range(6):
        ang = math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        projs = verts[:, 0] * nx + verts[:, 1] * ny
        max_proj = np.max(np.abs(projs))
        if max_proj > bound:
            max_viol = max(max_viol, max_proj - bound)
    return max_viol

def check_overlap(verts1, verts2):
    # Check if two convex polygons (hexagons) overlap.
    # Returns the minimum distance (negative if overlapping).
    # We can use the fact that if they don't overlap, there is a separating axis.
    # The candidate axes are the normals to the edges of both polygons.
    
    def get_normals(verts):
        normals = []
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i+1)%6]
            edge = p2 - p1
            # Normal perpendicular to edge
            n = np.array([-edge[1], edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm > 0:
                normals.append(n / n_norm)
        return normals

    normals = get_normals(verts1) + get_normals(verts2)
    
    min_dist = float('inf')
    
    for n in normals:
        # Project both polygons onto this normal
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Calculate gap
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            # Overlap in this direction
            # We need the actual minimum distance, which is 0 if they intersect.
            # But for penalty, we can use the depth of overlap.
            # A better metric for penalty is the maximum penetration depth.
            # Here we just return 0 if they intersect in this projection, 
            # but we need to check if they actually intersect.
            # If they intersect in all projections, they overlap.
            # If they separate in ANY projection, they don't overlap.
            # So if we find a separating axis, return the gap.
            # If we don't find one, they overlap.
            # To get a smooth penalty, we can accumulate the negative gap.
            # However, SAT says if there is ANY separating axis, they are disjoint.
            # So if we find one, we return the gap (positive).
            # If we check all and find none, they overlap.
            # For the penalty, we want a measure of "how much" they overlap.
            # We can use the minimum gap over all axes? No, that would be negative.
            # Let's use the maximum penetration depth.
            # If they overlap, the gap is 0 in the sense of separation.
            # We can return a small negative number or use a different metric.
            # For DE, a simple penalty for "intersecting" is sufficient.
            # Let's return 0 if they intersect (overlap), and the gap if they don't.
            # Wait, if they intersect, the gap is negative?
            # Let's define the "distance" as the minimum distance between the two sets.
            # If they overlap, distance is 0.
            # If they are disjoint, distance is > 0.
            # We want to penalize distance < 0? No, distance is always >= 0.
            # We want to penalize OVERLAP.
            # So if they overlap, penalty should be large.
            # How to measure overlap?
            # We can check if any vertex of 1 is inside 2 or vice versa.
            # Or we can use the fact that if they overlap, there is no separating axis.
            # Let's assume if no separating axis is found, they overlap.
            # To get a smooth penalty, we can sum the "negative gaps" from the axes where they overlap?
            # Actually, if they overlap, the gap is "negative" in the sense that the intervals intersect.
            # The depth of intersection for axis n is min(max1, max2) - max(min1, min2).
            # This is a good measure of penetration.
            # Let's accumulate the maximum penetration depth over all axes.
            
            depth = min(max1, max2) - max(min1, min2)
            if depth > 0:
                # They overlap in this direction.
                # We want to penalize this.
                # But if they separate in another direction, they don't overlap.
                # So we must check ALL axes.
                # If ANY axis separates them, they are disjoint.
                # So we should return the gap for the separating axis.
                # If NO axis separates them, they overlap.
                # In that case, we can return the maximum penetration depth as a negative penalty?
                # Or just a fixed penalty?
                # Let's store the maximum penetration depth found so far.
                # And a flag if they are separated.
                pass
            else:
                # This axis separates them?
                # No, if depth <= 0, the intervals [min1, max1] and [min2, max2] do not intersect?
                # Wait.
                # If max1 < min2, gap = min2 - max1 > 0. Separated.
                # If max2 < min1, gap = min1 - max2 > 0. Separated.
                # If they intersect, depth = min(max1, max2) - max(min1, min2) > 0.
                
                # So if we find an axis where they are separated (gap > 0), we can return that gap immediately.
                # If we go through all axes and find no separating axis, they overlap.
                # In that case, we return a negative value (penalty).
                # Let's return -max_depth to penalize.
                pass
    
    # Re-implementing the logic clearly:
    max_penetration = 0.0
    for n in normals:
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2:
            return min2 - max1
        elif max2 < min1:
            return min1 - max2
        else:
            # Overlap in this direction
            depth = min(max1, max2) - max(min1, min2)
            if depth > max_penetration:
                max_penetration = depth
                
    # If we get here, they overlap in all directions
    return -max_penetration

def objective(params):
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return 1e10

    # Unpack
    # 11 * 2 coords + 1 theta + 1 S
    coords = params[:22].reshape(11, 2)
    theta = params[22]
    S = params[23]
    
    if S < 0:
        return 1e10
        
    penalty = 0.0
    verts_list = []
    
    for i in range(11):
        cx, cy = coords[i]
        verts = get_hex_verts(cx, cy, theta)
        verts_list.append(verts)
        
        # Containment penalty
        viol = check_containment(verts, S)
        if viol > 0:
            penalty += viol * 10.0
            
    # Overlap penalty
    for i in range(11):
        for j in range(i+1, 11):
            dist = check_overlap(verts_list[i], verts_list[j])
            if dist < 0:
                # Overlapping
                penalty += (-dist) * 10.0
                
    return S + penalty

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial guess: 4-3-4 arrangement
    # Approximate positions for 4-3-4
    # Row 0 (y=0): x = -1.5, -0.5, 0.5, 1.5
    # Row 1 (y=sqrt(3)/2): x = -1.0, 0.0, 1.0
    # Row 2 (y=sqrt(3)): x = -1.5, -0.5, 0.5, 1.5
    # This assumes aligned hexagons.
    
    init_coords = []
    # Row 0
    for x in [-1.5, -0.5, 0.5, 1.5]:
        init_coords.append([x, 0.0])
    # Row 1
    for x in [-1.0, 0.0, 1.0]:
        init_coords.append([x, math.sqrt(3)/2])
    # Row 2
    for x in [-1.5, -0.5, 0.5, 1.5]:
        init_coords.append([x, math.sqrt(3)])
        
    init_theta = 0.0
    init_S = 4.0
    
    init_params = np.array(init_coords).flatten()
    init_params = np.append(init_params, [init_theta, init_S])
    
    bounds = [(-2.0, 2.0)] * 22  # x, y for 11 hexagons
    bounds.append((0.0, math.pi/3)) # theta
    bounds.append((3.0, 6.0)) # S
    
    try:
        res = differential_evolution(objective, bounds, seed=42, maxiter=200, tol=1e-8, polish=True, workers=-1, updating='immediate')
        if res.success and res.fun < 1e9:
            best_params = res.x
        else:
            # Fallback to initial guess if DE fails
            best_params = init_params
            # Refine initial guess with local minimization
            res_local = minimize(objective, init_params, method='Nelder-Mead', options={'maxiter': 1000})
            if res_local.fun < objective(init_params):
                best_params = res_local.x
    except Exception:
        best_params = init_params
        try:
            res_local = minimize(objective, init_params, method='Nelder-Mead', options={'maxiter': 1000})
            if res_local.fun < objective(init_params):
                best_params = res_local.x
        except Exception:
            pass

    # Refine with local minimization to get precise S
    try:
        res_final = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res_final.fun < objective(best_params):
            best_params = res_final.x
    except Exception:
        pass

    coords = best_params[:22].reshape(11, 2)
    theta = best_params[22]
    S = best_params[23]
    
    # Ensure S is positive and reasonable
    if S <= 0:
        S = 4.0
        
    # Calculate actual S required for the found configuration to be valid
    # The optimizer minimizes S + penalty. If penalty is 0, S is the answer.
    # If penalty > 0, the configuration is invalid.
    # We should re-calculate the minimal S that contains this configuration.
    
    verts_list = []
    for i in range(11):
        cx, cy = coords[i]
        verts = get_hex_verts(cx, cy, theta)
        verts_list.append(verts)
    
    # Find minimal S such that all vertices are inside
    min_S = 0.0
    for verts in verts_list:
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            projs = verts[:, 0] * nx + verts[:, 1] * ny
            max_proj = np.max(np.abs(projs))
            req_S = max_proj / APOTHEM
            if req_S > min_S:
                min_S = req_S
                
    # Check overlaps again. If there are overlaps, the solution is invalid.
    # We assume the optimizer found a valid packing.
    # If not, we might need to scale up or shift, but for now we trust the optimizer.
    # To be safe, if overlaps exist, we can try to fix them by increasing S?
    # No, increasing S doesn't fix inner overlaps.
    # If inner overlaps exist, the score will be invalid.
    # Let's verify overlaps.
    valid = True
    for i in range(11):
        for j in range(i+1, 11):
            dist = check_overlap(verts_list[i], verts_list[j])
            if dist < -1e-5: # Allow small numerical error
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid configuration (the seed's best)
        # We can reconstruct the seed's best axial_26 configuration
        axial_26 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2), (3, 2),
            (1, 3), (2, 3)
        ]
        # Convert to cartesian
        pts = []
        for q, r in axial_26:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        pts = np.array(pts)
        
        # Center of mass
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        pts_centered = pts - [cx, cy]
        
        # Optimize orientation for this fixed shape
        best_S_fallback = float('inf')
        best_theta_fallback = 0.0
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            verts_list_fb = []
            for p in pts_centered:
                v = get_hex_verts(p[0], p[1], beta)
                verts_list_fb.append(v)
            min_S_fb = 0.0
            for verts in verts_list_fb:
                for k in range(6):
                    ang = math.pi / 6.0 + k * math.pi / 3.0
                    nx = math.cos(ang)
                    ny = math.sin(ang)
                    projs = verts[:, 0] * nx + verts[:, 1] * ny
                    max_proj = np.max(np.abs(projs))
                    req_S = max_proj / APOTHEM
                    if req_S > min_S_fb:
                        min_S_fb = req_S
            if min_S_fb < best_S_fallback:
                best_S_fallback = min_S_fb
                best_theta_fallback = beta
                
        coords = pts_centered
        theta = best_theta_fallback
        S = best_S_fallback
        center = [0.0, 0.0]
    else:
        center = [0.0, 0.0]
        S = min_S

    # Construct output
    inner = []
    for i in range(11):
        x, y = coords[i]
        inner.append([float(x), float(y), float(math.degrees(theta))])
        
    return inner, center, float(S), float(math.degrees(theta))

