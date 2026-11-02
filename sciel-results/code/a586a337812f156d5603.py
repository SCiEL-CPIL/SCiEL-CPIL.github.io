import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_vertices(angle_deg):
    """Return vertices of a unit regular hexagon rotated by angle_deg."""
    base_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    rot = np.deg2rad(angle_deg)
    angles = base_angles + rot
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _minkowski_diff_poly(angle1_deg, angle2_deg):
    """
    Compute the Minkowski difference H(a1) - H(a2).
    This is the set of center differences d such that H1 and H2 intersect.
    We return the vertices of this convex polygon.
    """
    # H1 - H2 = H1 + (-H2)
    # -H2 is H2 rotated by 180.
    # Since H is centrally symmetric, -H2 is just H2 reflected?
    # Actually, H(a) is centrally symmetric. So -H(a2) is the same set as H(a2).
    # Wait, H is the set of points. -H is the set of -p for p in H.
    # Since H is symmetric about origin, -H = H.
    # So H1 - H2 = H1 + H2? No.
    # H1 - H2 = { x - y | x in H1, y in H2 }.
    # Since H2 is symmetric, -y is in H2. So x - y = x + (-y).
    # So H1 - H2 = H1 + H2.
    # The Minkowski sum of two regular hexagons.
    
    v1 = _hex_vertices(angle1_deg)
    v2 = _hex_vertices(angle2_deg)
    
    # The Minkowski sum of two convex polygons is a convex polygon.
    # We can compute it by merging edge vectors sorted by angle.
    # But for simplicity, we can just check if a point is inside the polygon defined by the sum.
    # Actually, we can just compute the support function of the sum.
    # h_sum(u) = h_H1(u) + h_H2(u).
    # The point d is inside if max_u (u dot d) <= h_sum(u) for all u?
    # No, d is inside if d <= h_sum(u) for all u (in the sense of support).
    # Specifically, d is in H1+H2 iff for all u, u dot d <= h_H1(u) + h_H2(u).
    
    # We will return a function that checks if a vector d is inside the forbidden zone.
    # To make it fast, we precompute the normals and max support values for a grid of angles?
    # Or just use the 6 normals of the "aligned" case? No, the shape changes.
    
    # For the objective, we need a smooth penalty.
    # We can approximate the boundary by checking the 6 normals of the average orientation?
    # Better: Use the 12 normals (6 from each) to define the polygon.
    
    # Let's just return the vertices of the Minkowski Sum.
    # Algorithm for Minkowski Sum of two convex polygons:
    # 1. Sort edges of P and Q by angle.
    # 2. Walk around the perimeter.
    
    # For this specific task, we can hardcode the check for the few relative angles we care about.
    # Relative angles: 0, 30, 60.
    
    pass

def _hex_normals():
    """Return the 6 outward normals of a unit regular hexagon (pointy-top)."""
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _support_function(points, normals):
    """Return the support function h(u) = max_i u·p_i for each normal u."""
    return np.max(points @ normals.T, axis=0)

def _is_inside_minkowski_sum(d, angle1, angle2):
    """
    Check if vector d is inside H(angle1) + H(angle2).
    Uses the support function definition.
    d is inside if max_u (u dot d) <= h_H1(u) + h_H2(u) for all u?
    Actually, d is inside the convex set K iff u dot d <= h_K(u) for all u.
    Here K = H1 + H2. h_K(u) = h_H1(u) + h_H2(u).
    """
    # We need to check this for all u.
    # The maximum violation is max_u (u dot d - h_H1(u) - h_H2(u)).
    # If this max is <= 0, d is inside.
    # We need d to be OUTSIDE. So we want max_u (h_H1(u) + h_H2(u) - u dot d) >= 0?
    # No, if d is outside, there exists u such that u dot d > h_K(u).
    # So the "distance" to the boundary is min_u (h_K(u) - u dot d) ?
    # If d is inside, h_K(u) >= u dot d for all u.
    # If d is outside, there is some u where u dot d > h_K(u).
    
    # To get a smooth penalty, we can use the max over a set of normals.
    # The shape of H1+H2 is a dodecagon if angles differ by 30.
    # We should use the 12 normals of the dodecagon.
    
    # Generate normals for H1 and H2.
    n1 = _hex_normals() # These are for 0 deg rotation?
    # _hex_normals returns normals for a hexagon with vertices at 0, 60...
    # If we rotate the hexagon, the normals rotate too.
    
    # Let's compute normals for specific angles.
    def get_normals(angle_deg):
        base = np.deg2rad([30, 90, 150, 210, 270, 330])
        rot = np.deg2rad(angle_deg)
        angles = base + rot
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    n1 = get_normals(angle1)
    n2 = get_normals(angle2)
    
    # The relevant normals for the Minkowski sum are the union of normals from both polygons?
    # Actually, the edges of the sum are from the edges of the components.
    # So the normals of the sum are the union of the normals of the components.
    all_normals = np.vstack((n1, n2))
    
    # Compute h_H1(u) and h_H2(u) for these normals.
    v1 = _hex_vertices(angle1)
    v2 = _hex_vertices(angle2)
    
    h1 = np.max(v1 @ all_normals.T, axis=1)
    h2 = np.max(v2 @ all_normals.T, axis=1)
    h_sum = h1 + h2
    
    # Project d onto these normals
    proj = d @ all_normals.T
    
    # Violation: We want d to be outside.
    # If d is inside, proj <= h_sum for all normals.
    # The "depth" of penetration is max(0, h_sum - proj).
    # We want to minimize this depth? No, we want it to be 0.
    # If d is outside, there exists a normal where proj > h_sum.
    # So the minimum over normals of (proj - h_sum) should be > 0?
    # No. If d is far outside, proj - h_sum is large for some u.
    # If d is inside, proj - h_sum is negative for all u.
    # So the "distance" to the boundary is min_u (h_sum(u) - proj(u))?
    # If d is inside, this min is positive.
    # If d is outside, this min is negative (for the normal facing d).
    
    # We want a penalty if d is inside.
    # Penalty = max(0, min_u (h_sum(u) - proj(u)))?
    # If d is inside, min_u (...) is positive.
    # If d is outside, min_u (...) is negative.
    # So Penalty = max(0, min_u (h_sum - proj)).
    
    diff = h_sum - proj
    min_diff = np.min(diff)
    if min_diff > 0:
        return min_diff
    return 0.0

def _min_separation(centers, angles_deg, normals):
    """Return the minimum separation violation across all pairs."""
    min_viol = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            diff = centers[i] - centers[j]
            viol = _is_inside_minkowski_sum(diff, angles_deg[i], angles_deg[j])
            if viol > min_viol:
                min_viol = viol
    return min_viol

def construct():
    normals = _hex_normals()
    sqrt3 = np.sqrt(3.0)
    
    best_L = float('inf')
    best_inner = None
    best_params = None
    
    # Radial Slicing: 1 center, 5 inner ring, 6 outer ring
    # Parameters: [r_inner, r_outer, rot_inner, rot_outer]
    # Inner ring angles: 0, 72, 144, 216, 288 + rot_inner
    # Outer ring angles: 0, 60, 120, 180, 240, 300 + rot_outer
    
    def build_centers(params):
        r_in, r_out, rot_in, rot_out = params
        rot_in_rad = np.deg2rad(rot_in)
        rot_out_rad = np.deg2rad(rot_out)
        centers = np.zeros((N, 2))
        centers[0] = [0.0, 0.0]
        
        inner_angles = np.deg2rad([0, 72, 144, 216, 288]) + rot_in_rad
        for k in range(5):
            centers[1 + k] = [r_in * np.cos(inner_angles[k]), r_in * np.sin(inner_angles[k])]
            
        outer_angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + rot_out_rad
        for k in range(6):
            centers[6 + k] = [r_out * np.cos(outer_angles[k]), r_out * np.sin(outer_angles[k])]
        return centers

    def get_angles(params):
        r_in, r_out, rot_in, rot_out = params
        angles = np.zeros(N)
        angles[0] = 0.0
        inner_angles = np.deg2rad([0, 72, 144, 216, 288]) + np.deg2rad(rot_in)
        angles[1:6] = np.rad2deg(inner_angles)
        outer_angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + np.deg2rad(rot_out)
        angles[6:12] = np.rad2deg(outer_angles)
        return angles

    def objective(params):
        centers = build_centers(params)
        angles = get_angles(params)
        
        max_abs_dot = np.max(np.abs(centers @ normals.T))
        L_req = 2.0 * max_abs_dot / sqrt3 + 1.0
        
        viol = _min_separation(centers, angles, normals)
        
        return L_req + 1000.0 * viol * viol
    
    # Try multiple random starts
    for seed in range(50):
        np.random.seed(seed)
        x0 = np.array([
            np.random.uniform(1.5, 3.5),   # r_inner
            np.random.uniform(3.0, 6.0),   # r_outer
            np.random.uniform(0, 360),     # rot_inner
            np.random.uniform(0, 360)      # rot_outer
        ])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            params = result.x
            centers = build_centers(params)
            angles = get_angles(params)
            
            viol = _min_separation(centers, angles, normals)
            if viol > 1e-5:
                continue
            
            max_abs_dot = np.max(np.abs(centers @ normals.T))
            L_final = 2.0 * max_abs_dot / sqrt3 + 1.0
            
            if L_final < best_L:
                best_L = L_final
                best_params = params.copy()
                best_inner = [[centers[i, 0], centers[i, 1], angles[i]] for i in range(N)]
        except Exception:
            continue
    
    if best_inner is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, (0.0, 0.0), best_L, 0.0
