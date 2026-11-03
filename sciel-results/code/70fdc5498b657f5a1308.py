import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0  # Safety margin for 150s limit
START_TIME = time.time()

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _distance_point_segment(p, a, b):
    """
    Calculate the minimum distance from point p to segment ab.
    Returns (distance, closest_point_on_segment).
    """
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    d = np.linalg.norm(p - proj)
    return d, proj

def _min_dist_polygons(v1, v2):
    """
    Calculate the minimum distance between two convex polygons.
    If they overlap, returns a negative value representing the depth of penetration
    approximated by the minimum translation vector magnitude.
    """
    # First check if any vertex of v1 is inside v2 or vice versa
    # For simplicity and robustness in optimization, we use a hybrid approach:
    # 1. If separated, min vertex-vertex or vertex-edge distance.
    # 2. If overlapping, we need a penalty. 
    # A robust differentiable proxy for "separation distance" that becomes negative on overlap
    # is difficult. Instead, we use the sum of squared overlaps in SAT or similar.
    
    # Let's use a simpler, robust metric:
    # Min distance between any vertex of v1 and any edge of v2, and vice versa.
    # If a vertex is inside, the distance to the boundary is the distance to the closest edge, 
    # but we need to know if it's inside.
    
    min_d = float('inf')
    
    # Check vertices of v1 against edges of v2
    for p in v1:
        for k in range(len(v2)):
            a = v2[k]
            b = v2[(k+1)%len(v2)]
            d, _ = _distance_point_segment(p, a, b)
            if d < min_d:
                min_d = d
                
    # Check vertices of v2 against edges of v1
    for p in v2:
        for k in range(len(v1)):
            a = v1[k]
            b = v1[(k+1)%len(v1)]
            d, _ = _distance_point_segment(p, a, b)
            if d < min_d:
                min_d = d
                
    # If min_d is very small, they might be touching or overlapping.
    # To penalize overlap, we can check if centers are too close or use a more complex check.
    # For a unit hexagon, if min_dist < epsilon, we assume they are touching/overlapping.
    # However, min_dist is 0 if touching. We need a penalty for being INSIDE.
    
    # Better approach for optimization:
    # Use the "signed distance" concept. 
    # If a vertex of v1 is inside v2, the distance to the boundary is positive, 
    # but we want to push it out.
    
    # Let's refine: 
    # For each vertex of v1, calculate distance to boundary of v2.
    # If vertex is inside v2, distance is negative (penalty).
    # How to check inside? For convex polygon, check cross products.
    
    def _signed_dist_to_boundary(p, poly):
        """
        Returns negative distance if p is inside poly, positive if outside.
        Distance is to the closest edge.
        """
        min_dist = float('inf')
        is_inside = True
        
        for k in range(len(poly)):
            a = poly[k]
            b = poly[(k+1)%len(poly)]
            # Check if p is on the "outside" of edge a->b
            # Cross product (b-a) x (p-a)
            cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
            if cross < -1e-9:
                is_inside = False
                break
            
            d, _ = _distance_point_segment(p, a, b)
            if d < min_dist:
                min_dist = d
                
        if is_inside:
            return -min_dist
        else:
            return min_dist

    # The separation distance between two convex polygons is the maximum of the minimum signed distances?
    # No. If they are disjoint, the distance is the min distance between boundaries.
    # If they overlap, the "distance" is negative, roughly the depth of penetration.
    
    # A common differentiable penalty for non-overlap is:
    # max(0, overlap_depth)^2
    # Overlap depth can be approximated by the minimum translation vector (MTV) magnitude.
    
    # Let's calculate MTV magnitude using SAT.
    min_overlap = float('inf')
    
    # Axes from v1
    for i in range(len(v1)):
        edge = v1[(i+1) % len(v1)] - v1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12: continue
        normal = normal / norm
        
        proj1 = np.array([np.dot(p, normal) for p in v1])
        proj2 = np.array([np.dot(p, normal) for p in v2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        overlap = min(max1, max2) - max(min1, min2)
        
        if overlap <= 0:
            return 0.0 # Disjoint or touching on this axis
        
        if overlap < min_overlap:
            min_overlap = overlap

    # Axes from v2
    for i in range(len(v2)):
        edge = v2[(i+1) % len(v2)] - v2[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12: continue
        normal = normal / norm
        
        proj1 = np.array([np.dot(p, normal) for p in v1])
        proj2 = np.array([np.dot(p, normal) for p in v2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        overlap = min(max1, max2) - max(min1, min2)
        
        if overlap <= 0:
            return 0.0
            
        if overlap < min_overlap:
            min_overlap = overlap
            
    # If we get here, they are overlapping.
    # Return a negative value to indicate violation? 
    # Or return the overlap depth as a positive penalty?
    # In our objective, we want to minimize a penalty. 
    # If they overlap, penalty = overlap_depth.
    # If they don't, penalty = 0.
    # But we also want to encourage separation if they are close?
    # Usually, just penalizing overlap is enough for packing.
    
    return min_overlap

def _objective(params):
    """
    Objective function for the optimizer.
    params: [x1, y1, x2, y2, ..., x12, y12]
    Returns: total penalty
    """
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return 1e10 # Timeout penalty
        
    centers = params.reshape(N, 2)
    angles = np.zeros(N) # Fixed rotation 0 for simplicity in this strategy
    
    penalty = 0.0
    
    # 1. Inner-Inner collisions
    for i in range(N):
        v1 = _hex_vertices(centers[i], 1.0, 0.0)
        for j in range(i + 1, N):
            v2 = _hex_vertices(centers[j], 1.0, 0.0)
            
            # Use a smooth approximation or the raw overlap depth
            # Overlap depth is not differentiable at 0, but L-BFGS-B can handle it with finite differences
            # To be safer, we can use a quadratic penalty on the overlap depth
            depth = _min_dist_polygons(v1, v2)
            if depth > 0:
                penalty += depth * depth
    
    # 2. Containment in outer hexagon
    # We will estimate the outer side length S from the current configuration
    # and penalize if S is too large? No, we want to minimize S.
    # Strategy: 
    # Let S be a parameter? No, S is determined by the extent.
    # We can define the objective as:
    # Minimize S such that all hexes fit.
    # But S is not a variable in `params`.
    # Instead, we can fix S to a target value and penalize violations?
    # Or, we can minimize the "required S" directly.
    
    # Let's compute the required S for the current centers.
    # The required S is the max projection of all vertices onto the 6 normal axes, multiplied by 2/sqrt(3).
    
    max_proj = 0.0
    for k in range(6):
        angle = np.deg2rad(k * 60) # Outer hexagon angle 0
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            verts = _hex_vertices(centers[i], 1.0, 0.0)
            for v in verts:
                proj = np.dot(v, normal)
                if proj > max_proj:
                    max_proj = proj
                    
    required_side = max_proj * 2.0 / np.sqrt(3.0)
    
    # We want to minimize required_side, but subject to no overlaps.
    # If we just minimize required_side, the hexes will collapse to the center and overlap heavily.
    # The overlap penalty will stop that.
    # So the objective is: required_side + lambda * overlap_penalty
    
    # However, required_side is not smooth (max function).
    # We can use a smooth approximation or just rely on the optimizer's finite differences.
    
    # To make it more stable, we can add a small penalty for being too close to the boundary?
    # No, we want them to touch the boundary to minimize S.
    
    # Let's use: required_side + 10 * overlap_penalty
    # The weight 10 ensures overlap is avoided before minimizing size.
    
    return required_side + 10.0 * penalty

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    np.random.seed(42)
    
    # Try different initializations
    for restart in range(10):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        if restart == 0:
            # Hexagonal lattice initial
            pts = [[0,0]]
            for k in range(6):
                angle = np.pi/3 * k
                pts.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)])
            for k in range(6):
                angle = np.pi/3 * k + np.pi/6
                pts.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle)])
                if len(pts) == 12:
                    break
            centers = np.array(pts)
        else:
            # Random positions in a disk
            r = np.sqrt(np.random.uniform(0, 16))
            theta = np.random.uniform(0, 2*np.pi)
            centers = np.array([[r*np.cos(theta), r*np.sin(theta)] for _ in range(N)])
        
        # Flatten params
        x0 = centers.flatten()
        
        # Define bounds? Centers can be anywhere, but let's bound them to a reasonable area
        # to help the optimizer. E.g., -10 to 10.
        bounds = [(-10, 10)] * (2 * N)
        
        # Run optimization
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if time.time() - start_time > TIME_LIMIT - 2:
                continue
                
            opt_centers = res.x.reshape(N, 2)
            
            # Calculate the resulting side length
            max_proj = 0.0
            for k in range(6):
                angle = np.deg2rad(k * 60)
                normal = np.array([np.cos(angle), np.sin(angle)])
                for i in range(N):
                    verts = _hex_vertices(opt_centers[i], 1.0, 0.0)
                    for v in verts:
                        proj = np.dot(v, normal)
                        if proj > max_proj:
                            max_proj = proj
                            
            current_side = max_proj * 2.0 / np.sqrt(3.0)
            
            # Check validity
            inner_hexes = [[opt_centers[i][0], opt_centers[i][1], 0.0] for i in range(N)]
            
            # Simple validity check
            valid = True
            outer_verts = _hex_vertices((0,0), current_side, 0.0)
            
            # Check overlaps
            for i in range(N):
                v1 = _hex_vertices(opt_centers[i], 1.0, 0.0)
                for j in range(i+1, N):
                    v2 = _hex_vertices(opt_centers[j], 1.0, 0.0)
                    if _min_dist_polygons(v1, v2) > 1e-6:
                        valid = False
                        break
                if not valid: break
                # Check containment
                for p in v1:
                    inside = True
                    for k in range(6):
                        a = outer_verts[k]
                        b = outer_verts[(k+1)%6]
                        cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                        if cross < -1e-6:
                            inside = False
                            break
                    if not inside:
                        valid = False
                        break
                if not valid: break
            
            if valid and current_side < best_side:
                best_side = current_side
                best_inner = inner_hexes
                best_outer_center = (0.0, 0.0)
                best_outer_angle = 0.0
                
        except Exception:
            continue
            
    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
