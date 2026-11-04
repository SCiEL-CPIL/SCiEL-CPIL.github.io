import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def _triangle_areas(points):
    """
    Compute areas of all triangles formed by the points.
    """
    n = len(points)
    tri_indices = list(combinations(range(n), 3))
    areas = np.empty(len(tri_indices))
    for i, (i1, i2, i3) in enumerate(tri_indices):
        p1 = points[i1]
        p2 = points[i2]
        p3 = points[i3]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        areas[i] = 0.5 * abs(cross)
    return areas

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Compute the raw score: min triangle area / convex hull area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _regular_polygon():
    """
    Generate a regular 13-gon centered at origin.
    """
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    # Initial best is regular polygon
    best_points = _regular_polygon()
    best_score = _score(best_points)
    
    # Strategy: Direct Coordinate Optimization with SLSQP
    # We fix P0 at (0,0) and P1 at (1,0) to remove translation and rotation symmetry.
    # The remaining N-2 points have 2*(N-2) free coordinates.
    # We optimize these coordinates to maximize the minimum triangle area,
    # subject to the constraint that the points form a convex polygon in the given order.
    
    n_free = N - 2
    dim = 2 * n_free
    
    # Initial guess: Regular 13-gon, transformed to fix P0=(0,0), P1=(1,0)
    # Regular 13-gon vertices:
    # v_i = (cos(2pi*i/13), sin(2pi*i/13))
    # We want to map v_0 to (0,0) and v_1 to (1,0).
    # v_0 is at angle 0. v_1 is at angle 2pi/13.
    # Translation: subtract v_0.
    # Rotation: rotate by -angle(v_1 - v_0).
    # Scaling: divide by |v_1 - v_0|.
    
    pts_init = _regular_polygon()
    p0 = pts_init[0]
    p1 = pts_init[1]
    
    # Translate
    pts_trans = pts_init - p0
    # Now p0 is (0,0)
    
    # Rotate
    v1 = pts_trans[1]
    angle_v1 = np.arctan2(v1[1], v1[0])
    cos_a = np.cos(-angle_v1)
    sin_a = np.sin(-angle_v1)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts_rot = pts_trans @ rot_matrix.T
    
    # Scale
    scale = np.linalg.norm(pts_rot[1])
    pts_scaled = pts_rot / scale
    
    # Extract free coordinates (indices 2 to N-1)
    x0 = pts_scaled[2:, 0]
    y0 = pts_scaled[2:, 1]
    x0_flat = np.concatenate([x0, y0])
    
    def unpack(x):
        """Convert flat array to (N, 2) points with P0=(0,0), P1=(1,0)"""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(n_free):
            pts[i+2, 0] = x[2*i]
            pts[i+2, 1] = x[2*i+1]
        return pts

    def objective_and_constraints(x):
        """
        Returns (objective_value, constraints_list)
        Objective: -min_triangle_area (we want to maximize min area, so minimize negative)
        Constraints:
        1. Convexity: Cross products of consecutive edges > 0
        2. Min Area >= t (handled by adding t as a variable? No, SLSQP handles constraints directly)
        
        Actually, to maximize the minimum area, we can introduce a variable t.
        Let x_aug = [x_coords..., t].
        Objective: -t
        Constraints:
        - Area(i,j,k) - t >= 0 for all i,j,k
        - Convexity constraints >= 0
        """
        # This approach requires augmenting the variable space.
        # Let's stick to a simpler surrogate for SLSQP if augmenting is complex.
        # Or, we can use the fact that SLSQP can handle 'ineq' constraints.
        
        pts = unpack(x)
        
        # Check convexity of the polygon formed by pts in order 0..N-1
        # Note: The order 0..N-1 must correspond to the angular order.
        # If the points get "twisted", the convexity constraint will fail (cross product < 0).
        
        cross_prods = []
        for i in range(N):
            p_curr = pts[i]
            p_next = pts[(i+1) % N]
            p_next2 = pts[(i+2) % N]
            v1 = p_next - p_curr
            v2 = p_next2 - p_next
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            cross_prods.append(cross)
        
        # Convexity constraint: all cross products must be positive
        # We add a small epsilon to ensure strict convexity
        convexity_violations = np.array(cross_prods) - 1e-6
        
        # Calculate all triangle areas
        areas = _triangle_areas(pts)
        min_area = np.min(areas)
        
        # We want to maximize min_area.
        # So objective is -min_area.
        obj = -min_area
        
        # Constraints for SLSQP:
        # 1. Convexity: cross_prods >= 0
        # 2. We don't have a direct "min_area >= t" constraint unless we augment variables.
        # Instead, we can just maximize min_area directly. The gradient of min_area is non-smooth.
        # SLSQP might struggle with the non-differentiability of min().
        
        # Alternative: Use a smooth approximation of the min area in the objective?
        # If we use smooth approx in objective, we don't need area constraints.
        # But we still need convexity constraints.
        
        # Let's use a smooth min for the objective to help SLSQP.
        beta = 100.0
        shifted = -beta * areas
        max_shifted = np.max(shifted)
        sum_exp = np.sum(np.exp(shifted - max_shifted))
        log_sum = np.log(sum_exp) + max_shifted
        soft_min = -1.0 / beta * log_sum
        
        obj_smooth = -soft_min
        
        # Constraints: Convexity
        # SLSQP expects constraints in form: fun(x) >= 0
        # We return a list of dicts.
        
        # Note: SLSQP can be slow with 22 variables and many constraints if we add all 286 area constraints.
        # So we rely on the smooth objective for the area part, and only use constraints for convexity.
        
        cons = []
        # Add convexity constraints
        # To reduce the number of constraints, we can just check the worst one?
        # No, SLSQP handles multiple. 13 constraints is fine.
        for i in range(N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: (lambda p: (p[(i+1)%N] - p[i]) @ np.array([-(p[(i+2)%N][1] - p[(i+1)%N][1]), p[(i+2)%N][0] - p[(i+1)%N][0]]))(unpack(x))
            })
            
        return obj_smooth, cons

    # SLSQP might be slow or get stuck.
    # Let's try a simpler approach first: L-BFGS-B with penalty for non-convexity.
    # This is faster and often robust enough.
    
    def penalized_obj(x):
        pts = unpack(x)
        
        # Convexity check
        min_cross = np.inf
        for i in range(N):
            p_curr = pts[i]
            p_next = pts[(i+1) % N]
            p_next2 = pts[(i+2) % N]
            v1 = p_next - p_curr
            v2 = p_next2 - p_next
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            if cross < min_cross:
                min_cross = cross
        
        # Penalty for non-convexity
        if min_cross < 0:
            penalty = 100.0 * min_cross**2
        else:
            penalty = 0.0
            
        # Smooth min area
        areas = _triangle_areas(pts)
        beta = 50.0
        shifted = -beta * areas
        max_shifted = np.max(shifted)
        sum_exp = np.sum(np.exp(shifted - max_shifted))
        log_sum = np.log(sum_exp) + max_shifted
        soft_min = -1.0 / beta * log_sum
        
        return -soft_min + penalty

    # Bounds for free coordinates
    # Points should be roughly within [-2, 2] to avoid numerical issues and ensure hull area is not degenerate
    bounds = [(-2.0, 2.0)] * dim
    
    # Multi-start L-BFGS-B
    inits = [x0_flat]
    # Add some random perturbations
    np.random.seed(42)
    for _ in range(5):
        noise = np.random.normal(0, 0.1, dim)
        inits.append(x0_flat + noise)
        
    for init in inits:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                penalized_obj,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            
            if res.success:
                candidate_points = unpack(res.x)
                
                # Verify convexity
                is_convex = True
                for i in range(N):
                    p_curr = candidate_points[i]
                    p_next = candidate_points[(i+1) % N]
                    p_next2 = candidate_points[(i+2) % N]
                    v1 = p_next - p_curr
                    v2 = p_next2 - p_next
                    cross = v1[0]*v2[1] - v1[1]*v2[0]
                    if cross < 1e-8:
                        is_convex = False
                        break
                        
                if is_convex:
                    score = _score(candidate_points)
                    if score > best_score:
                        best_score = score
                        best_points = candidate_points
                        # Update init for next iteration to be the current best
                        new_x0 = candidate_points[2:, 0]
                        new_y0 = candidate_points[2:, 1]
                        inits[0] = np.concatenate([new_x0, new_y0])
                        
        except Exception:
            pass

    # If we didn't find a better point, return the regular polygon
    return best_points
