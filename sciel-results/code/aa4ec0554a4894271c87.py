import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

def get_score(points):
    N = len(points)
    if N < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area calculation
    # Generate all indices i < j < k
    # 13 points -> 286 triangles
    # We can vectorize this by creating arrays of indices
    i, j, k = np.indices((N, N, N))
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
    # cross = (p2 - p1) x (p3 - p1)
    # cross_z = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def soft_min(areas, tau):
    """
    Smooth approximation of the minimum.
    Uses log-sum-exp trick for stability.
    min(x) ~ -1/tau * log(sum(exp(-tau * x)))
    """
    # For numerical stability, we can shift by min(areas)
    # But since we want the gradient to be smooth and point towards the minimum,
    # the standard soft-min is sufficient if tau is not too large.
    # To avoid overflow in exp(-tau * x) if x is large, we can normalize areas first.
    # However, areas are small (normalized by hull area, so < 1).
    # Let's use the log-sum-exp formulation carefully.
    # -1/tau * log(sum(exp(-tau * areas)))
    
    # If tau is large, exp(-tau * areas) can underflow for larger areas, which is fine.
    # It can overflow if areas are negative, but areas are non-negative.
    # So exp(-tau * areas) <= 1. No overflow.
    
    # To improve gradient flow, we can just compute it directly.
    weighted_sum = np.sum(np.exp(-tau * areas))
    if weighted_sum == 0:
        return 0.0
    return -1.0 / tau * np.log(weighted_sum)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 13
    best_score = 0.0
    best_points = None
    
    # Strategy:
    # 1. Define a parameterization that enforces a "wheel" topology:
    #    - 1 point at center (or near center)
    #    - 12 points on the convex hull, ordered by angle.
    #    - Parameters: 
    #      - r_center: distance of center from origin (we fix center at (0,0) for simplicity? 
    #        No, allowing it to move might help. But for a "wheel", the center is usually the hub.
    #        Let's fix the center at (0,0) to reduce dimensionality and enforce symmetry breaking via outer points?
    #        Actually, the optimal solution for max min area often has the center slightly off-center 
    #        or the outer points non-uniform.
    #        Let's parameterize the 12 outer points by their angles and radii.
    #        And the center point by its x, y coordinates.
    #        Total params: 12 (angles) + 12 (radii) + 2 (center) = 26. Same dimensionality.
    #        But the topology is fixed: the center is connected to all 12 outer points in the Delaunay triangulation
    #        if the center is strictly inside the convex hull of the outer points and the outer points are in convex position.
    
    # Let's try a simpler parameterization first:
    # 12 points on a circle of radius R, with slightly perturbed angles and radii?
    # No, let's stick to the "wheel" idea:
    # Point 12: Center (cx, cy)
    # Points 0-11: Outer ring.
    #   Angle_i = base_angle + perturbation_i
    #   Radius_i = base_radius * (1 + perturbation_i)
    
    # To allow the "wheel" to deform, we optimize:
    # x: [cx, cy, r_0, ..., r_11, a_0, ..., a_11]
    # where r_i are radii and a_i are angles.
    # We enforce that the outer points are sorted by angle.
    
    def points_from_params(params):
        cx, cy = params[0], params[1]
        radii = params[2:14]
        angles = params[14:26]
        
        # Ensure angles are sorted? 
        # If we just map them, the "outer" points might cross if angles get too far apart.
        # To maintain the "wheel" topology (center connected to all, outer points forming a convex polygon),
        # we should probably keep the angles roughly uniform.
        # Let's define angles as deviations from uniform spacing.
        # base_angles = np.linspace(0, 2*pi, 12, endpoint=False)
        # actual_angles = base_angles + angles_dev
        # Then sort the points by actual_angles to ensure they form a convex polygon in order.
        
        base_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        actual_angles = base_angles + angles
        
        # Sort by angle to ensure convex order
        sort_idx = np.argsort(actual_angles)
        sorted_angles = actual_angles[sort_idx]
        sorted_radii = radii[sort_idx]
        
        outer_pts = np.column_stack((sorted_radii * np.cos(sorted_angles), 
                                      sorted_radii * np.sin(sorted_angles)))
        
        center_pt = np.array([[cx, cy]])
        
        # The order of points in the array doesn't matter for get_score, 
        # but it matters for the topology if we were using it. 
        # get_score is permutation invariant.
        # So we just return the 13 points.
        return np.vstack([outer_pts, center_pt])

    def objective(params):
        pts = points_from_params(params)
        
        # Check if center is inside convex hull of outer points?
        # If not, the topology changes and the "wheel" assumption breaks.
        # But get_score is still valid.
        # However, if the center goes outside, the minimum area might drop.
        # We can add a penalty if the center is outside the hull of the outer points.
        
        # For now, just compute score.
        # We want to maximize min area.
        # So we minimize -soft_min(areas, tau)
        
        # Compute areas
        N = len(pts)
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        i, j, k = i[mask], j[mask], k[mask]
        
        p1, p2, p3 = pts[i], pts[j], pts[k]
        dx1 = p2[:, 0] - p1[:, 0]
        dy1 = p2[:, 1] - p1[:, 1]
        dx2 = p3[:, 0] - p1[:, 0]
        dy2 = p3[:, 1] - p1[:, 1]
        
        cross = dx1 * dy2 - dy1 * dx2
        areas = 0.5 * np.abs(cross)
        
        # Normalize by hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1e6 # Penalty
        except Exception:
            return 1e6 # Penalty
            
        norm_areas = areas / hull_area
        
        tau = 50.0 # Temperature for soft-min
        # Soft min of norm_areas
        # We want to maximize the min, so minimize the soft-min.
        # soft_min = -1/tau * log(sum(exp(-tau * x)))
        # This is a lower bound on the min.
        # As tau -> inf, soft_min -> min.
        
        # For optimization, we want a smooth function.
        # Let's use the log-sum-exp formulation directly.
        # To avoid underflow, we can shift.
        # min(x) approx -1/tau * log(sum(exp(-tau * (x - min(x)))) - ... ) 
        # Actually, just computing it is fine for small arrays.
        
        # Note: np.exp(-tau * x) where x is in [0, 1] and tau=50.
        # exp(-50) is ~ 1e-22. This is fine.
        
        weighted_sum = np.sum(np.exp(-tau * norm_areas))
        if weighted_sum == 0:
            return 1e6
        soft_min_val = -1.0 / tau * np.log(weighted_sum)
        
        return soft_min_val

    # Initial guess
    # Regular 12-gon + center
    cx, cy = 0.0, 0.0
    radii = np.ones(12)
    angles_dev = np.zeros(12)
    
    params_init = np.concatenate([[cx, cy], radii, angles_dev])
    
    # Optimize
    # We can try multiple restarts with different tau or initial perturbations.
    
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        
        if n_restarts == 1:
            x_start = params_init
        else:
            # Perturb
            x_start = params_init + np.random.normal(0, 0.1, 26)
            # Keep radii positive
            x_start[2:14] = np.abs(x_start[2:14]) + 0.5
            
        try:
            # Use L-BFGS-B for better handling of the smooth objective
            # Bounds:
            # cx, cy: [-2, 2]
            # radii: [0.1, 2.0]
            # angles_dev: [-pi/12, pi/12] (so they don't cross neighbors too much)
            bounds = [(-2, 2), (-2, 2)] + [(0.1, 2.0)] * 12 + [(-np.pi/12, np.pi/12)] * 12
            
            res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e5:
                pts = points_from_params(res.x)
                score = get_score(pts)
                
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
                    
        except Exception as e:
            pass
        
        if time.time() > deadline - 10:
            break
            
    if best_points is None:
        # Fallback
        angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        points_init = np.column_stack((np.cos(angles_12), np.sin(angles_12)))
        center = np.array([0.0, 0.0])
        best_points = np.vstack([points_init, center])
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(get_score(pts))
