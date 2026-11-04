"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _generate_two_ring_points(r, theta):
    """
    Generate 14 points: 7 on unit circle (outer), 7 on circle of radius r (inner).
    Outer angles: 0, 2pi/7, ...
    Inner angles: theta, theta + 2pi/7, ...
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _min_triangle_area_fast(pts):
    """
    Fast calculation of minimum triangle area for a specific configuration.
    Assumes points are ordered or we can just brute force efficiently since N=14 is small.
    """
    n = len(pts)
    min_area = float('inf')
    
    # Vectorized min triangle area
    # i, j, k
    # We can precompute cross products? 
    # For N=14, 14*13*12 / 6 = 364 triangles. Brute force is fine.
    
    # Using the loop method from _score but without hull calculation for speed in optimization
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area

def _min_mixed_triangle_area(r, theta):
    """
    Calculate the minimum area of triangles with vertices from different rings.
    Types:
    1. Two outer, one inner (OOI)
    2. Two inner, one outer (OII)
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    min_area = float('inf')
    
    # Case 1: Two outer (O1, O2), One inner (I)
    # Area = 0.5 * | (O1 - I) x (O2 - I) |
    # Since O1, O2 are on unit circle, let their angular difference be delta.
    # The area depends on the position of I relative to the chord O1O2.
    # Actually, let's just iterate. 7*7*7 = 343 combinations. Very fast.
    
    for i1 in range(7):
        for i2 in range(7):
            if i1 == i2: continue
            o1 = outer_pts[i1]
            o2 = outer_pts[i2]
            for i3 in range(7):
                in1 = inner_pts[i3]
                # Area of triangle o1, o2, in1
                # 0.5 * | (o1-in1) x (o2-in1) |
                v1 = o1 - in1
                v2 = o2 - in1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
                    
    # Case 2: Two inner (I1, I2), One outer (O)
    for i1 in range(7):
        for i2 in range(7):
            if i1 == i2: continue
            in1 = inner_pts[i1]
            in2 = inner_pts[i2]
            for i3 in range(7):
                o1 = outer_pts[i3]
                # Area of triangle in1, in2, o1
                v1 = in1 - o1
                v2 = in2 - o1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
                    
    return min_area

def _min_pure_triangle_area(r, theta):
    """
    Calculate minimum area of triangles with vertices from the same ring.
    Types:
    1. Three outer (OOO)
    2. Three inner (III)
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    min_area = float('inf')
    
    # Case 1: Three outer
    # Since they are on a unit circle, the area is determined by the angular gaps.
    # The minimum area triangle in a regular heptagon is formed by 3 consecutive vertices?
    # Or vertices with gaps 1,1,5? No, gaps must sum to 7.
    # Possible gap sets (cyclic): (1,1,5), (1,2,4), (2,2,3).
    # Area for gaps (a,b,c) is 0.5 * R^2 * (sin(2pi*a/7) + sin(2pi*b/7) + sin(2pi*c/7)) ? No.
    # Area = 0.5 * R^2 * | sin(theta2-theta1) + sin(theta3-theta2) + sin(theta1-theta3) |
    # Let's just compute the min for the regular heptagon once.
    # It is constant for outer.
    min_outer = float('inf')
    for i in range(7):
        for j in range(i+1, 7):
            for k in range(j+1, 7):
                o1, o2, o3 = outer_pts[i], outer_pts[j], outer_pts[k]
                v1 = o1 - o3
                v2 = o2 - o3
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_outer:
                    min_outer = area
                    
    # Case 2: Three inner
    # Same geometry, scaled by r^2.
    min_inner = min_outer * (r * r)
    
    return min(min_outer, min_inner)

def _objective_for_theta(theta, r):
    """
    Compute the minimum triangle area for a given r and theta.
    We assume the bottleneck is either mixed or pure.
    """
    a_mixed = _min_mixed_triangle_area(r, theta)
    a_pure = _min_pure_triangle_area(r, theta)
    return min(a_mixed, a_pure)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Determine optimal r analytically or via 1D search for a fixed theta?
    # The strategy says: "inner radius is fixed analytically to maximize the minimum area of 'mixed' triangles"
    # Then "perform a 1D optimization solely over the relative angular rotation to balance the 'pure' triangle areas against the 'mixed' ones"
    
    # Let's first find the best r for a set of candidate thetas, or optimize jointly in 2D but very fast.
    # Actually, the prompt suggests:
    # 1. Fix r to maximize min mixed area.
    # 2. Optimize theta to balance pure and mixed.
    
    # Let's analyze _min_mixed_triangle_area(r, theta).
    # It depends on theta. So "fixed analytically" might mean for a *specific* theta, or we find r(theta).
    # Given the symmetry, the worst-case mixed triangle area likely occurs at specific alignments.
    
    # Let's do a coarse grid search over r and theta to find the landscape, then refine.
    # This is much faster than 28D search.
    
    r_range = np.linspace(0.3, 0.9, 50)
    theta_range = np.linspace(0, 2*np.pi/7, 50, endpoint=False)
    
    best_score = -float('inf')
    best_r = 0.5
    best_theta = np.pi/7
    best_pts = None
    
    # Phase 1: Coarse Grid
    for r in r_range:
        if time.time() > deadline - 20.0:
            break
        for theta in theta_range:
            if time.time() > deadline - 20.0:
                break
            
            # Calculate min area components
            a_mixed = _min_mixed_triangle_area(r, theta)
            a_pure = _min_pure_triangle_area(r, theta)
            min_area = min(a_mixed, a_pure)
            
            # Hull area for two concentric heptagons:
            # If r < 1, the convex hull is the outer heptagon.
            # Area of regular heptagon with circumradius R: 0.5 * R^2 * 7 * sin(2pi/7)
            hull_area = 0.5 * 1.0 * 7 * np.sin(2*np.pi/7)
            
            score = min_area / hull_area
            
            if score > best_score:
                best_score = score
                best_r = r
                best_theta = theta
                best_pts = _generate_two_ring_points(r, theta)
                
    # Phase 2: Refine with scipy minimize_scalar on theta, keeping r fixed? 
    # Or joint 2D minimization? 2D is still very fast.
    
    def _neg_score_2d(params):
        r, theta = params
        if r < 0.1 or r > 1.5:
            return 1e9
        a_mixed = _min_mixed_triangle_area(r, theta)
        a_pure = _min_pure_triangle_area(r, theta)
        min_area = min(a_mixed, a_pure)
        hull_area = 0.5 * 1.0 * 7 * np.sin(2*np.pi/7)
        return -min_area / hull_area
        
    # Use Nelder-Mead for 2D refinement (r, theta)
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_neg_score_2d, [best_r, best_theta], method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 200})
            
            r_opt, theta_opt = res.x
            # Ensure r is within reasonable bounds
            if 0.1 < r_opt < 1.5:
                pts_opt = _generate_two_ring_points(r_opt, theta_opt)
                score_opt = _score(pts_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt
        except Exception:
            pass

    # Fallback if optimization failed
    if best_pts is None:
        best_pts = _generate_two_ring_points(0.6, np.pi/7)
        best_score = _score(best_pts)
        
    return best_pts

# Need to import minimize for Nelder-Mead
from scipy.optimize import minimize

# EVOLVE-BLOCK-END
