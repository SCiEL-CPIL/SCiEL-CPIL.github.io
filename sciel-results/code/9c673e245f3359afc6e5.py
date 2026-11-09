import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize

N = 14

# Precompute triple indices for N=14
_i_list, _j_list, _k_list = [], [], []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _i_list.append(_i)
            _j_list.append(_j)
            _k_list.append(_k)
_TRIPLE_I = np.array(_i_list)
_TRIPLE_J = np.array(_j_list)
_TRIPLE_K = np.array(_k_list)

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    # 1. Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # 2. Min Triangle Area (Vectorized)
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]

    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0

    return min_area / hull_area

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop searching at 140s to be safe.
    deadline = start_time + 140.0

    # Objective function for minimizers (maximize score -> minimize -score)
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        return -get_score(pts)

    best_points = None
    best_score = 0.0

    # Strategy: 
    # The problem is to maximize the minimum triangle area normalized by hull area.
    # This is a "fairness" problem. Points should be well-distributed.
    # A regular polygon is a strong candidate. Perturbing it might help if the regular polygon 
    # has some "weak" triangles (e.g., if points are collinear or close in angle).
    # However, for a regular polygon, all points are on the convex hull.
    # If we move points inward, the hull area decreases, which might increase the ratio 
    # if the min triangle area doesn't decrease as fast. But typically, keeping points on the hull is good.
    # Let's focus on optimizing the positions of points on the convex hull, or allowing slight inward movement.
    
    # Actually, a known result for small N is that the optimal configuration is often a regular polygon or a slight perturbation.
    # For N=14, let's try:
    # 1. Regular 14-gon
    # 2. Perturbed regular 14-gon (random noise)
    # 3. Two concentric polygons? (7 inside, 7 outside) - unlikely to be optimal for min triangle area as inner points create small triangles with outer points.
    # 4. Grid-like?
    
    # Let's stick to optimizing the 2D coordinates directly, starting from good initializations.
    # Powell's method is robust for this.
    
    initializations = []
    rng = np.random.default_rng(42)

    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    reg_poly = np.column_stack((x, y))
    initializations.append(reg_poly)
    
    # 2. Perturbed Regular Polygons
    # Small perturbations to break symmetry if the regular polygon isn't optimal.
    for i in range(30):
        # Random rotation
        theta = rng.uniform(0, 2*np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        pts = reg_poly @ R.T
        
        # Random radial perturbation
        # Keep points mostly on the hull, but allow some inward movement
        # If we move inward, hull area shrinks. 
        # Let's try uniform radial perturbation in [0.9, 1.0]
        radii = rng.uniform(0.8, 1.0, N)
        pts = pts * radii[:, np.newaxis]
        
        # Small angular perturbation
        ang_noise = rng.normal(0, 0.05, N)
        new_angles = angles + ang_noise
        # Sort angles to maintain convexity order? No, just use the points.
        # Actually, if we perturb angles and radii, we might lose convexity.
        # Let's just use the points as is. If they are not convex, ConvexHull will handle it.
        # But we want them to be in convex position ideally.
        
        initializations.append(pts)

    # 3. Random points in unit disk, filtered for "goodness"
    # We want points that are well-separated.
    for i in range(50):
        # Sample random points
        pts = rng.uniform(-1, 1, (N, 2))
        # Check if they form a convex hull with decent area
        try:
            hull = ConvexHull(pts)
            if hull.volume > 0.5: # Arbitrary threshold
                initializations.append(pts)
        except:
            pass

    # 4. "Spiral" or other structured patterns?
    # Maybe a 2-row grid?
    # Let's try a few structured grids.
    # 7 points on top, 7 on bottom?
    # This might create very small triangles if rows are close.
    
    # Let's run Powell on each initialization.
    for init_pts in initializations:
        if time.time() > deadline - 5.0:
            break
            
        try:
            # Powell's method
            res = minimize(
                objective, 
                init_pts.flatten(), 
                method='Powell', 
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 50000}
            )
            
            if res.fun < -best_score:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(N, 2)
                    
        except Exception:
            pass

    # If we have time, do a final fine-tuning with Nelder-Mead on the best point
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            res_nm = minimize(
                objective, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 10000, 'maxfev': 100000}
            )
            if res_nm.fun < -best_score:
                best_score = -res_nm.fun
                best_points = res_nm.x.reshape(N, 2)
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        best_points = np.column_stack((x, y))

    return best_points
