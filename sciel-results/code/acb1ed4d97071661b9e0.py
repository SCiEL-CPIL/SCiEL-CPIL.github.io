import numpy as np
from scipy.optimize import minimize, basinhopping
import time
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3)

# Precompute all triplets
TRIPLETS = list(combinations(range(N), 3))
N_TRIPLETS = len(TRIPLETS)  # 165

def compute_signed_areas(points):
    """
    Compute signed areas for all 165 triangles.
    Returns array of shape (165,) with signed area values.
    """
    n = points.shape[0]
    areas = np.zeros(N_TRIPLETS)
    for idx, (i, j, k) in enumerate(TRIPLETS):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Signed area = 0.5 * (x1(y2-y3) + x2(y3-y1) + x3(y1-y2))
        areas[idx] = 0.5 * (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
    return areas

def triangle_areas_vectorized(points):
    """
    Calculate the areas of all triangles formed by the points using vectorization.
    """
    signed = compute_signed_areas(points)
    return np.abs(signed)

def penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def nqp_objective(flat_params):
    """
    NQP-inspired objective: maximize min |signed_area| using smooth approximation.
    Uses log-sum-exp style smoothing for the min operation and handles sign flips.
    """
    points = flat_params.reshape(N, 2)
    signed_areas = compute_signed_areas(points)
    abs_areas = np.abs(signed_areas)
    
    # Smooth minimum using log-sum-exp trick (negative log-sum-exp gives smooth min)
    # min(x) ≈ -logsumexp(-x) / 1
    # Use temperature for smoothness
    temp = 0.01
    neg_areas = -abs_areas / temp
    neg_areas -= np.max(neg_areas)  # for numerical stability
    smooth_min = -temp * np.log(np.sum(np.exp(neg_areas)))
    
    pen = penalty(points)
    return -smooth_min + 1000.0 * pen

def nqp_gradient_descent(points, lr=0.001, n_iter=500, momentum=0.9):
    """
    Custom gradient descent with momentum for the NQP formulation.
    Uses numerical gradients for the smooth min objective.
    """
    v = np.zeros_like(points)
    best_points = points.copy()
    best_score = -np.inf
    
    for it in range(n_iter):
        # Compute current score
        signed_areas = compute_signed_areas(points)
        abs_areas = np.abs(signed_areas)
        min_area = np.min(abs_areas)
        pen = penalty(points)
        score = min_area - 1000 * pen
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Compute numerical gradient of -min_area (we want to maximize min_area)
        # Use central differences
        grad = np.zeros_like(points)
        eps = 1e-5
        
        for i in range(N):
            for d in range(2):
                old_val = points[i, d]
                points[i, d] = old_val + eps
                signed_plus = compute_signed_areas(points)
                min_plus = np.min(np.abs(signed_plus))
                pen_plus = penalty(points)
                obj_plus = -min_plus + 1000 * pen_plus
                
                points[i, d] = old_val - eps
                signed_minus = compute_signed_areas(points)
                min_minus = np.min(np.abs(signed_minus))
                pen_minus = penalty(points)
                obj_minus = -min_minus + 1000 * pen_minus
                
                points[i, d] = old_val
                grad[i, d] = (obj_plus - obj_minus) / (2 * eps)
        
        # Update with momentum
        v = momentum * v - lr * grad
        points = points + v
        
        # Project back into triangle if needed
        x = points[:, 0]
        y = points[:, 1]
        # Soft projection: nudge points that are outside
        outside_mask = (y < -0.01) | (SQRT3 * x - y < -0.01) | (SQRT3 - SQRT3 * x - y < -0.01)
        if np.any(outside_mask):
            points[outside_mask] = 0.5 * points[outside_mask] + 0.5 * np.array([0.5, SQRT3/6])
    
    return best_points, best_score

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Symmetric initial guess based on known good configurations
    def get_symmetric_init():
        """
        Construct a symmetric initial configuration for 11 points.
        Uses barycentric coordinates for symmetry.
        """
        h = SQRT3 / 2
        pts = []
        
        # Centroid
        pts.append([0.5, h/3])
        
        # Three points at 1/3 height, symmetric
        pts.append([0.25, h/6])
        pts.append([0.75, h/6])
        pts.append([0.5, h/3 + 0.1])
        
        # Three points at 2/3 height, symmetric
        pts.append([0.25, h/2])
        pts.append([0.75, h/2])
        pts.append([0.5, h*2/3])
        
        # Three points near base
        pts.append([0.1, h/12])
        pts.append([0.9, h/12])
        pts.append([0.5, h/12])
        
        return np.array(pts)
    
    # Try NQP gradient descent from multiple starts
    for seed in range(50):
        if time.time() - start_time > time_limit - 15:
            break
        
        rng = np.random.RandomState(seed)
        
        if seed == 0:
            init_points = get_symmetric_init()
        else:
            # Random init in triangle
            init_points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u = rng.rand()
                    v = rng.rand()
                    if u + v <= 1:
                        break
                w = 1 - u - v
                init_points[i, 0] = v * 1.0 + w * 0.5
                init_points[i, 1] = w * SQRT3 / 2.0
        
        # Run NQP gradient descent
        try:
            gd_points, gd_score = nqp_gradient_descent(init_points, lr=0.005, n_iter=300, momentum=0.9)
            if gd_score > best_score:
                best_score = gd_score
                best_points = gd_points.copy()
        except Exception:
            pass
        
        # Also try Powell on the smooth objective for refinement
        if time.time() - start_time > time_limit - 10:
            break
        try:
            result = minimize(
                nqp_objective,
                init_points.flatten(),
                method='Powell',
                options={'maxiter': 100, 'maxfev': 5000, 'disp': False}
            )
            pw_points = result.x.reshape(N, 2)
            min_area = np.min(triangle_areas_vectorized(pw_points))
            pen = penalty(pw_points)
            if pen < 1e-4:
                pw_score = min_area
            else:
                pw_score = min_area - 1000 * pen
            
            if pw_score > best_score:
                best_score = pw_score
                best_points = pw_points.copy()
        except Exception:
            pass
    
    # Final refinement with basinhopping if time allows
    if best_points is not None and time.time() - start_time < time_limit - 10:
        try:
            bh_result = basinhopping(
                nqp_objective,
                best_points.flatten(),
                minimizer_kwargs={'method': 'Powell', 'maxiter': 50, 'maxfev': 3000},
                niter=10,
                seed=42
            )
            bh_points = bh_result.x.reshape(N, 2)
            bh_min_area = np.min(triangle_areas_vectorized(bh_points))
            bh_pen = penalty(bh_points)
            
            if bh_pen < 1e-4:
                bh_score = bh_min_area
            else:
                bh_score = bh_min_area - 1000 * bh_pen
                
            if bh_score > best_score:
                best_score = bh_score
                best_points = bh_points.copy()
        except Exception:
            pass
    
    if best_points is None:
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
