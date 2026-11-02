"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized for speed.
    """
    # points shape: (N, 2)
    # We need to compute area for all triples (i, j, k) with i < j < k
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i) |
    
    # Efficient way: 
    # Let P be (N, 2).
    # We can use broadcasting to compute vectors P_j - P_i and P_k - P_i.
    # But memory for N=14 is small, so we can just use loops or simple vectorization.
    # For N=14, C(14,3) = 364 triangles. This is very fast.
    
    min_area = float('inf')
    # Pre-compute coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(N):
        for j in range(i + 1, N):
            # Vectorized calculation of cross product for k > j
            # Cross product magnitude for vectors (P_j - P_i) and (P_k - P_i)
            # (x_j - x_i) * (y_k - y_i) - (x_k - x_i) * (y_j - y_i)
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            # For all k > j
            # We need k such that k > j
            if j + 1 >= N:
                continue
                
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # Fallback if hull fails (e.g., collinear points)
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _negative_score(points_flat):
    """
    Objective function for optimization.
    We want to maximize score, so we minimize negative score.
    points_flat: 1D array of length 2*N
    """
    points = points_flat.reshape(N, 2)
    
    # Check for degeneracy
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    score = _score(points)
    
    # If score is 0, it's a bad configuration (degenerate hull or triangle)
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Strategy:
    1. Initialize with regular 14-gon (known good starting point).
    2. Use Nelder-Mead optimization to refine the points.
    3. Normalize the final result to unit area.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization: Regular 14-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Flatten for optimizer
    x0 = initial_points.flatten()
    
    # 2. Optimization
    # Nelder-Mead is robust for this kind of non-differentiable objective (min of areas)
    # We maximize min_area / hull_area.
    # Note: The ratio is scale-invariant. So we can optimize in any scale, 
    # but keeping it around unit scale helps numerical stability.
    
    best_x = x0
    best_score = -_negative_score(x0)
    
    # Try a few different initializations to avoid local minima
    # 1. Regular polygon
    # 2. Regular polygon with slight noise
    # 3. Points on an ellipse
    
    initializations = []
    
    # Regular polygon
    initializations.append(x0)
    
    # Noisy regular polygon
    rng = np.random.RandomState(42)
    noisy_angles = angles + rng.normal(0, 0.05, N)
    noisy_points = np.column_stack([np.cos(noisy_angles), np.sin(noisy_angles)])
    initializations.append(noisy_points.flatten())
    
    # Ellipse points (aspect ratio 1.5)
    ellipse_points = np.column_stack([1.5 * np.cos(angles), np.sin(angles)])
    initializations.append(ellipse_points.flatten())
    
    # Square-ish distribution (corners + midpoints? No, must be convex region points)
    # Let's try points on a square boundary
    # 14 points on a square. 
    # 4 corners, 10 distributed on edges.
    # This might be a different local minimum.
    square_pts = []
    # Bottom edge: 4 points
    for i in range(4):
        square_pts.append([i/3.0, 0.0])
    # Right edge: 4 points
    for i in range(4):
        square_pts.append([1.0, (i+1)/4.0])
    # Top edge: 3 points
    for i in range(3):
        square_pts.append([1.0 - (i+1)/3.0, 1.0])
    # Left edge: 3 points
    for i in range(3):
        square_pts.append([0.0, 1.0 - (i+1)/3.0])
        
    square_arr = np.array(square_pts)
    initializations.append(square_arr.flatten())
    
    # Optimize each initialization
    for init_x in initializations:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                _negative_score, 
                init_x, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_x = res.x
                
        except Exception:
            pass
            
    # 3. Final Polish with smaller steps or different method if time permits
    if time.time() < deadline - 20:
        try:
            # Use Powell method which can be faster for this
            res2 = minimize(
                _negative_score, 
                best_x, 
                method='Powell', 
                options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            current_score = -res2.fun
            if current_score > best_score:
                best_score = current_score
                best_x = res2.x
        except Exception:
            pass

    # 4. Normalize to unit area
    final_points = best_x.reshape(N, 2)
    
    # Calculate hull area
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        # Fallback
        hull_area = 1.0
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    else:
        # If hull area is 0, return regular polygon as fallback
        final_points = initial_points.copy()
        hull_area = np.pi * (1.0) # Approx area of unit circle hull? No, regular 14-gon area.
        # Just normalize anyway
        try:
            hull = ConvexHull(final_points)
            hull_area = hull.volume
            if hull_area > 1e-10:
                scale = np.sqrt(1.0 / hull_area)
                final_points = final_points * scale
        except:
            pass

    return final_points
# EVOLVE-BLOCK-END
