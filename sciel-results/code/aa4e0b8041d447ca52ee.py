# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Precompute indices for all triangles
    # i < j < k
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    indices = np.array(indices)
    # indices shape: (286, 3)
    idx_i = indices[:, 0]
    idx_j = indices[:, 1]
    idx_k = indices[:, 2]

    def objective(x):
        """
        Minimize the negative ratio A_min / A_hull.
        x is a flattened array of 26 coordinates.
        """
        # Check time
        if time.time() > deadline:
            return 1e10  # Large penalty to stop optimization
        
        points = x.reshape(n, 2)
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(points)):
            return 1e10
            
        # Compute triangle areas vectorized
        # p_i, p_j, p_k
        pi = points[idx_i]
        pj = points[idx_j]
        pk = points[idx_k]
        
        # Cross product of (pj - pi) and (pk - pi)
        # (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = np.abs(cross) * 0.5
        
        min_area = np.min(areas)
        
        # If min area is 0, return large penalty
        if min_area < 1e-9:
            return 1e10

        # Compute Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 1e10
            
        if hull_area < 1e-9:
            return 1e10

        ratio = min_area / hull_area
        return -ratio

    best_score = -1e10
    best_points = None

    # Generate initial guesses
    initial_guesses = []
    
    # 1. Regular 12-gon + center
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    p1 = np.vstack([ring, center])
    initial_guesses.append(p1.flatten())
    
    # 2. Hexagonal lattice (approx)
    # 3 rings of hex lattice
    # Center (0,0)
    # 6 neighbors
    # 12 further neighbors
    # Let's just use a grid and pick 13 points
    grid_x = np.arange(-2, 3)
    grid_y = np.arange(-2, 3)
    gx, gy = np.meshgrid(grid_x, grid_y)
    # Hexagonal distortion
    # y' = y + 0.5 * (x % 2)
    hex_points = []
    for i in range(len(gx)):
        for j in range(len(gy)):
            x = gx[i, j]
            y = gy[i, j]
            if (x % 2 != 0):
                y += 0.5
            if x*x + y*y < 4.5: # radius ~2.1
                hex_points.append([x, y])
    if len(hex_points) >= 13:
        # Pick 13 points, maybe center + 12 surrounding
        # Just take first 13 or optimize a subset
        # For simplicity, take a random subset of 13 from the lattice points inside
        rng = np.random.default_rng(42)
        if len(hex_points) > 13:
            idxs = rng.choice(len(hex_points), 13, replace=False)
            p2 = np.array([hex_points[i] for i in idxs])
        else:
            p2 = np.array(hex_points[:13])
        initial_guesses.append(p2.flatten())

    # 3. Random points in a circle
    rng = np.random.default_rng(123)
    for _ in range(5):
        angles = rng.uniform(0, 2 * np.pi, n)
        radii = rng.uniform(0, 1, n)
        p_rand = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initial_guesses.append(p_rand.flatten())

    # 4. Regular 13-gon (all on hull)
    angles13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    p13 = np.column_stack([np.cos(angles13), np.sin(angles13)])
    initial_guesses.append(p13.flatten())

    # Run optimization for each guess
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead for robustness without gradient
            # Maximize iterations within time
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < 0: # Valid score
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
        except Exception:
            continue

    # If no valid points found, fallback to regular 12-gon + center
    if best_points is None:
        best_points = p1

    # Final check
    if not np.all(np.isfinite(best_points)):
        best_points = p1

    return best_points

# EVOLVE-BLOCK-END
