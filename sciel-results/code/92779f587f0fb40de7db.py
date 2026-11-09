# EVOLVE-BLOCK-START
import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 590  # Leave 40s margin from 630s limit
    
    # Precompute indices for all 286 triangles
    tri_indices = np.array(list(itertools.combinations(range(n), 3)))
    i_idx, j_idx, k_idx = tri_indices[:, 0], tri_indices[:, 1], tri_indices[:, 2]

    def calculate_score(points):
        # points: (13, 2)
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 0.0
        except Exception:
            return 0.0
        
        # Vectorized triangle area calculation
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 1e-10:
            return 0.0
            
        return A_min / A_hull

    def neg_score(points):
        s = calculate_score(points)
        return -s

    best_score = 0.0
    best_points = None

    # Generate initial guesses
    initial_guesses = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_guesses.append(reg_poly)
    
    rng = np.random.default_rng(42)
    
    # 2. Jittered Regular Polygon (20 variations)
    for _ in range(20):
        jitter = rng.normal(0, 0.2, (n, 2))
        guess = reg_poly + jitter
        initial_guesses.append(guess)
        
    # 3. Random points (20 variations)
    for _ in range(20):
        guess = rng.random((n, 2))
        initial_guesses.append(guess)
        
    # 4. "Square" based (4x4 grid subset)
    # Create a 4x4 grid
    grid_x = np.linspace(0, 1, 4)
    grid_y = np.linspace(0, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    # Pick 13 random points from the 16
    for _ in range(10):
        idx = rng.choice(16, 13, replace=False)
        guess = grid_points[idx]
        initial_guesses.append(guess)

    # Optimization Loop
    for guess in initial_guesses:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Normalize
        guess_centered = guess - np.mean(guess, axis=0)
        max_dist = np.max(np.linalg.norm(guess_centered, axis=1))
        if max_dist > 1e-10:
            guess_normalized = guess_centered / max_dist
        else:
            guess_normalized = guess_centered
        
        # Run Nelder-Mead
        try:
            result = scipy.optimize.minimize(
                neg_score, 
                guess_normalized, 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            
            if result.success or result.fun < 0: # Check if it found a valid config
                current_score = -result.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = result.x
        except Exception:
            pass

    # If time remains, refine the best point with Powell
    if best_points is not None and time.time() - start_time < time_limit - 30:
        try:
            result_powell = scipy.optimize.minimize(
                neg_score, 
                best_points, 
                method='Powell',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result_powell.fun < -best_score: # i.e. score increased
                best_score = -result_powell.fun
                best_points = result_powell.x
        except Exception:
            pass

    if best_points is None:
        best_points = initial_guesses[0]
        
    return best_points

# EVOLVE-BLOCK-END
