"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(points):
    """Compute the maximum common radius for a set of centers in [-0.5, 0.5]^2."""
    # points: (N, 2)
    # Boundary distance: min over i,k of (0.5 - |c_ik|)
    # Pairwise: min over i<j of dist(i,j)/2
    # r = min(boundary_min, pairwise_min)
    
    # Boundary
    # c_ik in [-0.5, 0.5]
    # distance to boundary for point (x,y) is min(0.5-x, 0.5+x, 0.5-y, 0.5+y)
    # = 0.5 - max(|x|, |y|)  ... wait, no.
    # distance to left is 0.5 - x
    # distance to right is 0.5 + x
    # distance to bottom is 0.5 - y
    # distance to top is 0.5 + y
    # min of these is 0.5 - max(|x|, |y|) ?
    # Let x = 0.4. 0.5-0.4=0.1, 0.5+0.4=0.9. min=0.1. 0.5-max(0.4, |y|). If |y|<0.4, max=0.4, 0.5-0.4=0.1. Correct.
    # Let x = -0.4. 0.5-(-0.4)=0.9, 0.5+(-0.4)=0.1. min=0.1. 0.5-max(0.4, |y|). Correct.
    # So boundary radius for point i is 0.5 - max(|x_i|, |y_i|).
    # We need min over i of this.
    
    abs_pts = np.abs(points)
    max_coord = np.max(abs_pts, axis=1)
    boundary_radii = 0.5 - max_coord
    min_boundary = np.min(boundary_radii)
    
    # Pairwise
    # Use scipy.spatial.distance or just broadcasting for N=200
    # N=200 -> 200x200 matrix is 40k elements, very fast.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    
    return min(min_boundary, min_pairwise)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial guess: Hexagonal packing
    # Or random
    # Let's use a mix of structured and random restarts
    
    best_points = None
    best_score = -1.0
    
    # 1. Structured Initializations
    # Square Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    init_square = grid[:N]
    
    # Hexagonal Grid (approx)
    # Hex packing density is higher.
    # Rows offset by half spacing.
    # Let's generate a hex grid and crop.
    hex_pts = []
    row = 0
    while len(hex_pts) < N + 10:
        y = (row * np.sqrt(3) / 2) - 0.5 * (row * np.sqrt(3) / 2) # Center vertically later
        # Let's just generate a bunch and center them
        for col in range(m + 2):
            x = col
            if row % 2 == 1:
                x += 0.5
            hex_pts.append([x, y])
        row += 1
    hex_pts = np.array(hex_pts)
    # Center hex_pts
    hex_pts -= np.mean(hex_pts, axis=0)
    # Scale to fit in [-0.5, 0.5]? 
    # We want to maximize radius, so we want to pack as tightly as possible.
    # The optimization will handle scaling/positioning.
    # Just take the first N points.
    init_hex = hex_pts[:N]
    
    # Random Initializations
    np.random.seed(42)
    inits = [init_square, init_hex]
    for _ in range(5):
        inits.append(np.random.uniform(-0.5, 0.5, (N, 2)))
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Local Optimization using BFGS or Nelder-Mead
        # Objective: maximize r. Minimize -r.
        # r is non-smooth (min of smooth functions).
        # We can use a smooth approximation or just minimize the negative min.
        # Nelder-Mead is derivative-free and handles non-smoothness well.
        
        def objective(points):
            # points is flattened
            pts = points.reshape(N, 2)
            # Penalty for going outside? 
            # The score function handles it by reducing r to 0 (or negative? no, radius is positive distance).
            # If a point is outside, boundary distance is negative.
            # min_boundary becomes negative.
            # Score becomes negative.
            # We want to maximize score. So minimize -score.
            # If score is negative, -score is positive (bad).
            # So the optimizer will naturally push points inside.
            return -_score(pts)
        
        # Try Nelder-Mead
        try:
            res = minimize(objective, init.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            if time.time() > deadline:
                break
            pts = res.x.reshape(N, 2)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
        except Exception:
            continue
            
    # If we have time, do a high-precision polish on the best solution
    if best_points is not None and time.time() < deadline - 5.0:
        # Use BFGS or L-BFGS-B on the best solution?
        # Nelder-Mead is often better for this. Let's do another NM with tighter tol.
        def objective(points):
            pts = points.reshape(N, 2)
            return -_score(pts)
        
        try:
            res = minimize(objective, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-10})
            pts = res.x.reshape(N, 2)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback to square grid
        return init_square
        
    return best_points
# EVOLVE-BLOCK-END
