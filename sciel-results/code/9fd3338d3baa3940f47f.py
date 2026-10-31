"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.optimize
import time
import itertools
import sys

N = 11
H = np.sqrt(3) / 2
DEADLINE = 140.0 # Safety margin from 150s limit

def is_inside(points):
    """Check if all points are inside the equilateral triangle."""
    x = points[:, 0]
    y = points[:, 1]
    # Conditions:
    # 1. y >= 0
    # 2. x >= 0
    # 3. y <= H * (1 - x)  => y + H*x <= H
    if np.any(x < -1e-9) or np.any(y < -1e-9):
        return False
    if np.any(y + H * x > H + 1e-9):
        return False
    return True

def min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by the points."""
    if not is_inside(points):
        return -1.0  # Large penalty for out-of-bounds points
    
    # Pre-compute combinations for speed? No, N is small.
    min_area = float('inf')
    # Using itertools.combinations is clean but might be slightly slow in pure Python loops.
    # For N=11, C(11,3)=165. This is very fast.
    for i, j, k in itertools.combinations(range(N), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p3[0] - p1[0]
        dy2 = p3[1] - p1[1]
        area = 0.5 * abs(dx1 * dy2 - dx2 * dy1)
        if area < min_area:
            min_area = area
            if min_area < 1e-6: # Early exit if points are almost collinear
                return min_area
    return min_area

def objective(flat_points):
    """Objective function for scipy.optimize.minimize (to be minimized)."""
    points = flat_points.reshape(N, 2)
    return -min_triangle_area(points)

def random_point_in_triangle():
    """Generate a single random point inside the triangle."""
    u = np.random.random()
    v = np.random.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    # Map to triangle vertices (0,0), (1,0), (0.5, H)
    # Barycentric coordinates (1-u-v, u, v)
    x = u * 1.0 + v * 0.5
    y = v * H
    return np.array([x, y])

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_points = None
    
    # Multi-start strategy
    n_starts = 50 # We will adjust based on time
    
    for i in range(n_starts):
        if time.time() > DEADLINE:
            break
            
        # Generate initial points
        # A good initialization is often a grid or jittered grid, or random.
        # Let's try a mix: some random, some structured.
        if i % 5 == 0:
            # Structured: Jittered grid
            points = []
            # 3x3 grid + 2 extra? 11 points.
            # Let's just use random for simplicity in the loop, but maybe a better init helps.
            # For now, pure random with a fixed seed for reproducibility of the start, 
            # but we want diversity.
            pass
            
        # Generate 11 random points
        init_points = np.array([random_point_in_triangle() for _ in range(N)])
        
        # Ensure initial points are valid
        if not is_inside(init_points):
            continue
            
        # Run local optimization
        try:
            # Nelder-Mead is robust for non-smooth functions
            res = scipy.optimize.minimize(
                objective, 
                init_points.flatten(), 
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000, 'maxfev': 2000}
            )
            
            if res.success or res.fun < best_score:
                current_points = res.x.reshape(N, 2)
                # Final check
                if is_inside(current_points):
                    score = min_triangle_area(current_points)
                    if score > best_score:
                        best_score = score
                        best_points = current_points
        except Exception:
            continue
            
        # Check time
        if time.time() > DEADLINE - 5:
            break

    if best_points is None:
        # Fallback to a simple random configuration if optimization failed completely
        best_points = np.array([random_point_in_triangle() for _ in range(N)])
        # Ensure it's inside
        while not is_inside(best_points):
            best_points = np.array([random_point_in_triangle() for _ in range(N)])

    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# In the seed, the code was inside the block. 
# I will place the logic inside the block to match the structure if required, 
# but the prompt says "Keep the function names and signatures".
# The original code had `def construct():` inside the block.
# I will put the entire implementation inside the block.
# EVOLVE-BLOCK-END
