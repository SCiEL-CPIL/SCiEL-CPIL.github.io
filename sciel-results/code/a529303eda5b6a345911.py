import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def calculate_score(points):
    """
    Calculate the score: minimum triangle area divided by convex hull area.
    points: array of shape (N, 2)
    """
    try:
        # Calculate areas of all triangles
        # There are C(N, 3) = 286 triangles
        # Vectorized calculation for efficiency
        n = len(points)
        # Get indices
        i = np.repeat(np.arange(n), n*(n-1)//2)
        # This construction is a bit complex to vectorize fully without loops for indices
        # Let's use a simple loop for clarity first, then optimize if needed.
        # With N=13, 286 triangles is very small.
        
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Area = 0.5 * |(p2-p1) x (p3-p1)|
                    v1 = p2 - p1
                    v2 = p3 - p1
                    cross = v1[0]*v2[1] - v1[1]*v2[0]
                    areas.append(abs(cross) * 0.5)
        
        min_area = np.min(areas)
        
        # Calculate hull area
        # Check if points are collinear or degenerate
        if n < 3:
            return 0.0
        
        # ConvexHull requires at least 3 non-collinear points
        # To be safe, check if all points are the same or collinear
        # A simple check: if min area is 0, score is 0
        if min_area < 1e-9:
            return 0.0
            
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume # In 2D, volume is area
        except Exception:
            return 0.0
            
        if hull_area < 1e-9:
            return 0.0
            
        return min_area / hull_area

    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Center at 0, radius 1
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_score = calculate_score(init_points)
    best_points = init_points.copy()
    
    # Define objective function to minimize (negative score)
    def objective(x):
        pts = x.reshape(-1, 2)
        # Optional: Normalize to prevent drift, though score is scale invariant
        # Let's just return -score
        return -calculate_score(pts)
    
    # Multi-start Nelder-Mead
    # We will try a few different initial configurations
    # 1. Regular polygon
    # 2. Regular polygon with small noise
    # 3. Slightly deformed regular polygon (alternating radii)
    
    configs = []
    
    # Config 1: Regular
    configs.append(init_points.flatten())
    
    # Config 2: Regular with noise
    rng = np.random.default_rng(42)
    noisy = init_points + rng.normal(0, 0.01, size=init_points.shape)
    configs.append(noisy.flatten())
    
    # Config 3: Alternating radii (Star-like but convex? No, must be convex hull)
    # Let's try a regular polygon but with slightly varying radii
    radii = np.ones(N)
    radii[::2] = 1.05
    radii[1::2] = 0.95
    # Ensure convexity: if radii vary too much, points might go inside
    # For N=13, small variation is safe
    alt_points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    configs.append(alt_points.flatten())
    
    # Config 4: Another noise seed
    rng2 = np.random.default_rng(123)
    noisy2 = init_points + rng2.normal(0, 0.05, size=init_points.shape)
    configs.append(noisy2.flatten())
    
    for i, x0 in enumerate(configs):
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(-1, 2)
                    
        except Exception as e:
            pass

    return best_points

# EVOLVE-BLOCK-START
# The code above is the implementation. 
# The prompt asks to keep the function names and signatures.
# The current program has `construct()` returning an array.
# I will place the logic inside the EVOLVE-BLOCK as requested by the structure, 
# or replace the whole block if the structure implies replacing the content.
# The prompt says: "Keep the function names and signatures of the seed program."
# "Output the COMPLETE program in a single python code block."
# I will provide the full script.
# EVOLVE-BLOCK-END
