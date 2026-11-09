import numpy as np
from scipy.optimize import differential_evolution

N = 11
H = np.sqrt(3) / 2

def construct():
    # Barycentric coordinates or Cartesian with penalty?
    # Let's use Cartesian with penalty for simplicity and robustness in DE.
    
    def min_triangle_area(pts):
        # pts is (N, 2)
        min_area = np.inf
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = pts[i]
                    b = pts[j]
                    c = pts[k]
                    area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                    if area < min_area:
                        min_area = area
        return min_area

    def objective(x):
        pts = x.reshape(-1, 2)
        penalty = 0.0
        for i in range(N):
            px, py = pts[i]
            # Check inside triangle: (0,0), (1,0), (0.5, H)
            # y >= 0
            if py < 0:
                penalty += (0 - py) ** 2
            # y <= sqrt(3) * x  => py - sqrt(3)*px <= 0
            if py > np.sqrt(3) * px:
                penalty += (py - np.sqrt(3) * px) ** 2
            # y <= sqrt(3) * (1 - x) => py - sqrt(3)*(1-px) <= 0
            if py > np.sqrt(3) * (1 - px):
                penalty += (py - np.sqrt(3) * (1 - px)) ** 2
        
        if penalty > 1e-6:
            return 1.0 + penalty # Large penalty
            
        area = min_triangle_area(pts)
        return -area

    # Bounds: x in [0, 1], y in [0, H]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, H))
        
    # Initial guess: random or symmetric
    # DE generates its own initial population, but we can provide a seed or just let it run.
    # To save time, we can limit the maxiter or use a smaller population.
    # N=11, 22 variables.
    
    # Use a known good starting point? DE doesn't take a single start, it uses a population.
    # We can set `popsize` and `maxiter`.
    
    result = differential_evolution(
        objective, 
        bounds=bounds, 
        seed=42, 
        maxiter=100, 
        tol=1e-8,
        polish=True, # Polish the result with L-BFGS-B or similar? Actually polish uses minpack.
        # Note: polish=True uses `minimize` with `L-BFGS-B` on the best result. 
        # We need to ensure the objective is smooth for polish. It is.
        # However, penalty functions can be non-smooth at the boundary. 
        # Let's rely on DE itself for global search and then do a local polish manually or with DE's polish.
    )
    
    best_pts = result.x.reshape(-1, 2)
    
    # Final check and clamp
    for i in range(N):
        px, py = best_pts[i]
        # Project to triangle if slightly outside due to numerical errors
        # Simple projection: move towards centroid if outside
        cx, cy = 0.5, H/3 # Centroid is (0.5, H/3) ? No, centroid of (0,0),(1,0),(0.5,H) is (0.5, H/3).
        # Wait, H = sqrt(3)/2. Centroid y is sqrt(3)/6 = H/3. Correct.
        
        if py < 0 or py > np.sqrt(3)*px or py > np.sqrt(3)*(1-px):
             # Move 90% towards centroid
             px = 0.9 * px + 0.1 * 0.5
             py = 0.9 * py + 0.1 * (H/3)
        best_pts[i] = [px, py]
        
    return best_pts
