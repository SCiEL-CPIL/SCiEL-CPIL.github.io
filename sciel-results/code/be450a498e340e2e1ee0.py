"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

def construct():
    H = np.sqrt(3) / 2
    indices = list(combinations(range(N), 3))
    
    def min_area_fast(points):
        """Fast calculation of minimum triangle area."""
        xs = points[:, 0]
        ys = points[:, 1]
        min_a = float('inf')
        for i in range(N-2):
            xi, yi = xs[i], ys[i]
            for j in range(i+1, N-1):
                xj, yj = xs[j], ys[j]
                dx_ji = xj - xi
                dy_ji = yj - yi
                for k in range(j+1, N):
                    xk, yk = xs[k], ys[k]
                    dx_ki = xk - xi
                    dy_ki = yk - yi
                    det = dx_ji * dy_ki - dy_ji * dx_ki
                    a = 0.5 * abs(det)
                    if a < min_a:
                        min_a = a
        return min_a

    def is_inside(points):
        """Check if all points are inside the equilateral triangle."""
        for p in points:
            x, y = p
            if y < -1e-9:
                return False
            if y > np.sqrt(3)*x + 1e-9:
                return False
            if y > np.sqrt(3)*(1-x) + 1e-9:
                return False
            if x < -1e-9 or x > 1 + 1e-9:
                return False
        return True

    def objective(coords):
        """Objective function to minimize (negative of min area).
        Adds penalty if points are outside."""
        points = coords.reshape(-1, 2)
        if not is_inside(points):
            # Calculate penalty based on distance outside
            pen = 0.0
            for p in points:
                x, y = p
                if y < 0: pen += -y
                if y > np.sqrt(3)*x: pen += y - np.sqrt(3)*x
                if y > np.sqrt(3)*(1-x): pen += y - np.sqrt(3)*(1-x)
                if x < 0: pen += -x
                if x > 1: pen += x - 1
            return 100.0 + pen
        return -min_area_fast(points)

    def random_points(rng):
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    pts[i, 0] = u + 0.5 * v
                    pts[i, 1] = H * v
                    break
        return pts

    rng = np.random.RandomState(123)
    best_points = None
    best_score = -1
    
    n_restarts = 50
    
    for r in range(n_restarts):
        x0 = random_points(rng).flatten()
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            final_points = res.x.reshape(-1, 2)
            
            if not is_inside(final_points):
                continue
                
            current_score = min_area_fast(final_points)
            
            if current_score > best_score:
                best_score = current_score
                best_points = final_points
                
        except Exception:
            continue
            
    if best_points is None:
        return random_points(rng)
        
    return best_points
# EVOLVE-BLOCK-END
