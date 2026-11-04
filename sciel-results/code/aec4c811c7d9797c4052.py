import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
TRI_AREA = np.sqrt(3) / 4.0  # area of unit equilateral triangle

def construct():
    t_start = time.time()
    DEADLINE = 115.0  # seconds, safety margin before 120s kill
    
    # Vertices of equilateral triangle
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3) / 2])
    
    def points_from_barycentric(coords):
        """Convert flat array of 2N barycentric pairs to Cartesian points.
        coords[i*2], coords[i*2+1] are (a, b) for point i, c = 1-a-b."""
        pts = np.zeros((N, 2))
        for i in range(N):
            a = coords[i * 2]
            b = coords[i * 2 + 1]
            c = 1.0 - a - b
            pts[i] = a * A + b * B + c * C
        return pts
    
    def barycentric_from_points(pts):
        """Convert Cartesian points to barycentric (a, b) pairs."""
        coords = np.zeros(2 * N)
        for i in range(N):
            x, y = pts[i]
            # Solve: x = a*0 + b*1 + c*0.5, y = a*0 + b*0 + c*sqrt(3)/2
            # c = 2y/sqrt(3), b = x - c*0.5, a = 1-b-c
            c = 2.0 * y / np.sqrt(3)
            b = x - 0.5 * c
            a = 1.0 - b - c
            coords[i * 2] = a
            coords[i * 2 + 1] = b
        return coords
    
    def min_triangle_area(pts):
        """Compute minimum area of triangle formed by any 3 points."""
        min_a = float('inf')
        for i, j, k in combinations(range(N), 3):
            p1, p2, p3 = pts[i], pts[j], pts[k]
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            area = 0.5 * abs(cross)
            if area < min_a:
                min_a = area
        return min_a
    
    def objective(coords):
        """Negative of smooth surrogate: we maximize sum of log(areas) to make it smooth."""
        if time.time() - t_start > DEADLINE:
            return 0.0  # signal to stop
        pts = points_from_barycentric(coords)
        # Check if all points are valid (barycentric coords in [0,1] and sum to 1)
        for i in range(N):
            a = coords[i * 2]
            b = coords[i * 2 + 1]
            c = 1.0 - a - b
            if a < -1e-6 or b < -1e-6 or c < -1e-6:
                return 1e10  # large penalty
        # Compute sum of log of all triangle areas
        total_log = 0.0
        count = 0
        for i, j, k in combinations(range(N), 3):
            p1, p2, p3 = pts[i], pts[j], pts[k]
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            area = 0.5 * abs(cross)
            if area < 1e-12:
                return 1e10  # degenerate
            total_log += np.log(area)
            count += 1
        # We want to maximize sum of log areas, so minimize negative
        return -total_log / count  # average log area
    
    # Bounds for barycentric coordinates: a >= 0, b >= 0, a+b <= 1
    # So a in [0,1], b in [0, 1-a]
    # For simplicity, use a in [0,1], b in [0,1] and rely on penalty
    lower = np.zeros(2 * N)
    upper = np.ones(2 * N)
    # Actually let's be smarter: a in [0,1], b in [0,1], but we penalize invalid
    
    best_score = 0.0
    best_pts = None
    
    # Initial points: use a grid-like or symmetric initialization
    rng = np.random.RandomState(42)
    
    # Try multiple restarts
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() - t_start > DEADLINE - 10:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Symmetric initialization: vertices + midpoints + center
            init_pts = [A, B, C,
                        (A + B) / 2, (B + C) / 2, (C + A) / 2,
                        (A + B + C) / 3,
                        (2 * A + B + C) / 4, (A + 2 * B + C) / 4,
                        (A + B + 2 * C) / 4,
                        (A + B + C) / 3 + np.array([0.01, 0.01])]
        elif restart == 1:
            # Random initialization
            init_pts = []
            for _ in range(N):
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                pt = u * A + v * B + (1 - u - v) * C
                init_pts.append(pt)
        else:
            # Perturbed best
            if best_pts is not None:
                init_pts = best_pts + rng.normal(0, 0.05, size=(N, 2))
                # Project back to triangle
                init_pts = []
                for pt in best_pts:
                    u = rng.random()
                    v = rng.random()
                    if u + v > 1:
                        u = 1 - u
                        v = 1 - v
                    pt_new = u * A + v * B + (1 - u - v) * C
                    init_pts.append(pt_new)
            else:
                init_pts = []
                for _ in range(N):
                    u = rng.random()
                    v = rng.random()
                    if u + v > 1:
                        u = 1 - u
                        v = 1 - v
                    pt = u * A + v * B + (1 - u - v) * C
                    init_pts.append(pt)
        
        init_coords = barycentric_from_points(np.array(init_pts))
        
        # Clip to valid range
        for i in range(N):
            init_coords[i * 2] = np.clip(init_coords[i * 2], 1e-6, 1 - 1e-6)
            init_coords[i * 2 + 1] = np.clip(init_coords[i * 2 + 1], 1e-6, 1 - 1e-6)
            # Ensure a + b < 1
            s = init_coords[i * 2] + init_coords[i * 2 + 1]
            if s > 1 - 1e-6:
                init_coords[i * 2] *= (1 - 1e-6) / s
                init_coords[i * 2 + 1] *= (1 - 1e-6) / s
        
        try:
            result = minimize(
                objective,
                init_coords,
                method='L-BFGS-B',
                bounds=list(zip(lower, upper)),
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
            )
        except Exception:
            continue
        
        if time.time() - t_start > DEADLINE:
            break
        
        if result.success or result.fun < 1e8:
            opt_pts = points_from_barycentric(result.x)
            # Verify all points are inside triangle
            valid = True
            for pt in opt_pts:
                x, y = pt
                if y < -1e-9 or y > np.sqrt(3) * x + 1e-9 or y > -np.sqrt(3) * (x - 1) + 1e-9:
                    valid = False
                    break
            if valid:
                score = min_triangle_area(opt_pts) / TRI_AREA
                if score > best_score:
                    best_score = score
                    best_pts = opt_pts.copy()
    
    if best_pts is None:
        # Fallback: return a simple configuration
        best_pts = np.array([
            A, B, C,
            (A + B) / 2, (B + C) / 2, (C + A) / 2,
            (A + B + C) / 3,
            (2 * A + B + C) / 4, (A + 2 * B + C) / 4,
            (A + B + 2 * C) / 4,
            (A + B + C) / 3 + np.array([0.01, 0.01])
        ])
    
    return best_pts
