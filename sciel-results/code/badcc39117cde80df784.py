"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 140  # safety margin from 150s limit


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                area = abs(cross) * 0.5
                if area < min_a:
                    min_a = area
    return min_a


def _is_inside_triangle(p):
    """Check if point p is inside or on the equilateral triangle (0,0),(1,0),(0.5,sqrt(3)/2)."""
    x, y = p
    c = y / S3H
    b = x - 0.5 * c
    a = 1.0 - b - c
    if a < -1e-9 or b < -1e-9 or c < -1e-9:
        return False
    if a > 1 + 1e-9 or b > 1 + 1e-9 or c > 1 + 1e-9:
        return False
    return True


def _all_inside(pts):
    """Check if all points are inside the triangle."""
    for p in pts:
        if not _is_inside_triangle(p):
            return False
    return True


def _penalized_min_area(flat_params):
    """Objective: minimize negative of min area, with penalty for points outside triangle."""
    points = flat_params.reshape(-1, 2)
    # Penalty for being outside the triangle
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += abs(y) * 100
        if y > S3H:
            penalty += (y - S3H) * 100
        if x < 0:
            penalty += abs(x) * 100
        if x > 1:
            penalty += (x - 1) * 100
        # Left edge: y <= sqrt(3)*x  =>  y - sqrt(3)*x <= 0
        left_viol = y - S3 * x
        if left_viol > 0:
            penalty += left_viol * 100
        # Right edge: y <= sqrt(3)*(1-x)  =>  y + sqrt(3)*x - sqrt(3) <= 0
        right_viol = y + S3 * x - S3
        if right_viol > 0:
            penalty += right_viol * 100
    ma = _min_triangle_area(points)
    return -ma + penalty


def _random_point_in_triangle(rng):
    """Generate a random point inside the triangle using barycentric coordinates."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    # Barycentric: a=1-u-v, b=u, c=v
    a = 1 - u - v
    b = u
    c = v
    x = a * 0 + b * 1 + c * 0.5
    y = a * 0 + b * 0 + c * S3H
    return np.array([x, y])


def _project_to_triangle(pts):
    """Project points back into the triangle if they are outside."""
    for i in range(len(pts)):
        x, y = pts[i]
        # Clip y
        y = max(0, min(S3H, y))
        # Compute valid x range for this y
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    # Initial random configuration
    pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
    area = _min_triangle_area(pts)
    best_pts = pts.copy()
    best_area = area
    
    # Phase 1: Differential evolution for global search
    elapsed = time.time() - start_time
    if elapsed < 60:
        try:
            # Bounds: x in [0,1], y in [0, S3H]
            bounds = [(0, 1), (0, S3H)] * N
            result = differential_evolution(
                _penalized_min_area,
                bounds,
                maxiter=50,
                popsize=20,
                seed=42,
                tol=1e-6,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            if result.fun < 0:  # negative means valid improvement
                opt_pts = result.x.reshape(-1, 2)
                if _all_inside(opt_pts):
                    area = _min_triangle_area(opt_pts)
                    if area > best_area:
                        best_area = area
                        best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Local optimization from multiple random starts
    elapsed = time.time() - start_time
    if elapsed < 100:
        n_starts = 30
        for s in range(n_starts):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            # Generate random starting configuration
            pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
            
            # Optimize with L-BFGS-B
            try:
                result = minimize(
                    _penalized_min_area,
                    pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
            
            opt_pts = result.x.reshape(-1, 2)
            
            # Check validity
            if not _all_inside(opt_pts):
                continue
            
            area = _min_triangle_area(opt_pts)
            if area > best_area:
                best_area = area
                best_pts = opt_pts.copy()
    
    # Phase 3: Polish the best configuration with Nelder-Mead
    elapsed = time.time() - start_time
    if best_pts is not None and elapsed < DEADLINE - 5:
        try:
            result = minimize(
                _penalized_min_area,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < 0:
                opt_pts = result.x.reshape(-1, 2)
                if _all_inside(opt_pts):
                    area = _min_triangle_area(opt_pts)
                    if area > best_area:
                        best_area = area
                        best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Phase 4: Perturb and re-optimize the best configuration
    elapsed = time.time() - start_time
    if best_pts is not None and elapsed < DEADLINE - 5:
        for s in range(10):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Perturb the best configuration
            perturbed = best_pts + rng.normal(0, 0.03, size=(N, 2))
            perturbed = _project_to_triangle(perturbed)
            
            try:
                result = minimize(
                    _penalized_min_area,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
            
            opt_pts = result.x.reshape(-1, 2)
            
            if not _all_inside(opt_pts):
                continue
            
            area = _min_triangle_area(opt_pts)
            if area > best_area:
                best_area = area
                best_pts = opt_pts.copy()
    
    if best_pts is None:
        # Fallback: return random points
        return np.array([_random_point_in_triangle(rng) for _ in range(N)])
    
    # Final projection to ensure validity
    best_pts = _project_to_triangle(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
