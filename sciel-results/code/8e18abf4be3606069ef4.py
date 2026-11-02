"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 140  # safety margin from 150s limit

# Precompute all triplets for N=11 points (indices into the 11 points)
_TRIPLETS = list(itertools.combinations(range(N), 3))
_N_TRIPLETS = len(_TRIPLETS)  # 165

# Precompute index arrays for vectorized area computation
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                area = abs(cross) * 0.5
                if area < min_a:
                    min_a = area
    return min_a


def _vectorized_min_area(pts):
    """Vectorized computation of min triangle area using precomputed triplets."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    # cross = (pj - pi) x (pk - pi)
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _soft_min_area(pts, beta=50.0):
    """Soft minimum of triangle areas using log-sum-exp (differentiable).
    Returns a smooth approximation to the minimum area."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # Use log-sum-exp trick for numerical stability
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _penalized_soft_min_area(flat_params, beta=50.0):
    """Objective: minimize negative soft-min area, with penalty for points outside triangle."""
    points = flat_params.reshape(-1, 2)
    
    # Penalty for being outside the triangle
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += abs(y) * 1000
        if y > S3H:
            penalty += (y - S3H) * 1000
        if x < 0:
            penalty += abs(x) * 1000
        if x > 1:
            penalty += (x - 1) * 1000
        # Left edge: y <= sqrt(3)*x
        left_viol = y - S3 * x
        if left_viol > 0:
            penalty += left_viol * 1000
        # Right edge: y <= sqrt(3)*(1-x)
        right_viol = y + S3 * x - S3
        if right_viol > 0:
            penalty += right_viol * 1000
    
    ma = _soft_min_area(points, beta)
    return -ma + penalty


def _penalized_hard_min_area(flat_params):
    """Objective: minimize negative hard-min area, with penalty for points outside triangle."""
    points = flat_params.reshape(-1, 2)
    
    # Penalty for being outside the triangle
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += abs(y) * 1000
        if y > S3H:
            penalty += (y - S3H) * 1000
        if x < 0:
            penalty += abs(x) * 1000
        if x > 1:
            penalty += (x - 1) * 1000
        left_viol = y - S3 * x
        if left_viol > 0:
            penalty += left_viol * 1000
        right_viol = y + S3 * x - S3
        if right_viol > 0:
            penalty += right_viol * 1000
    
    ma = _vectorized_min_area(points)
    return -ma + penalty


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


def _random_point_in_triangle(rng):
    """Generate a random point inside the triangle using barycentric coordinates."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
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
        y = max(0.0, min(S3H, y))
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
    area = _vectorized_min_area(pts)
    best_pts = pts.copy()
    best_area = area
    
    # Phase 1: Differential evolution for global search
    elapsed = time.time() - start_time
    if elapsed < 50:
        try:
            bounds = [(0, 1), (0, S3H)] * N
            result = differential_evolution(
                _penalized_hard_min_area,
                bounds,
                maxiter=40,
                popsize=15,
                seed=42,
                tol=1e-5,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            if result.fun < 0:
                opt_pts = result.x.reshape(-1, 2)
                if _all_inside(opt_pts):
                    area = _vectorized_min_area(opt_pts)
                    if area > best_area:
                        best_area = area
                        best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Soft-min gradient optimization (smooth objective enables L-BFGS-B)
    elapsed = time.time() - start_time
    if elapsed < 90:
        # Use soft-min with annealed beta for smooth optimization
        n_starts = 20
        for s in range(n_starts):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            # Generate random starting configuration
            pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
            
            # Multi-scale soft-min optimization: start with low beta (smooth), increase
            try:
                for beta in [10.0, 30.0, 80.0, 200.0]:
                    result = minimize(
                        lambda x: _penalized_soft_min_area(x, beta),
                        pts.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9}
                    )
                    pts = result.x.reshape(-1, 2)
                    # Project to ensure validity
                    pts = _project_to_triangle(pts)
            except Exception:
                continue
            
            if _all_inside(pts):
                area = _vectorized_min_area(pts)
                if area > best_area:
                    best_area = area
                    best_pts = pts.copy()
    
    # Phase 3: Local optimization with hard min from multiple starts
    elapsed = time.time() - start_time
    if elapsed < 120:
        n_starts = 15
        for s in range(n_starts):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Start from best or random
            if s % 2 == 0 and best_pts is not None:
                pts = best_pts + rng.normal(0, 0.02, size=(N, 2))
                pts = _project_to_triangle(pts)
            else:
                pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
            
            try:
                result = minimize(
                    _penalized_hard_min_area,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
            except Exception:
                continue
            
            opt_pts = result.x.reshape(-1, 2)
            
            if not _all_inside(opt_pts):
                continue
            
            area = _vectorized_min_area(opt_pts)
            if area > best_area:
                best_area = area
                best_pts = opt_pts.copy()
    
    # Phase 4: Polish the best configuration with high-precision Nelder-Mead
    elapsed = time.time() - start_time
    if best_pts is not None and elapsed < DEADLINE - 5:
        try:
            result = minimize(
                _penalized_hard_min_area,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < 0:
                opt_pts = result.x.reshape(-1, 2)
                if _all_inside(opt_pts):
                    area = _vectorized_min_area(opt_pts)
                    if area > best_area:
                        best_area = area
                        best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Phase 5: Perturb and re-optimize the best configuration
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
                    _penalized_hard_min_area,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
            
            opt_pts = result.x.reshape(-1, 2)
            
            if not _all_inside(opt_pts):
                continue
            
            area = _vectorized_min_area(opt_pts)
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
