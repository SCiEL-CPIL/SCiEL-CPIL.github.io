"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 135  # seconds, safety margin from 150s limit


def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the unit equilateral triangle area."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Compute all pairwise differences
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    # Cross product for all triples (i, j, k)
    # area(i,j,k) = 0.5 * |dx(i,j)*dy(i,k) - dy(i,j)*dx(i,k)|
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    # Extract i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    
    min_area = np.min(valid_areas)
    return min_area / TRI_AREA


def _is_inside_triangle(p):
    """Check if point p is inside the equilateral triangle with small margin."""
    x, y = p
    eps = 1e-9
    if y < -eps or y > S3H + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    if y > S3H * (1 - x) + eps:
        return False
    if y > S3H * x + eps:
        return False
    return True


def _penalized_min_area(x):
    """Objective: minimize negative min area plus penalty for out-of-bounds."""
    points = x.reshape(-1, 2)
    
    # Penalty for points outside triangle
    penalty = 0.0
    for i in range(N):
        px, py = points[i]
        d_bottom = -py
        d_top = py - S3H
        d_left = -px
        d_right = px - 1
        d_slant_right = py - S3H * (1 - px)
        d_slant_left = py - S3H * px
        
        dist = max(0.0, d_bottom, d_top, d_left, d_right, d_slant_right, d_slant_left)
        penalty += dist ** 2
    
    min_a = _min_area_normalized(points)
    return -min_a + 100.0 * penalty


def _local_polish(points, maxiter=5000):
    """Nelder-Mead polish on the point configuration."""
    x0 = np.array(points, dtype=float).flatten()
    try:
        res = minimize(
            _penalized_min_area,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        polished = res.x.reshape(-1, 2)
        valid = all(_is_inside_triangle(p) for p in polished)
        if valid:
            return polished
    except Exception:
        pass
    return None


def construct():
    start_time = time.time()
    
    # Define bounds for differential evolution
    # Points must be inside the equilateral triangle
    # We use a box that contains the triangle, then rely on penalty
    bounds = [(0.0, 1.0), (0.0, S3H)] * N
    
    best_points = None
    best_score = -float('inf')
    
    # Phase 1: Differential Evolution for global search
    # This is a structural change from pure Nelder-Mead restarts
    def elapsed():
        return time.time() - start_time
    
    # Try differential evolution with limited iterations
    if elapsed() < DEADLINE - 60:
        try:
            def de_callback(xk, convergence):
                if elapsed() > DEADLINE - 30:
                    raise TimeoutError()
                return False
            
            res = differential_evolution(
                _penalized_min_area,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                atol=1e-12,
                mutation=(0.3, 1.0),
                recombination=0.7,
                seed=42,
                callback=de_callback
            )
            
            if elapsed() < DEADLINE - 10:
                pts = res.x.reshape(-1, 2)
                valid = all(_is_inside_triangle(p) for p in pts)
                if valid:
                    score = _min_area_normalized(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
        except (TimeoutError, Exception):
            pass
    
    # Phase 2: Multiple Nelder-Mead restarts from diverse initial points
    rng = np.random.RandomState(123)
    
    def random_point_in_triangle():
        while True:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = S3H * v
                if 0.005 < x < 0.995 and 0.005 < y < S3H * 0.995:
                    if y < S3H * (1 - x) * 0.995 and y < S3H * x * 0.995:
                        return [x, y]
    
    n_restarts = 20
    
    for r in range(n_restarts):
        if elapsed() > DEADLINE - 20:
            break
        
        # Generate diverse initial configuration
        if best_points is not None and r % 3 == 0:
            # Perturb the best found so far
            x0 = best_points.flatten() + rng.randn(N * 2) * 0.05
            # Clip to bounds
            for i in range(N):
                x0[2*i] = np.clip(x0[2*i], 0.01, 0.99)
                x0[2*i+1] = np.clip(x0[2*i+1], 0.01, S3H * 0.99)
        else:
            x0 = np.array([random_point_in_triangle() for _ in range(N)]).flatten()
        
        # Run Nelder-Mead
        try:
            res = minimize(
                _penalized_min_area,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
        except Exception:
            continue
        
        if elapsed() > DEADLINE - 10:
            break
            
        pts = res.x.reshape(-1, 2)
        
        # Check validity and score
        valid = True
        for p in pts:
            if not _is_inside_triangle(p):
                valid = False
                break
        
        if valid:
            score = _min_area_normalized(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
    
    if best_points is None:
        # Fallback: return a valid random configuration
        best_points = np.array([random_point_in_triangle() for _ in range(N)])
        best_score = _min_area_normalized(best_points)
    
    # Phase 3: Final polish on the best candidate if time permits
    if elapsed() < DEADLINE - 2:
        polished = _local_polish(best_points, maxiter=10000)
        if polished is not None:
            score = _min_area_normalized(polished)
            if score > best_score:
                best_points = polished
                best_score = score
    
    return best_points
# EVOLVE-BLOCK-END
