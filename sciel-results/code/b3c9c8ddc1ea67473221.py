"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute convex hull area."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _regular_polygon(n, radius=1.0):
    """Regular n-gon centered at origin."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _calculate_score(points):
    """Compute the raw score: min triangle area / convex hull area."""
    return _score(points)

def _all_triangle_areas(points):
    """Compute all triangle areas as a flat array."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
    return np.array(areas)

def _soft_min_score(points, beta):
    """
    Soft-min approximation of min triangle area using log-sum-exp.
    As beta -> inf, this approaches the true minimum.
    Returns the soft-min value (positive).
    """
    areas = _all_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    # Use log-sum-exp for numerical stability
    # soft_min = -(1/beta) * log(sum(exp(-beta * a_i)))
    # = min(a_i) - (1/beta)*log(sum(exp(-beta*(a_i - min_a))))
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = min_a - (1.0/beta) * log_sum
    return soft_min_val

def _soft_min_score_normalized(points, beta):
    """Soft-min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    sm = _soft_min_score(points, beta)
    return sm / hull_area

def _generate_jittered_polygon(theta, seed=42):
    """
    Generate a jittered regular 14-gon.
    theta = [scale, rotation, shear_x, shear_y, jitter_amplitude, jitter_phase]
    The jitter is a fixed low-dimensional Gaussian field applied radially.
    """
    scale, rot, shear_x, shear_y, jitter_amp, jitter_phase = theta
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Base regular polygon
    base_r = 1.0
    base_x = base_r * np.cos(angles)
    base_y = base_r * np.sin(angles)
    
    # Radial jitter from a fixed low-dim Gaussian field
    # Use a small number of Fourier modes to create smooth radial perturbation
    rng = np.random.RandomState(seed)
    # Pre-compute jitter pattern (fixed, depends only on seed)
    # Use 3 Fourier modes for the radial perturbation
    n_modes = 3
    # Fixed random coefficients for the Fourier modes
    # These are deterministic given the seed
    fourier_coeffs = []
    for m in range(1, n_modes+1):
        fourier_coeffs.append(rng.randn())
    fourier_coeffs = np.array(fourier_coeffs)
    
    # Radial jitter: sum of cos(m*angle + phase_m) * coeff_m
    radial_jitter = np.zeros(N)
    for m in range(1, n_modes+1):
        radial_jitter += fourier_coeffs[m-1] * np.cos(m * angles + jitter_phase * m)
    
    # Apply jitter amplitude
    r = base_r + jitter_amp * radial_jitter
    
    # Apply rotation
    cos_r = np.cos(rot)
    sin_r = np.sin(rot)
    x = r * (cos_r * np.cos(angles) - sin_r * np.sin(angles))
    y = r * (sin_r * np.cos(angles) + cos_r * np.sin(angles))
    
    # Apply shear
    x = x + shear_x * y
    y = y + shear_y * x  # Note: use original x for shear coupling
    
    # Actually, let's do proper affine: [x'; y'] = [[1, sx], [sy, 1]] * [x; y]
    # Redo:
    x_base = r * (cos_r * np.cos(angles) - sin_r * np.sin(angles))
    y_base = r * (sin_r * np.cos(angles) + cos_r * np.sin(angles))
    x_out = x_base + shear_x * y_base
    y_out = y_base + shear_y * x_base
    
    # Scale
    x_out = scale * x_out
    y_out = scale * y_out
    
    return np.column_stack([x_out, y_out])

def _jittered_polygon_score(theta, beta, seed=42):
    """Score of jittered polygon with given parameters."""
    points = _generate_jittered_polygon(theta, seed)
    return _soft_min_score_normalized(points, beta)

def _optimize_jittered_polygon(seed, deadline):
    """
    Optimize the jittered polygon parameters using L-BFGS-B with beta annealing.
    """
    rng = np.random.RandomState(seed)
    
    # Initial parameters: [scale, rotation, shear_x, shear_y, jitter_amp, jitter_phase]
    # Start near regular polygon
    best_score = 0.0
    best_theta = None
    
    # Try several initial conditions
    n_starts = 8
    for start_i in range(n_starts):
        if time.time() > deadline - 5:
            break
        
        # Initialize parameters
        scale0 = 1.0 + rng.uniform(-0.1, 0.1)
        rot0 = rng.uniform(0, 2*np.pi)
        sx0 = rng.uniform(-0.2, 0.2)
        sy0 = rng.uniform(-0.2, 0.2)
        ja0 = rng.uniform(0.01, 0.15)
        jp0 = rng.uniform(0, 2*np.pi)
        
        theta0 = np.array([scale0, rot0, sx0, sy0, ja0, jp0])
        
        # Beta annealing schedule
        betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        
        current_theta = theta0.copy()
        
        for beta in betas:
            if time.time() > deadline - 3:
                break
            
            def objective(x):
                if time.time() > deadline:
                    return 0.0
                pts = _generate_jittered_polygon(x, seed)
                sm = _soft_min_score_normalized(pts, beta)
                return -sm
            
            try:
                res = minimize(objective, current_theta, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                if time.time() > deadline - 1:
                    break
                current_theta = res.x
            except Exception:
                pass
        
        # Evaluate final score with true min
        final_pts = _generate_jittered_polygon(current_theta, seed)
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_theta = current_theta.copy()
    
    if best_theta is None:
        return None, 0.0
    
    return best_theta, best_score

def _coordinate_descent_polish(points, deadline, max_iter=200):
    """
    Coordinate descent polish: move one point at a time to improve min triangle area.
    """
    pts = points.copy()
    n = len(pts)
    min_area = _min_triangle_area(pts)
    hull_area = _convex_hull_area(pts)
    current_score = min_area / hull_area if hull_area > 1e-10 else 0.0
    
    for iteration in range(max_iter):
        if time.time() > deadline - 2:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 1:
                break
            
            # Try moving point i in small steps
            original = pts[i].copy()
            best_local = original.copy()
            best_local_score = current_score
            
            # Try small perturbations in x and y
            for dx in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                for dy in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                    if dx == 0 and dy == 0:
                        continue
                    pts[i] = original + np.array([dx, dy])
                    sc = _score(pts)
                    if sc > best_local_score + 1e-12:
                        best_local_score = sc
                        best_local = pts[i].copy()
            
            if np.any(best_local != original):
                pts[i] = best_local
                current_score = best_local_score
                improved = True
        
        if not improved:
            break
    
    return pts, current_score

def _polish_points(points, deadline):
    """Polish points using coordinate descent."""
    return _coordinate_descent_polish(points, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    
    # Phase 1: Optimize jittered polygon model with multiple seeds
    for seed in [42, 123, 456, 789, 1000, 7, 13, 23]:
        if time.time() > deadline - 10:
            break
        
        theta, sc = _optimize_jittered_polygon(seed, deadline)
        if theta is not None:
            pts = _generate_jittered_polygon(theta, seed)
            # Verify with true score
            true_sc = _score(pts)
            if true_sc > best_score:
                best_score = true_sc
                best_pts = pts.copy()
    
    # Phase 2: Coordinate descent polish on best result
    if best_pts is not None and time.time() < deadline - 5:
        polished_pts, polished_score = _coordinate_descent_polish(best_pts, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts
    
    # Phase 3: Try a few free-form Nelder-Mead from best as starting point
    if best_pts is not None and time.time() < deadline - 15:
        def objective(x):
            if time.time() > deadline:
                return 0.0
            pts = x.reshape(N, 2)
            return -_score(pts)
        
        x0 = best_pts.flatten()
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True, 'initial_simplex_size': 0.05})
            if time.time() < deadline:
                refined_pts = res.x.reshape(N, 2)
                refined_score = _score(refined_pts)
                if refined_score > best_score:
                    best_score = refined_score
                    best_pts = refined_pts
        except Exception:
            pass
    
    # Phase 4: Final coordinate descent polish
    if best_pts is not None and time.time() < deadline - 3:
        polished_pts, polished_score = _coordinate_descent_polish(best_pts, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts
    
    if best_pts is None:
        # Fallback: regular polygon
        best_pts = _regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
