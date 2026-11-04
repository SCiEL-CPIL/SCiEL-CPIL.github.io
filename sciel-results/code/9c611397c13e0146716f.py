"""Heilbronn convex regions n=14 - Fourier-radius parameterization with coordinate descent polish."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar, minimize

def _score(pts):
    """Compute the score: min triangle area / convex hull area."""
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-12:
        return 0.0
    
    n = len(pts)
    min_area = float('inf')
    
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _compute_all_areas(pts):
    """Compute all C(n,3) triangle areas vectorized. Returns array of shape (C(n,3),)."""
    n = len(pts)
    areas = []
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas.append(0.5 * crosses)
    return np.concatenate(areas)

def _fast_score(pts):
    """Fast score computation: min triangle area / hull area."""
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_area = float('inf')
    n = len(pts)
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area / hull_area

def _soft_min_score(pts, temperature=0.05):
    """Soft minimum of triangle areas divided by hull area. More differentiable."""
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 1e9
    n = len(pts)
    log_sum = 0.0
    count = 0
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            # Soft min: -T * log(sum(exp(-area/T)))
            exp_vals = np.exp(-areas / temperature)
            log_sum += np.log(np.sum(exp_vals))
            count += len(areas)
    soft_min = -temperature * log_sum / count
    return soft_min / hull_area

def _fourier_points(radii, angles=None):
    """
    Generate 14 points using Fourier-radius parameterization.
    radii: array of 14 radial distances from center.
    If angles is None, use evenly spaced angles.
    """
    n = len(radii)
    if angles is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _fourier_from_coeffs(coeffs):
    """
    Generate 14 points from Fourier coefficients.
    coeffs: array of shape (n_modes, 2) where n_modes includes DC term.
    radius(theta) = coeffs[0,0] + sum_k [coeffs[k,0]*cos(k*theta) + coeffs[k,1]*sin(k*theta)]
    """
    n = 14
    thetas = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = np.zeros(n)
    for k in range(len(coeffs)):
        if k == 0:
            radii += coeffs[k, 0]  # DC term (mean radius)
        else:
            radii += coeffs[k, 0] * np.cos(k * thetas) + coeffs[k, 1] * np.sin(k * thetas)
    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    return _fourier_points(radii, thetas)

def _optimize_fourier(deadline):
    """
    Optimize Fourier-radius parameterization.
    Uses Nelder-Mead on the Fourier coefficients.
    """
    # Start with a reasonable initial configuration
    # Two-ring: 7 at r=1.0, 7 at r=0.6, with angular offset
    # In Fourier terms, this is approximately:
    # DC = 0.8, mode 7 has amplitude ~0.2
    
    n_modes = 8  # DC + modes 1-7
    # Initial guess: mostly DC with some mode-7
    best_coeffs = np.zeros((n_modes, 2))
    best_coeffs[0, 0] = 0.8  # mean radius
    best_coeffs[7, 0] = 0.2  # mode 7 cosine
    best_coeffs[7, 1] = 0.0  # mode 7 sine
    
    best_pts = _fourier_from_coeffs(best_coeffs)
    best_score = _fast_score(best_pts)
    
    def neg_score(coeffs_flat):
        coeffs = coeffs_flat.reshape(n_modes, 2)
        # Ensure DC is positive and large enough
        if coeffs[0, 0] < 0.3:
            return 1e9
        # Clamp radii to be positive
        pts = _fourier_from_coeffs(coeffs)
        return -_fast_score(pts)
    
    # Multi-start Nelder-Mead
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            x0 = best_coeffs.flatten()
        else:
            # Perturb around best
            perturbation = rng.normal(0, 0.05, size=(n_modes, 2))
            perturbation[0, 0] = 0  # Don't perturb DC too much
            x0 = best_coeffs.flatten() + perturbation.flatten()
        
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200, 'adaptive': True})
            if res.fun < -best_score:
                best_score = -res.fun
                best_coeffs = res.x.reshape(n_modes, 2)
                best_pts = _fourier_from_coeffs(best_coeffs)
        except Exception:
            pass
    
    # Final polish with coordinate descent on individual point positions
    best_pts, best_score = _coordinate_descent_polish(best_pts, deadline)
    
    return best_pts, best_score

def _coordinate_descent_polish(pts, deadline):
    """
    Coordinate descent polish: move each point slightly in x and y to improve score.
    Uses the fast score function.
    """
    n = len(pts)
    current_score = _fast_score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # Try small perturbations in each coordinate
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    for step in step_sizes:
        if time.time() > deadline - 5.0:
            break
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                for coord in [0, 1]:
                    # Try positive and negative moves
                    for direction in [1, -1]:
                        new_pts = best_pts.copy()
                        new_pts[i, coord] += direction * step
                        new_score = _fast_score(new_pts)
                        if new_score > best_score:
                            best_score = new_score
                            best_pts = new_pts
                            improved = True
    
    return best_pts, best_score

def _generate_two_ring(r, theta_offset):
    """Generate 14 points: 7 on unit circle, 7 on circle of radius r with angular offset."""
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta_offset
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _optimize_two_ring(deadline):
    """Optimize two-ring configuration."""
    best_score = -float('inf')
    best_r = 0.6
    best_theta = np.pi / 7
    
    # Coarse grid over theta
    theta_coarse = np.linspace(0, np.pi/7, 10)
    
    for th in theta_coarse:
        if time.time() > deadline - 30.0:
            break
        
        def neg_score_r(r):
            if r < 0.1 or r > 0.99:
                return 1e9
            return -_fast_score(_generate_two_ring(r, th))
        
        try:
            res_r = minimize_scalar(neg_score_r, bounds=(0.2, 0.95), method='bounded', options={'xatol': 1e-5})
            r_opt = res_r.x
            score_opt = -res_r.fun
        except Exception:
            continue
        
        if score_opt > best_score:
            best_score = score_opt
            best_r = r_opt
            best_theta = th
    
    # Polish
    pts = _generate_two_ring(best_r, best_theta)
    pts, score = _coordinate_descent_polish(pts, deadline)
    
    if score > best_score:
        best_score = score
    
    return pts, best_score

def _generate_single_ring(radii):
    """Generate 14 points on a single ring with varying radii."""
    angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _optimize_single_ring(deadline):
    """
    Optimize single-ring with varying radii.
    This allows asymmetric radial distortions.
    """
    # Start with uniform radius
    best_radii = np.ones(14) * 0.8
    best_score = _fast_score(_generate_single_ring(best_radii))
    
    # Coordinate descent on radii
    step = 0.05
    for _ in range(50):
        if time.time() > deadline - 10.0:
            break
        improved = False
        for i in range(14):
            if time.time() > deadline - 10.0:
                break
            for direction in [1, -1]:
                new_radii = best_radii.copy()
                new_radii[i] += direction * step
                if np.any(new_radii < 0.1) or np.any(new_radii > 1.5):
                    continue
                new_pts = _generate_single_ring(new_radii)
                new_score = _fast_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_radii = new_radii
                    improved = True
        if not improved:
            step *= 0.7
            if step < 0.001:
                break
    
    best_pts = _generate_single_ring(best_radii)
    best_pts, best_score = _coordinate_descent_polish(best_pts, deadline)
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Strategy:
    # 1. Run Fourier optimization (new structural family)
    # 2. Run two-ring optimization with polish
    # 3. Run single-ring optimization
    # 4. Return the best
    
    best_pts = None
    best_score = -float('inf')
    
    # Fourier optimization
    pts_fourier, score_fourier = _optimize_fourier(deadline)
    if score_fourier > best_score:
        best_score = score_fourier
        best_pts = pts_fourier
    
    # Two-ring optimization
    if time.time() < deadline - 15.0:
        pts_2ring, score_2ring = _optimize_two_ring(deadline)
        if score_2ring > best_score:
            best_score = score_2ring
            best_pts = pts_2ring
    
    # Single-ring optimization
    if time.time() < deadline - 10.0:
        pts_1ring, score_1ring = _optimize_single_ring(deadline)
        if score_1ring > best_score:
            best_score = score_1ring
            best_pts = pts_1ring
    
    # Final verification
    if best_pts is not None:
        final_score = _score(best_pts)
        if final_score < best_score * 0.99:
            # Something went wrong; fall back to a known good configuration
            best_pts = _generate_two_ring(0.6, np.pi/7)
            final_score = _score(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
