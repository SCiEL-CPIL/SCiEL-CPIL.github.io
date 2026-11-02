import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _objective(flat_points):
    pts = flat_points.reshape(N, 2)
    return -_score(pts)

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _objective_soft(flat_points, beta=50.0):
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _fourier_radial(theta, coeffs):
    r = np.zeros_like(theta)
    KH = len(coeffs) // 2
    r += coeffs[0]
    for k in range(1, KH + 1):
        r += coeffs[k] * np.cos(k * theta)
        r += coeffs[KH + k] * np.sin(k * theta)
    return r

def _fourier_to_points(coeffs, angles):
    radii = _fourier_radial(angles, coeffs)
    radii = np.maximum(radii, 0.01)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

def _fourier_score(coeffs, angles):
    pts = _fourier_to_points(coeffs, angles)
    return _score(pts)

KHARMONICS = 3

def _fourier_objective(flat):
    n_coeffs = 2 * KHARMONICS + 1
    coeffs = flat[:n_coeffs]
    angles = flat[n_coeffs:]
    angles = np.sort(angles)
    return -_fourier_score(coeffs, angles)

def _cs_params_to_points(radii, base_angles):
    """Convert centrally symmetric parameters to points.
    radii: array of shape (N//2,) giving distances from center
    base_angles: array of shape (N//2,) giving base angles
    Returns N points where point i and point i+N//2 are antipodal.
    """
    half = N // 2
    angles = base_angles
    pts_half = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    pts_anti = -pts_half
    pts = np.vstack([pts_half, pts_anti])
    return pts

def _cs_score(flat):
    half = N // 2
    radii = flat[:half]
    angles = flat[half:]
    if np.any(radii < 0.01):
        return 0.0
    pts = _cs_params_to_points(radii, angles)
    return _score(pts)

def _cs_objective(flat):
    return -_cs_score(flat)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def refine(points_init, method='Nelder-Mead', maxiter=2000, tol=1e-8):
        pts = points_init.copy()
        center = np.mean(pts, axis=0)
        pts = pts - center
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
            
        try:
            if method == 'Nelder-Mead':
                options = {'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                res = minimize(_objective, pts.flatten(), method='Nelder-Mead', options=options)
            elif method == 'Powell':
                res = minimize(_objective, pts.flatten(), method='Powell', 
                               options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
            elif method == 'L-BFGS-B':
                res = minimize(_objective, pts.flatten(), method='L-BFGS-B', 
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
            else:
                return None, -np.inf
                
            cand_pts = res.x.reshape(N, 2)
            cand_score = -res.fun
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    def refine_soft(points_init, method='L-BFGS-B', maxiter=3000, beta=50.0):
        pts = points_init.copy()
        center = np.mean(pts, axis=0)
        pts = pts - center
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
            
        try:
            if method == 'L-BFGS-B':
                res = minimize(lambda x: _objective_soft(x, beta=beta), pts.flatten(), 
                               method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            elif method == 'Powell':
                res = minimize(lambda x: _objective_soft(x, beta=beta), pts.flatten(), 
                               method='Powell', options={'maxiter': maxiter, 'xtol': 1e-11, 'ftol': 1e-13})
            else:
                res = minimize(lambda x: _objective_soft(x, beta=beta), pts.flatten(), 
                               method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    def refine_cs(flat_init, method='Nelder-Mead', maxiter=3000):
        """Refine in centrally symmetric parameter space."""
        half = N // 2
        try:
            if method == 'Nelder-Mead':
                options = {'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                res = minimize(_cs_objective, flat_init, method='Nelder-Mead', options=options)
            elif method == 'Powell':
                res = minimize(_cs_objective, flat_init, method='Powell',
                               options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9})
            else:
                return None, -np.inf
            cand_flat = res.x
            cand_score = -res.fun
            radii = cand_flat[:half]
            angles = cand_flat[half:]
            if np.any(radii < 0.01):
                return None, -np.inf
            cand_pts = _cs_params_to_points(radii, angles)
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    def coordinate_descent(points, max_rounds=20, time_limit=None):
        """Coordinate descent refinement."""
        pts = points.copy()
        center = np.mean(pts, axis=0)
        pts = pts - center
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
        
        current_score = _score(pts)
        
        for rnd in range(max_rounds):
            if time_limit is not None and time.time() > time_limit:
                break
            improved = False
            for i in range(N):
                if time_limit is not None and time.time() > time_limit:
                    break
                # Try small perturbations in x and y for point i
                for dim in range(2):
                    for step in [0.01, 0.005, 0.002, 0.001, -0.001, -0.002, -0.005, -0.01]:
                        trial = pts.copy()
                        trial[i, dim] += step
                        trial_center = np.mean(trial, axis=0)
                        trial = trial - trial_center
                        max_n = np.max(np.linalg.norm(trial, axis=1))
                        if max_n > 0:
                            trial = trial / max_n
                        s = _score(trial)
                        if s > current_score + 1e-12:
                            pts = trial
                            current_score = s
                            improved = True
                            break
            if not improved:
                break
        
        return pts, current_score

    # 1. Initializations
    initializations = []
    
    # Regular 14-gon
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    x_reg = np.column_stack((np.cos(angles_reg), np.sin(angles_reg)))
    initializations.append(x_reg)
    
    # Two concentric rings (7+7)
    for r1 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]:
        a1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        initializations.append(pts)
    
    # Three rings (5+4+5)
    for r1, r2 in [(0.3, 0.6), (0.4, 0.75), (0.5, 0.85)]:
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        a3 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)]),
            np.column_stack([np.cos(a3), np.sin(a3)])
        ])
        initializations.append(pts)
    
    # Elliptical configurations
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7), (1.1, 0.9)]:
        a = np.linspace(0, 2*np.pi, N, endpoint=False)
        initializations.append(np.column_stack([rx*np.cos(a), ry*np.sin(a)]))
    
    # Halton sequence
    def halton(index, base):
        f = 1
        r = 0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    halton_pts = []
    for i in range(1, N+1):
        x = halton(i, 2)
        y = halton(i, 3)
        halton_pts.append([x, y])
    x_halton = np.array(halton_pts)
    initializations.append(x_halton)
    
    # Random points
    rng = np.random.default_rng(123)
    for _ in range(3):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2*np.pi, N)
        x_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        initializations.append(x_rand)
    
    # Jittered regular polygon
    for js in [0.05, 0.1, 0.2]:
        initializations.append(x_reg + rng.normal(0, js, size=(N, 2)))

    # 2. Refine initializations
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
            
        cand_pts, cand_score = refine(init_pts, 'Nelder-Mead', maxiter=800)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
        
        if time.time() < deadline - 30.0 and cand_pts is not None:
            cand_pts2, cand_score2 = refine_soft(cand_pts, 'L-BFGS-B', maxiter=1500, beta=50.0)
            if cand_pts2 is not None:
                update_best(cand_pts2, cand_score2)
        
        if time.time() < deadline - 30.0 and best_points is not None:
            cand_pts3, cand_score3 = refine(best_points, 'Powell', maxiter=200)
            if cand_pts3 is not None:
                update_best(cand_pts3, cand_score3)

    # 3. Centrally symmetric parameterization optimization
    if time.time() < deadline - 35.0:
        half = N // 2
        # Initialize CS parameters from regular polygon
        base_angles_init = np.linspace(0, 2*np.pi, half, endpoint=False)
        radii_init = np.ones(half)
        flat_cs_init = np.concatenate([radii_init, base_angles_init])
        
        # Multiple CS initializations
        cs_inits = [flat_cs_init.copy()]
        
        # Perturbed angles
        rng_cs = np.random.default_rng(99)
        for _ in range(5):
            r_pert = np.ones(half) + rng_cs.normal(0, 0.1, size=half)
            r_pert = np.maximum(r_pert, 0.1)
            a_pert = base_angles_init + rng_cs.normal(0, 0.15, size=half)
            cs_inits.append(np.concatenate([r_pert, a_pert]))
        
        # Different angle spacings
        for offset in [0.1, 0.2, 0.3]:
            a_off = base_angles_init + offset
            cs_inits.append(np.concatenate([np.ones(half), a_off]))
        
        for i, flat_init in enumerate(cs_inits):
            if time.time() > deadline - 30.0:
                break
            
            cand_pts, cand_score = refine_cs(flat_init, 'Nelder-Mead', maxiter=2000)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)
            
            if time.time() < deadline - 30.0 and best_points is not None:
                cand_pts2, cand_score2 = refine_cs(flat_init, 'Powell', maxiter=1000)
                if cand_pts2 is not None:
                    update_best(cand_pts2, cand_score2)

    # 4. Fourier-based optimization
    if time.time() < deadline - 30.0:
        n_coeffs = 2 * KHARMONICS + 1
        
        fourier_inits = []
        coeffs_circle = np.zeros(n_coeffs)
        coeffs_circle[0] = 1.0
        angles_uniform = np.linspace(0, 2*np.pi, N, endpoint=False)
        flat_init = np.concatenate([coeffs_circle, angles_uniform])
        fourier_inits.append(flat_init.copy())
        
        rng_fourier = np.random.default_rng(7)
        for _ in range(3):
            coeffs_pert = np.zeros(n_coeffs)
            coeffs_pert[0] = 1.0
            coeffs_pert[1:] += rng_fourier.normal(0, 0.05, size=n_coeffs-1)
            angles_pert = angles_uniform + rng_fourier.normal(0, 0.1, size=N)
            flat_pert = np.concatenate([coeffs_pert, angles_pert])
            fourier_inits.append(flat_pert)
        
        for i, flat_init in enumerate(fourier_inits):
            if time.time() > deadline - 25.0:
                break
            
            try:
                res_fm = minimize(_fourier_objective, flat_init, method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
                if res_fm.fun < 0:
                    coeffs_opt = res_fm.x[:n_coeffs]
                    angles_opt = np.sort(res_fm.x[n_coeffs:])
                    cand_pts = _fourier_to_points(coeffs_opt, angles_opt)
                    cand_score = _score(cand_pts)
                    update_best(cand_pts, cand_score)
            except Exception:
                pass

    # 5. Differential Evolution for global exploration
    if time.time() < deadline - 30.0:
        def de_objective(flat_points):
            pts = flat_points.reshape(N, 2)
            if np.allclose(pts, pts[0]):
                return 1.0
            return -_score(pts)
            
        try:
            bounds = [(-1.5, 1.5)] * (N * 2)
            res_de = minimize(
                lambda x: de_objective(x),
                np.random.default_rng(42).uniform(-1, 1, size=N*2),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if res_de.fun < 0:
                cand_pts = res_de.x.reshape(N, 2)
                cand_score = -res_de.fun
                update_best(cand_pts, cand_score)
        except Exception:
            pass

    # 6. Coordinate descent refinement
    if best_points is not None and time.time() < deadline - 15.0:
        cd_pts, cd_score = coordinate_descent(best_points, max_rounds=15, time_limit=deadline - 10.0)
        update_best(cd_pts, cd_score)

    # 7. Final polish on the global best
    if best_points is not None and time.time() < deadline - 12.0:
        cand_pts, cand_score = refine(best_points, 'Nelder-Mead', maxiter=1000)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
        
        if time.time() < deadline - 10.0:
            cand_pts2, cand_score2 = refine_soft(best_points, 'L-BFGS-B', maxiter=2000, beta=100.0)
            if cand_pts2 is not None:
                update_best(cand_pts2, cand_score2)
        
        if time.time() < deadline - 8.0:
            cand_pts3, cand_score3 = refine(best_points, 'Powell', maxiter=500)
            if cand_pts3 is not None:
                update_best(cand_pts3, cand_score3)
        
        if time.time() < deadline - 5.0:
            cd_pts, cd_score = coordinate_descent(best_points, max_rounds=10, time_limit=deadline - 3.0)
            update_best(cd_pts, cd_score)

    # 8. Random restarts if time allows
    if best_points is not None and time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(42)
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            noise = rng2.normal(0, 0.1, size=(N, 2))
            perturbed = best_points + noise
            cand_pts, cand_score = refine(perturbed, 'Nelder-Mead', maxiter=500)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)
            if time.time() < deadline - 5.0:
                cand_pts2, cand_score2 = refine_soft(cand_pts, 'L-BFGS-B', maxiter=800, beta=50.0)
                if cand_pts2 is not None:
                    update_best(cand_pts2, cand_score2)

    if best_points is None:
        best_points = initializations[0]
        
    return best_points
