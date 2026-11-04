"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 13
N_BASIS = 7  # number of Gaussian basis functions for radial expansion

# EVOLVE-BLOCK-START

# Pre-compute triangle indices for efficiency
_idx_i, _idx_j, _idx_k = np.indices((N, N, N)).reshape(3, -1)
_mask = (_idx_i < _idx_j) & (_idx_j < _idx_k)
_TRI_I = _idx_i[_mask]
_TRI_J = _idx_j[_mask]
_TRI_K = _idx_k[_mask]
_NUM_TRI = len(_TRI_I)

def _compute_score(pts):
    """
    Compute the minimum triangle area normalized by the convex hull area.
    Returns 0 if invalid.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    # Check for duplicates or invalid points
    if np.any(~np.isfinite(pts)):
        return 0.0
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas vectorized
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _smooth_obj(params, tau):
    """
    Objective function for optimization.
    Minimizes the negative smooth minimum area ratio.
    params: (2N,) array of [x_1, y_1, ..., x_N, y_N]
    tau: sharpness parameter for logsumexp approximation of min
    """
    pts = params.reshape(N, 2)
    
    # Basic validity check
    if np.any(~np.isfinite(pts)):
        return 1e9

    # Compute Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 1e9
    except Exception:
        return 1e9

    # Compute all triangle areas using pre-computed indices
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    
    # We want to maximize min(ratios).
    # Minimize -min(ratios).
    # Approx -min(y) with (1/tau) * log(sum(exp(-tau * y)))
    
    # Numerical stability
    min_ratio = np.min(ratios)
    exponents = -tau * (ratios - min_ratio)
    exponents = np.clip(exponents, -500, 0)  # clip to prevent underflow issues
    
    sum_exp = np.sum(np.exp(exponents))
    
    if sum_exp == 0:
        return 1e9
        
    log_sum = np.log(sum_exp)
    
    val = min_ratio + (log_sum / tau)
    
    return -val

def _radial_to_points(basis_params):
    """
    Convert radial basis parameters to 2D points.
    basis_params: [r0, w1, w2, ..., wN_BASIS, theta1, theta2, ..., thetaN_BASIS]
    r0: base radius
    w_i: weights for Gaussian basis functions
    theta_i: centers for Gaussian basis functions
    """
    r0 = basis_params[0]
    weights = basis_params[1:1+N_BASIS]
    thetas = basis_params[1+N_BASIS:1+2*N_BASIS]
    
    # Angular positions for the N points
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Compute radius for each point using Gaussian basis expansion
    radii = np.zeros(N)
    for i in range(N):
        radii[i] = r0
        for j in range(N_BASIS):
            # Circular distance
            diff = angles[i] - thetas[j]
            diff = np.arccos(np.clip(np.cos(diff), -1, 1))
            # Gaussian kernel with fixed width
            width = 0.5  # radians
            radii[i] += weights[j] * np.exp(-0.5 * (diff / width) ** 2)
    
    # Convert to Cartesian
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    return np.column_stack((x, y))

def _radial_obj(basis_params, tau):
    """
    Objective for radial basis parameter optimization.
    """
    pts = _radial_to_points(basis_params)
    
    if np.any(~np.isfinite(pts)):
        return 1e9
    
    return _smooth_obj(pts, tau)

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    best_pts = None
    best_score = 0.0
    
    # 1. Radial basis optimization with differential evolution
    # Parameters: r0, w1..w7, theta1..theta7 = 1 + 7 + 7 = 15 parameters
    # Bounds:
    #   r0: [0.5, 2.0]
    #   w_i: [-0.5, 0.5]
    #   theta_i: [0, 2*pi]
    
    bounds = [(0.5, 2.0)]  # r0
    bounds += [(-0.5, 0.5)] * N_BASIS  # weights
    bounds += [(0, 2 * np.pi)] * N_BASIS  # theta centers
    
    tau_de = 50.0
    
    def de_radial_obj(params):
        return _radial_obj(params, tau_de)
    
    try:
        res_de = differential_evolution(
            de_radial_obj,
            bounds,
            maxiter=150,
            popsize=25,
            tol=1e-6,
            seed=42,
            init='sobol',
            mutation=(0.5, 1.5),
            recombination=0.8
        )
        
        radial_pts = _radial_to_points(res_de.x)
        radial_score = _compute_score(radial_pts)
        
        if radial_score > best_score:
            best_score = radial_score
            best_pts = radial_pts.copy()
            
    except Exception as e:
        print(f"DE radial failed: {e}")
    
    # 2. Full 2D differential evolution as backup
    elapsed = time.time() - start_time
    if elapsed < time_limit - 40:
        bounds_2d = [(-2.0, 2.0)] * (2 * N)
        
        def de_2d_obj(params):
            return _smooth_obj(params, 20.0)
        
        try:
            res_de2d = differential_evolution(
                de_2d_obj,
                bounds_2d,
                maxiter=80,
                popsize=20,
                tol=1e-5,
                seed=123,
                init='sobol'
            )
            
            de2d_pts = res_de2d.x.reshape(N, 2)
            de2d_score = _compute_score(de2d_pts)
            
            if de2d_score > best_score:
                best_score = de2d_score
                best_pts = de2d_pts.copy()
                
        except Exception as e:
            print(f"DE 2D failed: {e}")
    
    # 3. Local polish with L-BFGS-B on multiple starting points
    polish_starts = []
    
    if best_pts is not None:
        polish_starts.append(best_pts)
    
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    polish_starts.append(p_reg)
    
    # Add a few random starts
    rng = np.random.default_rng(456)
    for _ in range(4):
        p_rand = rng.uniform(-1, 1, (N, 2))
        polish_starts.append(p_rand)
    
    # Also add radial-optimized points from a different DE run if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        try:
            res_de2 = differential_evolution(
                de_radial_obj,
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-5,
                seed=789,
                init='sobol'
            )
            radial_pts2 = _radial_to_points(res_de2.x)
            radial_score2 = _compute_score(radial_pts2)
            if radial_score2 > best_score:
                best_score = radial_score2
                best_pts = radial_pts2.copy()
            polish_starts.append(radial_pts2)
        except Exception:
            pass
    
    tau_polish = 300.0
    
    for p0 in polish_starts:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            break
            
        def polish_obj(params):
            return _smooth_obj(params, tau_polish)
            
        try:
            res_polish = minimize(
                polish_obj,
                p0.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-10}
            )
            
            new_pts = res_polish.x.reshape(N, 2)
            new_score = _compute_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                
        except Exception:
            continue
    
    # 4. Final high-precision polish
    elapsed = time.time() - start_time
    if best_pts is not None and elapsed < time_limit - 5:
        tau_final = 1000.0
        
        def final_obj(params):
            return _smooth_obj(params, tau_final)
            
        try:
            res_final = minimize(
                final_obj,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-12}
            )
            
            new_pts = res_final.x.reshape(N, 2)
            new_score = _compute_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                
        except Exception:
            pass
    
    # 5. Worst-triangle targeted optimization
    elapsed = time.time() - start_time
    if best_pts is not None and elapsed < time_limit - 3:
        # Identify the worst (smallest) triangle and try to optimize specifically
        pts = best_pts
        p_i = pts[_TRI_I]
        p_j = pts[_TRI_J]
        p_k = pts[_TRI_K]
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Find worst triangles
        worst_indices = np.argsort(areas)[:5]
        
        # Try small perturbations around the worst triangle vertices
        for wi in worst_indices[:3]:
            i, j, k = _TRI_I[wi], _TRI_J[wi], _TRI_K[wi]
            
            # Create a modified objective that emphasizes the worst triangle
            def targeted_obj(params, wi=wi):
                pts_local = params.reshape(N, 2)
                if np.any(~np.isfinite(pts_local)):
                    return 1e9
                try:
                    hull = ConvexHull(pts_local)
                    hull_area = hull.volume
                    if hull_area < 1e-9:
                        return 1e9
                except Exception:
                    return 1e9
                
                p_i_l = pts_local[_TRI_I]
                p_j_l = pts_local[_TRI_J]
                p_k_l = pts_local[_TRI_K]
                v1_l = p_j_l - p_i_l
                v2_l = p_k_l - p_i_l
                cross_l = v1_l[:, 0] * v2_l[:, 1] - v1_l[:, 1] * v2_l[:, 0]
                areas_l = 0.5 * np.abs(cross_l)
                ratios_l = areas_l / hull_area
                
                # Emphasize the worst triangle
                min_ratio = np.min(ratios_l)
                exponents = -50.0 * (ratios_l - min_ratio)
                exponents = np.clip(exponents, -500, 0)
                sum_exp = np.sum(np.exp(exponents))
                if sum_exp == 0:
                    return 1e9
                log_sum = np.log(sum_exp)
                val = min_ratio + (log_sum / 50.0)
                return -val
            
            try:
                res_target = minimize(
                    targeted_obj,
                    best_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'gtol': 1e-10}
                )
                new_pts = res_target.x.reshape(N, 2)
                new_score = _compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            except Exception:
                pass
    
    if best_pts is None:
        return p_reg
        
    return best_pts

# EVOLVE-BLOCK-END
