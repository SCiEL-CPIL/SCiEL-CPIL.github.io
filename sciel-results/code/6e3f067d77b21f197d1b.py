import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools
from functools import partial

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def objective_ratio_smooth(x, tau=10.0):
    """
    Smooth approximation of the ratio of max to min squared distance.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-10)
    
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    return np.log(soft_max) - np.log(soft_min)

def raw_score(points):
    """Compute the true squared distance ratio."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-10:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def get_tetrahedron_vertices(center=np.zeros(3), scale=1.0):
    """Return vertices of a regular tetrahedron centered at 'center'."""
    # Regular tetrahedron vertices (normalized to have circumradius 1)
    base = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    base = base / np.sqrt(3.0)  # normalize to circumradius 1
    return center + scale * base

def rotate_matrix(angle, axis):
    """Return 3D rotation matrix for angle around axis (x=0, y=1, z=2)."""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + s * K + (1 - c) * (K @ K)

def params_to_points(params):
    """
    Convert 9 parameters to 14 points.
    params: [scale1, scale2, rot_angle, rot_axis_x, rot_axis_y, rot_axis_z, tx, ty, tz]
    - Two tetrahedra of scales scale1, scale2
    - Second tetrahedron rotated by rot_angle around normalized axis (rot_axis_x, rot_axis_y, rot_axis_z)
    - Second tetrahedron translated by (tx, ty, tz)
    """
    scale1 = params[0]
    scale2 = params[1]
    rot_angle = params[2]
    axis = params[3:6]
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        axis = np.array([0, 0, 1])
    else:
        axis = axis / axis_norm
    translation = params[6:9]
    
    tet1 = get_tetrahedron_vertices(np.zeros(3), scale1)
    R = rotate_matrix(rot_angle, axis)
    tet2 = (R @ get_tetrahedron_vertices(np.zeros(3), scale2).T).T + translation
    
    points = np.vstack([tet1, tet2])
    return points

def objective_from_params(params, tau=10.0):
    """Objective function for differential evolution over 9 parameters."""
    points = params_to_points(params)
    # Center the points to remove translational DOF
    points = points - np.mean(points, axis=0)
    return objective_ratio_smooth(points.flatten(), tau=tau)

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    best_score = np.inf
    best_points = None
    
    # Phase 1: Differential evolution over 9 parameters (tetrahedron assembly)
    bounds = [
        (0.3, 3.5),   # scale1
        (0.3, 3.5),   # scale2
        (0.0, np.pi), # rot_angle
        (-1.0, 1.0),  # rot_axis_x
        (-1.0, 1.0),  # rot_axis_y
        (-1.0, 1.0),  # rot_axis_z
        (-2.5, 2.5),  # tx
        (-2.5, 2.5),  # ty
        (-2.5, 2.5),  # tz
    ]
    
    try:
        de_result = differential_evolution(
            objective_from_params,
            bounds=bounds,
            seed=42,
            maxiter=80,
            tol=1e-7,
            popsize=20,
            polish=True,
            init='sobol'
        )
        
        if de_result.x is not None:
            pts_de = params_to_points(de_result.x)
            pts_de = pts_de - np.mean(pts_de, axis=0)
            score_de = raw_score(pts_de)
            if score_de < best_score:
                best_score = score_de
                best_points = pts_de
    except Exception:
        pass
    
    if time.time() - start_time > timeout - 30:
        if best_points is not None:
            return best_points
        points_grid = []
        for x in range(3):
            for y in range(3):
                for z in range(2):
                    points_grid.append((x, y, z))
        to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
        final_points_grid = [p for p in points_grid if p not in to_remove]
        init_grid = np.array(final_points_grid, dtype=float)
        init_grid = init_grid - np.mean(init_grid, axis=0)
        return init_grid
    
    # Phase 2: Refine using L-BFGS-B in full coordinate space
    candidates = []
    if best_points is not None:
        candidates.append(best_points.flatten())
    
    # Add grid initialization as backup
    points_grid = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                points_grid.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_points_grid = [p for p in points_grid if p not in to_remove]
    init_grid = np.array(final_points_grid, dtype=float)
    init_grid = init_grid - np.mean(init_grid, axis=0)
    candidates.append(init_grid.flatten())
    
    # Add icosahedron-based initialization (12 vertices + 2 center-ish points)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    # Add 2 more points near center to make 14
    ico_full = np.vstack([ico_verts, [[0, 0, 0.5], [0, 0, -0.5]]])
    ico_full = ico_full - np.mean(ico_full, axis=0)
    candidates.append(ico_full.flatten())
    
    # Add a few random starts for diversity
    rng = np.random.default_rng(123)
    for _ in range(8):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
    
    tau_schedule = [5.0, 20.0, 100.0]
    maxiter_schedule = [300, 400, 400]
    
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 20:
            break
            
        x_curr = init_x.copy()
        
        try:
            for idx, tau in enumerate(tau_schedule):
                if time.time() - start_time > timeout - 20:
                    break
                
                obj = partial(objective_ratio_smooth, tau=tau)
                maxiter = maxiter_schedule[idx]
                
                res = minimize(
                    obj, 
                    x_curr, 
                    method='L-BFGS-B', 
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res.x is not None:
                    x_curr = res.x
            
            pts = x_curr.reshape(N, D)
            score = raw_score(pts)
            if score < best_score:
                best_score = score
                best_points = pts
                
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish with very high tau
    if best_points is not None and time.time() - start_time < timeout - 5:
        try:
            x_curr = best_points.flatten()
            obj_final = partial(objective_ratio_smooth, tau=3000.0)
            res_final = minimize(
                obj_final,
                x_curr,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res_final.x is not None:
                pts_final = res_final.x.reshape(N, D)
                score_final = raw_score(pts_final)
                if score_final < best_score:
                    best_score = score_final
                    best_points = pts_final
        except Exception:
            pass
    
    # Phase 4: Ultra-precise final polish
    if best_points is not None and time.time() - start_time < timeout - 1:
        try:
            x_curr = best_points.flatten()
            obj_ultra = partial(objective_ratio_smooth, tau=10000.0)
            res_ultra = minimize(
                obj_ultra,
                x_curr,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res_ultra.x is not None:
                pts_ultra = res_ultra.x.reshape(N, D)
                score_ultra = raw_score(pts_ultra)
                if score_ultra < best_score:
                    best_score = score_ultra
                    best_points = pts_ultra
        except Exception:
            pass
    
    if best_points is None:
        best_points = init_grid
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
