import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
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

def raw_score(points):
    """Compute the true squared distance ratio."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-10:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

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

def get_tetrahedron_vertices(center=np.zeros(3), scale=1.0):
    """Return vertices of a regular tetrahedron centered at 'center'."""
    base = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    base = base / np.sqrt(3.0)
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
    points = points - np.mean(points, axis=0)
    return objective_ratio_smooth(points.flatten(), tau=tau)

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
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
            maxiter=50,
            tol=1e-6,
            popsize=15,
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
    
    if time.time() - start_time > timeout:
        if best_points is not None:
            return best_points
        # Fallback
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
    
    # Add Fibonacci sphere initialization
    golden_angle = np.pi * (3. - np.sqrt(5.))
    X_fib = np.zeros((N, 3))
    for i in range(N):
        z = 1. - (i / (N - 1.)) * 2.
        radius = np.sqrt(max(0, 1 - z**2))
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        X_fib[i] = [x, y, z]
    candidates.append(X_fib.flatten())
    
    # Add a few random starts for diversity
    rng = np.random.default_rng(123)
    for _ in range(10):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
    
    # Extended tau schedule for better approximation of nonsmooth max/min
    tau_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0]
    maxiter_schedule = [300, 400, 500, 500, 500]
    
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout:
            break
            
        x_curr = init_x.copy()
        
        try:
            for idx, tau in enumerate(tau_schedule):
                if time.time() - start_time > timeout:
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
    
    # Phase 3: Final high-precision polish with SLSQP
    if best_points is not None and time.time() - start_time < timeout - 5:
        try:
            x_curr = best_points.flatten()
            obj_final = partial(objective_ratio_smooth, tau=5000.0)
            res_final = minimize(
                obj_final,
                x_curr,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_final.x is not None:
                pts_final = res_final.x.reshape(N, D)
                score_final = raw_score(pts_final)
                if score_final < best_score:
                    best_score = score_final
                    best_points = pts_final
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
