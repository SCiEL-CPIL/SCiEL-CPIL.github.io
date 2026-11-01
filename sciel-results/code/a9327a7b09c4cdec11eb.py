import numpy as np
import time
from scipy.optimize import minimize, basinhopping

N = 14
D = 3

# Precompute pairwise indices for efficiency
_pair_i, _pair_j = np.triu_indices(N, k=1)
_n_pairs = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    return sq_dists[_pair_i, _pair_j]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def generate_D3d_points(params):
    """
    Generate 14 points with D3d symmetry.
    params: [x, y, z, a]
    """
    x, y, z, a = params
    
    c = 0.5
    s = np.sqrt(3.0) / 2.0
    
    # C3 orbit of (x, y, z)
    p1 = np.array([x, y, z])
    p2 = np.array([-c*x - s*y, s*x - c*y, z])
    p3 = np.array([-c*x + s*y, -s*x - c*y, z])
    
    # C2' orbit (rotation by 180 around x-axis)
    q1 = np.array([x, -y, -z])
    q2 = np.array([-c*x - s*y, -s*x + c*y, -z])
    q3 = np.array([-c*x + s*y, s*x + c*y, -z])
    
    # Inversion orbit
    r1 = -p1
    r2 = -p2
    r3 = -p3
    s1 = -q1
    s2 = -q2
    s3 = -q3
    
    # Axis points
    t1 = np.array([0.0, 0.0, a])
    t2 = np.array([0.0, 0.0, -a])
    
    points = np.array([p1, p2, p3, q1, q2, q3, r1, r2, r3, s1, s2, s3, t1, t2])
    return points

def objective_D3d(params, tau=50.0):
    """
    Objective function for D3d symmetric configuration.
    """
    points = generate_D3d_points(params)
    points = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    
    points = points / np.sqrt(mean_sq_dist)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    log_sq_dists = np.log(sq_dists)
    
    m_max = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max)) / tau
    
    m_min = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min)) / tau
    
    return soft_max - soft_min

def objective_full_space(points_flat, tau=50.0):
    """
    Objective function for full 42-dimensional coordinate space.
    points_flat: array of shape (N*D,)
    """
    points = points_flat.reshape(N, D)
    
    # Center
    points = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    
    # Normalize to unit variance (or similar) to keep numbers stable
    points = points / np.sqrt(mean_sq_dist)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    log_sq_dists = np.log(sq_dists)
    
    m_max = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max)) / tau
    
    m_min = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min)) / tau
    
    return soft_max - soft_min

def objective_epigraph(params, tau=100.0):
    """
    Epigraph formulation: minimize t subject to constraints that all pairwise
    squared distances are <= 1 and >= 1/t.
    params: [point_coords..., t] where t is the ratio (max_sq/min_sq)
    """
    n_vars = N * D + 1
    points_flat = params[:N*D]
    t = params[N*D]
    
    if t < 1e-10:
        return 1e10
    
    points = points_flat.reshape(N, D)
    points = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    
    # Normalize so that max sq dist = 1
    max_sq = np.max(sq_dists)
    if max_sq < 1e-15:
        return 1e10
    
    points = points / np.sqrt(max_sq)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Now max sq dist is 1, min sq dist is 1/ratio
    min_sq = np.min(sq_dists)
    actual_ratio = 1.0 / min_sq
    
    # Soft constraint: penalize if t < actual_ratio
    penalty = 0.0
    if t < actual_ratio:
        penalty = tau * (actual_ratio - t) ** 2
    
    return t + penalty

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        # Check for duplicates
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
            
    rng = np.random.default_rng(42)
    
    # 1. First, get the best D3d result quickly as a baseline
    inits_D3d = [
        [1.0, 0.5, 0.5, 0.5],
        [1.0, 0.2, 0.8, 0.3],
        [0.8, 0.8, 0.2, 0.6],
    ]
    
    for init in inits_D3d:
        if time.time() - start_time > timeout - 30:
            break
        pts = generate_D3d_points(init)
        update_best(pts)
        
        try:
            res = minimize(
                objective_D3d, 
                init, 
                method='L-BFGS-B', 
                args=(50.0,),
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = generate_D3d_points(res.x)
                update_best(pts)
        except Exception:
            pass
            
    # 2. Now, attempt to break symmetry and optimize in full space
    if best_points is not None:
        # Flatten the best points
        start_flat = best_points.flatten()
        
        # Define basin hopping objective
        def bh_obj_full(p_flat):
            return objective_full_space(p_flat, tau=20.0)
        
        # Basin Hopping settings
        bh_minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'options': {'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
        }
        
        try:
            bh_res = basinhopping(
                bh_obj_full,
                start_flat,
                minimizer_kwargs=bh_minimizer_kwargs,
                stepsize=0.2,
                niter=50,
                seed=42
            )
            if bh_res.x is not None:
                pts = bh_res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass
        
        # 3. Try epigraph formulation from best points
        if best_points is not None and time.time() - start_time < timeout - 20:
            pts = best_points.copy()
            sq_dists = get_pairwise_sq_dists(pts)
            max_sq = np.max(sq_dists)
            min_sq = np.min(sq_dists)
            current_ratio = max_sq / min_sq
            
            # Initialize epigraph params
            epigraph_init = np.concatenate([pts.flatten(), [current_ratio]])
            
            try:
                res_ep = minimize(
                    objective_epigraph,
                    epigraph_init,
                    method='L-BFGS-B',
                    args=(100.0,),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res_ep.x is not None:
                    pts_ep = res_ep.x[:N*D].reshape(N, D)
                    update_best(pts_ep)
            except Exception:
                pass
        
        # 4. Try a few random restarts in full space
        for i in range(8):
            if time.time() - start_time > timeout - 15:
                break
            
            # Random points on sphere with varied scales
            scale = 0.5 + 0.5 * i / 7
            rand_pts = rng.normal(size=(N, D)) * scale
            norms = np.linalg.norm(rand_pts, axis=1, keepdims=True)
            rand_pts = rand_pts / np.maximum(norms, 1e-10)
            
            start_flat = rand_pts.flatten()
            
            # Quick local optimization
            try:
                res = minimize(
                    objective_full_space,
                    start_flat,
                    method='L-BFGS-B',
                    args=(50.0,),
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    
                    # If this random start improved things, try basin hopping
                    if best_score < 1.9:
                        try:
                            bh_res = basinhopping(
                                bh_obj_full,
                                res.x,
                                minimizer_kwargs=bh_minimizer_kwargs,
                                stepsize=0.1,
                                niter=20,
                                seed=i
                            )
                            if bh_res.x is not None:
                                pts = bh_res.x.reshape(N, D)
                                update_best(pts)
                        except Exception:
                            pass
            except Exception:
                pass

    # 5. Final high-precision polish on the best found points
    if best_points is not None and time.time() - start_time < timeout - 5:
        start_flat = best_points.flatten()
        try:
            # High-precision polish with very tight tolerances
            res = minimize(
                objective_full_space,
                start_flat,
                method='L-BFGS-B',
                args=(500.0,),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass
        
        # Try epigraph polish if we have time
        if time.time() - start_time < timeout - 2:
            pts = best_points.copy()
            sq_dists = get_pairwise_sq_dists(pts)
            max_sq = np.max(sq_dists)
            min_sq = np.min(sq_dists)
            current_ratio = max_sq / min_sq
            
            epigraph_init = np.concatenate([pts.flatten(), [current_ratio]])
            
            try:
                res_ep = minimize(
                    objective_epigraph,
                    epigraph_init,
                    method='L-BFGS-B',
                    args=(200.0,),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res_ep.x is not None:
                    pts_ep = res_ep.x[:N*D].reshape(N, D)
                    update_best(pts_ep)
            except Exception:
                pass
            
    if best_points is None:
        # Fallback
        best_points = generate_D3d_points([1.0, 0.5, 0.5, 0.5])

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
