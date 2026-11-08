import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14
TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))
I_IDX = TRI_INDICES[:, 0]
J_IDX = TRI_INDICES[:, 1]
K_IDX = TRI_INDICES[:, 2]

def _score(pts):
    """Compute min triangle area / convex hull area."""
    if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
        return -1e10
    try:
        hull = ConvexHull(pts)
        h_area = hull.volume
    except Exception:
        return -1e10
    if h_area < 1e-8:
        return -1e10
    pi = pts[I_IDX]
    pj = pts[J_IDX]
    pk = pts[K_IDX]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    min_a = np.min(np.abs(cross)) * 0.5
    return min_a / h_area

def _polar_to_cart(params):
    """Convert polar params (n_inner, n_outer angles, r_in, r_out) to Cartesian.
    
    params layout: [r_in, r_out, angles_in_0..angles_in_{n_in-1}, angles_out_0..angles_out_{n_out-1}]
    """
    n_in = params[2]  # stored as int in metadata, but for optimization we fix counts
    # We'll handle this differently - see _polar_objective
    return None  # placeholder

def _make_polar_objective(n_in, n_out):
    """Create objective function for polar parameterization.
    
    params: [r_in, r_out, theta_in_0, ..., theta_in_{n_in-1}, theta_out_0, ..., theta_out_{n_out-1}]
    Total: 2 + n_in + n_out = 2 + N parameters
    """
    def objective(params):
        r_in = params[0]
        r_out = params[1]
        theta_in = params[2:2 + n_in]
        theta_out = params[2 + n_in:2 + n_in + n_out]
        
        # Build points
        x_in = r_in * np.cos(theta_in)
        y_in = r_in * np.sin(theta_in)
        x_out = r_out * np.cos(theta_out)
        y_out = r_out * np.sin(theta_out)
        
        pts = np.column_stack([np.concatenate([x_in, x_out]), 
                               np.concatenate([y_in, y_out])])
        
        return -_score(pts)
    
    return objective

def _make_polar_soft_objective(n_in, n_out, beta):
    """Create soft-min objective for polar parameterization."""
    def objective(params):
        r_in = params[0]
        r_out = params[1]
        theta_in = params[2:2 + n_in]
        theta_out = params[2 + n_in:2 + n_in + n_out]
        
        x_in = r_in * np.cos(theta_in)
        y_in = r_in * np.sin(theta_in)
        x_out = r_out * np.cos(theta_out)
        y_out = r_out * np.sin(theta_out)
        
        pts = np.column_stack([np.concatenate([x_in, x_out]), 
                               np.concatenate([y_in, y_out])])
        
        # Compute soft min score
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return 1e10
        try:
            hull = ConvexHull(pts)
            h_area = hull.volume
        except Exception:
            return 1e10
        if h_area < 1e-8:
            return 1e10
        
        pi = pts[I_IDX]
        pj = pts[J_IDX]
        pk = pts[K_IDX]
        v1 = pj - pi
        v2 = pk - pi
        cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) * 0.5
        
        m = np.min(cross)
        shifted = cross - m
        soft_min_val = m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))
        
        return -soft_min_val / h_area
    
    return objective

def _initial_polar(n_in, n_out, rng, r_in_base=0.5, r_out_base=1.0, noise=0.1):
    """Generate initial polar parameters."""
    r_in = r_in_base + rng.normal(0, noise * 0.1)
    r_out = r_out_base + rng.normal(0, noise * 0.1)
    theta_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + rng.normal(0, noise, n_in)
    theta_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False) + rng.normal(0, noise, n_out)
    return np.concatenate([[r_in, r_out], theta_in, theta_out])

def _polar_to_points(params, n_in, n_out):
    """Convert polar params to Cartesian points."""
    r_in = params[0]
    r_out = params[1]
    theta_in = params[2:2 + n_in]
    theta_out = params[2 + n_in:2 + n_in + n_out]
    
    x_in = r_in * np.cos(theta_in)
    y_in = r_in * np.sin(theta_in)
    x_out = r_out * np.cos(theta_out)
    y_out = r_out * np.sin(theta_out)
    
    return np.column_stack([np.concatenate([x_in, x_out]), 
                            np.concatenate([y_in, y_out])])

def _lbfgs_polar(n_in, n_out, init_params, beta, maxiter=2000):
    """Optimize polar parameters using L-BFGS-B with soft objective."""
    obj = _make_polar_soft_objective(n_in, n_out, beta)
    res = minimize(
        obj,
        init_params,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
    )
    if res.success or res.fun < obj(init_params):
        return res.x
    return init_params

def _nm_polar(n_in, n_out, init_params, maxiter=2000):
    """Optimize polar parameters using Nelder-Mead with hard objective."""
    obj = _make_polar_objective(n_in, n_out)
    res = minimize(
        obj,
        init_params,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    if res.fun < obj(init_params):
        return res.x
    return init_params

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    rng = np.random.RandomState(42)
    
    def time_ok():
        return time.time() < deadline
    
    best_pts = None
    best_score = -1e10
    
    # Try different splits of points between inner and outer rings
    splits = [(5, 9), (6, 8), (7, 7), (8, 6), (9, 5)]
    
    for n_in, n_out in splits:
        if not time_ok():
            break
        
        # Generate multiple initializations for each split
        for trial in range(3):
            if not time_ok():
                break
            
            init_params = _initial_polar(n_in, n_out, rng, 
                                        r_in_base=0.4 + 0.1 * trial, 
                                        r_out_base=0.9 + 0.05 * trial,
                                        noise=0.15)
            
            current_params = init_params.copy()
            current_pts = _polar_to_points(current_params, n_in, n_out)
            current_score = _score(current_pts)
            
            # Annealing schedule: start with high beta (smooth), end with low beta (sharp)
            beta_schedule = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5]
            
            for beta in beta_schedule:
                if not time_ok():
                    break
                
                polished_params = _lbfgs_polar(n_in, n_out, current_params, beta, maxiter=300)
                polished_pts = _polar_to_points(polished_params, n_in, n_out)
                polished_score = _score(polished_pts)
                
                if polished_score > current_score:
                    current_params = polished_params
                    current_pts = polished_pts
                    current_score = polished_score
            
            # Final Nelder-Mead polish
            if time_ok():
                final_params = _nm_polar(n_in, n_out, current_params, maxiter=1000)
                final_pts = _polar_to_points(final_params, n_in, n_out)
                final_score = _score(final_pts)
                
                if final_score > current_score:
                    current_pts = final_pts
                    current_score = final_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
    
    # Fallback: also try Cartesian optimization from best polar result
    if best_pts is not None and time_ok():
        # Convert to Cartesian and do a final polish
        res = minimize(
            lambda x: -_score(x.reshape(N, 2)),
            best_pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        if res.fun < -_score(best_pts):
            cart_pts = res.x.reshape(N, 2)
            cart_score = _score(cart_pts)
            if cart_score > best_score:
                best_score = cart_score
                best_pts = cart_pts
    
    if best_pts is None:
        # Ultimate fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts
