"""Direct optimization of the covering radius using smooth approximation and local search."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius_exact(pts):
    """
    Compute the exact covering radius in degrees using Convex Hull facets.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_radius_deg = 0.0
    for simplex in hull.simplices:
        p0 = pts[simplex[0]]
        p1 = pts[simplex[1]]
        p2 = pts[simplex[2]]
        
        a = p1 - p0
        b = p2 - p0
        aa = np.dot(a, a)
        bb = np.dot(b, b)
        ab = np.dot(a, b)
        
        denom = aa * bb - ab * ab
        if abs(denom) < 1e-15:
            continue
            
        s = (aa * bb - ab * ab) / (2.0 * denom)
        t = (aa * ab - aa * aa) / (2.0 * denom)
        
        circumcenter_2d = p0 + s * a + t * b
        R = np.linalg.norm(circumcenter_2d - p0)
        C_norm = np.linalg.norm(circumcenter_2d)
        
        if C_norm < 1e-15:
            theta = np.pi / 2.0
        else:
            cos_theta = (C_norm * C_norm + 1.0 - R * R) / (2.0 * C_norm)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
        
        radius_deg = np.degrees(theta)
        if radius_deg > max_radius_deg:
            max_radius_deg = radius_deg
            
    return max_radius_deg

def _smooth_objective(params, samples, beta=100.0):
    """
    Compute a smooth approximation of the covering radius.
    Minimizes the negative of the soft-minimum dot product.
    """
    pts = _normalize(params.reshape(N, 3))
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    
    # Soft min of dots: - (1/beta) * log(sum(exp(-beta * dots)))
    neg_beta_dots = -beta * dots
    max_val = np.max(neg_beta_dots, axis=1)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_dots - max_val), axis=1))
    soft_min_dot = -log_sum_exp / beta
    
    # We want to maximize the minimum dot product (minimize covering radius).
    # So we minimize -soft_min_dot.
    return -np.mean(soft_min_dot)

def construct():
    """
    Optimize the covering radius using multi-start local optimization.
    """
    # Generate sample points
    n_samples = 1000
    i = np.arange(n_samples)
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    best_pts = None
    best_radius = np.inf
    rng = np.random.RandomState(42)
    
    n_starts = 10
    
    for start_idx in range(n_starts):
        # Initialize with Fibonacci sphere + noise
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        init_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        noise_scale = 0.5 if start_idx == 0 else 0.2
        noise = rng.randn(N, 3) * noise_scale
        init_pts = init_pts + noise
        init_pts = _normalize(init_pts)
        
        try:
            # Phase 1: Smooth local optimization
            res_smooth = minimize(
                lambda x: _smooth_objective(x, samples, beta=50.0),
                init_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            pts_smooth = _normalize(res_smooth.x.reshape(N, 3))
            
            # Phase 2: Exact local optimization
            res_exact = minimize(
                lambda x: _covering_radius_exact(x.reshape(N, 3)),
                pts_smooth.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            pts_final = _normalize(res_exact.x.reshape(N, 3))
            final_radius = _covering_radius_exact(pts_final)
            
            if final_radius < best_radius:
                best_radius = final_radius
                best_pts = pts_final
                
        except Exception:
            continue
            
    if best_pts is None:
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
        
    return best_pts
