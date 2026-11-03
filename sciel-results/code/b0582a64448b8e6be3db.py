"""Global optimization using Basin Hopping for covering the sphere with 50 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, basin_hopping

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

def _objective(x):
    """Objective function for the optimizer."""
    pts = x.reshape(N, 3)
    return _covering_radius_exact(pts)

def _local_minimize(x):
    """Local minimization step using Nelder-Mead."""
    result = minimize(
        _objective,
        x,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
    )
    return result.x

def _fibonacci_sphere(n, offset=0.5):
    """Generate points on a sphere using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def construct():
    """
    Global optimization using Basin Hopping for minimizing the covering radius.
    """
    best_pts = None
    best_radius = np.inf
    
    # Initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random starts
    rng = np.random.RandomState(42)
    for _ in range(3):
        pts = rng.randn(N, 3)
        inits.append(_normalize(pts))
        
    # 3. Icosahedron-based start
    phi_golden = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1,  phi_golden, 0], [ 1,  phi_golden, 0], [-1, -phi_golden, 0], [ 1, -phi_golden, 0],
        [0, -1,  phi_golden], [0,  1,  phi_golden], [0, -1, -phi_golden], [0,  1, -phi_golden],
        [ phi_golden, 0, -1], [ phi_golden, 0,  1], [-phi_golden, 0, -1], [-phi_golden, 0,  1]
    ], dtype=float)
    ico_verts = _normalize(ico_verts)
    remaining = N - len(ico_verts)
    extra = rng.randn(remaining, 3)
    extra = _normalize(extra)
    inits.append(_normalize(np.vstack([ico_verts, extra])))

    # Run Basin Hopping from each initial configuration
    for init_idx, init in enumerate(inits):
        try:
            # Basin Hopping parameters
            # niter: number of basin hopping steps
            # T: temperature parameter
            # stepsize: perturbation size
            result = basin_hopping(
                _objective,
                init.flatten(),
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}},
                niter=50,
                T=1.0,
                stepsize=0.5,
                seed=init_idx
            )
            
            pts_final = _normalize(result.x.reshape(N, 3))
            final_radius_deg = _covering_radius_exact(pts_final)
            
            if final_radius_deg < best_radius:
                best_radius = final_radius_deg
                best_pts = pts_final
                
        except Exception:
            continue
    
    if best_pts is None:
        best_pts = inits[0]
        
    return best_pts
