"""SCiEL-evolved program: Spherical optimization with L-BFGS-B restarts."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3
_IDX = np.array([(i, j) for i in range(N) for j in range(i + 1, N)], dtype=int)
DEADLINE = time.time() + 120.0 - 10.0  # Safety margin

def _score(pts):
    """Calculate the actual score: (max dist / min dist)^2"""
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = (diff ** 2).sum(-1)
    iu = np.triu_indices(N, 1)
    dd = d2[iu]
    lo = dd.min()
    hi = dd.max()
    if lo <= 1e-12:
        return 1e30
    return (hi / lo)

def _soft_objective(theta_phi, tau=50.0):
    """
    Objective function for points on the unit sphere.
    theta: (N,), phi: (N,)
    Returns soft log ratio of squared distances.
    """
    theta = theta_phi[:N]
    phi = theta_phi[N:]
    
    # Convert to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    pts = np.vstack([x, y, z]).T
    
    # Calculate squared distances
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = (diff ** 2).sum(-1)
    d2 = d2[_IDX]
    d2 = np.maximum(d2, 1e-12)
    
    # Soft log ratio: logsumexp(log d2) - logsumexp(-log d2)
    # This approximates max(log d2) - min(log d2)
    log_d2 = np.log(d2)
    max_term = np.log(np.sum(np.exp(log_d2 / tau))) - np.log(len(d2)) + np.log(tau) # Wait, standard soft max is (1/tau)logsumexp(x/tau)
    # Let's use standard soft max/min
    soft_max = (1.0/tau) * np.log(np.sum(np.exp(log_d2 / tau)))
    soft_min = (1.0/tau) * np.log(np.sum(np.exp(-log_d2 / tau)))
    # We want to minimize soft_max - soft_min
    return soft_max - soft_min

def _objective_3d(x, tau=50.0):
    """
    Objective function for points in 3D space.
    x: (3N,)
    Constraints handled by optimizer or we fix scale/center here?
    L-BFGS-B doesn't handle equality constraints well without transformation.
    We will use a penalty or just optimize and normalize later?
    Better: Transform to remove invariances.
    But L-BFGS-B is better on unconstrained or simple bounds.
    Let's stick to Sphere for the main search, it's more constrained and likely optimal.
    """
    pts = x.reshape(N, D)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = (diff ** 2).sum(-1)
    d2 = d2[_IDX]
    d2 = np.maximum(d2, 1e-12)
    log_d2 = np.log(d2)
    soft_max = (1.0/tau) * np.log(np.sum(np.exp(log_d2 / tau)))
    soft_min = (1.0/tau) * np.log(np.sum(np.exp(-log_d2 / tau)))
    return soft_max - soft_min

def _fibonacci_sphere(n):
    """Generate n points on a sphere using Fibonacci lattice."""
    pts = np.zeros((n, 3))
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1.0 - (i / (n - 1)) * 2.0
        r = np.sqrt(max(0.0, 1.0 - z * z))
        th = phi * i
        pts[i] = [np.cos(th) * r, np.sin(th) * r, z]
    return pts

def _cartesian_to_sphere(pts):
    """Convert Cartesian points to (theta, phi) for optimization."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # theta: polar angle (0 to pi)
    theta = np.arccos(np.clip(z / np.linalg.norm(pts, axis=1), -1.0, 1.0))
    # phi: azimuthal angle (0 to 2pi)
    phi = np.arctan2(y, x)
    return np.hstack([theta, phi])

def _sphere_to_cartesian(theta_phi):
    """Convert (theta, phi) to Cartesian points on unit sphere."""
    theta = theta_phi[:N]
    phi = theta_phi[N:]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.vstack([x, y, z]).T

def _icosahedron_plus_2():
    """
    Create 12 icosahedron vertices + 2 antipodal points.
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, t], [0, -1, t],
        [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0],
        [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [t, 0, -1],
        [-t, 0, 1], [-t, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Add 2 points along z-axis (antipodal)
    extra = np.array([[0, 0, 1], [0, 0, -1]])
    return np.vstack([ico, extra])

def _cube_octahedron():
    """
    8 Cube vertices + 6 Octahedron vertices.
    """
    cube = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                     [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=float)
    octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                     [0, 0, 1], [0, 0, -1]], dtype=float)
    pts = np.vstack([cube, octa])
    # Normalize to sphere
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def construct():
    t0 = time.time()
    rng = np.random.RandomState(42)
    
    # Initial candidates in Cartesian (on sphere)
    cands_cart = [
        _fibonacci_sphere(N),
        _icosahedron_plus_2(),
        _cube_octahedron()
    ]
    # Add random spherical starts
    for _ in range(20):
        v = rng.randn(N, 3)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        cands_cart.append(v)
        
    best_pts = None
    best_val = np.inf
    
    # Convert to sphere params and optimize
    for cart in cands_cart:
        if time.time() > DEADLINE:
            break
            
        tp = _cartesian_to_sphere(cart)
        
        try:
            # L-BFGS-B optimization on sphere parameters
            # Bounds: theta in [0, pi], phi in [0, 2pi]
            bounds = [(0, np.pi)] * N + [(0, 2 * np.pi)] * N
            res = minimize(_soft_objective, tp, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
            
            if res.success:
                opt_tp = res.x
                opt_cart = _sphere_to_cartesian(opt_tp)
                val = _score(opt_cart)
                if val < best_val:
                    best_val = val
                    best_pts = opt_cart.copy()
        except Exception:
            pass
            
    # If no spherical solution found, fallback to random
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        
    # Optional: Try a few 3D free optimizations if time permits
    # This allows non-spherical shapes (e.g. flattened)
    if time.time() < DEADLINE - 20:
        for _ in range(5):
            if time.time() > DEADLINE:
                break
            # Start from current best, add noise
            x0 = best_pts.reshape(-1) + rng.randn(N, 3) * 0.1
            try:
                res = minimize(_objective_3d, x0, method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    opt_pts = res.x.reshape(N, D)
                    # Normalize to unit sphere to ensure fair comparison? 
                    # No, score is scale invariant. But we should ensure points are distinct.
                    val = _score(opt_pts)
                    if val < best_val:
                        best_val = val
                        best_pts = opt_pts.copy()
            except Exception:
                pass

    return best_pts

