"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius in radians.
    Uses a dense sampling of the sphere for robustness, or convex hull if available.
    For speed and reliability within the time budget, we use a combination of
    checking the convex hull facets (circumradii) which is the exact definition.
    """
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback if hull fails (e.g., coplanar points)
        # Use a grid of points on the sphere
        n_theta = 50
        n_phi = 100
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        theta, phi = np.meshgrid(theta, phi)
        y = np.column_stack([
            np.sin(theta)*np.cos(phi).ravel(),
            np.sin(theta)*np.sin(phi).ravel(),
            np.cos(theta).ravel()
        ])
        dots = y @ pts.T
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        min_angles = angles.min(axis=1)
        return np.max(min_angles)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        if len(simplex) == 3:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Side lengths
            ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
            bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
            ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
            
            # Circumradius of spherical triangle
            # sin(R) = sin(a)sin(b)sin(c) / (4 sin(s) cos(s-a) cos(s-b) cos(s-c))
            s = (ab + bc + ca) / 2.0
            sin_a = np.sin(ab)
            sin_b = np.sin(bc)
            sin_c = np.sin(ca)
            sin_s = np.sin(s)
            cos_sa = np.cos(s - ab)
            cos_sb = np.cos(s - bc)
            cos_sc = np.cos(s - ca)
            
            denom = 4.0 * sin_s * cos_sa * cos_sb * cos_sc
            if abs(denom) < 1e-15:
                R = max(ab, bc, ca) / 2.0
            else:
                sin_R = (sin_a * sin_b * sin_c) / denom
                sin_R = np.clip(sin_R, -1.0, 1.0)
                R = np.arcsin(sin_R)
            
            max_radius = max(max_radius, R)
    
    return max_radius

def _fibonacci_points(n, seed=0):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _energy_and_grad(pts, s):
    """
    Compute the Riesz s-energy and its gradient.
    E = sum_{i != j} ||x_i - x_j||^{-s}
    Grad_i = -s * sum_{j != i} (x_i - x_j) * ||x_i - x_j||^{-s-2}
    """
    n = len(pts)
    # Distance matrix
    # pts is (n, 3)
    # diff is (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dist_sq[mask] = 1.0 # Avoid div by zero, will be masked out in sum
    
    dist = np.sqrt(dist_sq)
    
    # Energy
    # E = 0.5 * sum_{i != j} dist^{-s} (factor 0.5 for double counting)
    # We sum over all i, j where i != j
    inv_dist = dist**(-s)
    inv_dist[~mask] = 0.0
    energy = 0.5 * np.sum(inv_dist)
    
    # Gradient
    # dE/dx_i = sum_{j != i} d/dx_i (||x_i - x_j||^{-s})
    # = sum_{j != i} (-s) ||x_i - x_j||^{-s-2} (x_i - x_j)
    # = -s * sum_{j != i} (x_i - x_j) * dist_{ij}^{-s-2}
    
    # dist_sq^{-s/2 - 1}
    # Note: dist^{-s-2} = (dist^2)^{-(s+2)/2}
    coeff = -s * (dist_sq**(-(s+2)/2.0))
    coeff[~mask] = 0.0
    
    # grad[i] = sum_j coeff[i,j] * diff[i,j]
    # diff[i,j] = x_i - x_j
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _optimize_energy(pts_init, deadline, s=20.0, max_iter=5000, lr_init=0.01):
    """
    Gradient descent with momentum on the sphere.
    """
    pts = pts_init.copy()
    n = len(pts)
    
    # Adaptive learning rate / momentum
    velocity = np.zeros_like(pts)
    momentum = 0.9
    lr = lr_init
    
    best_energy = np.inf
    best_pts = pts.copy()
    
    start_time = time.time()
    
    for k in range(max_iter):
        if time.time() > deadline:
            break
            
        energy, grad = _energy_and_grad(pts, s)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Update
        # Project gradient to tangent space
        # v_tangent = v - (v . x) x
        dot = np.sum(pts * grad, axis=1, keepdims=True)
        grad_tangent = grad - dot * pts
        
        # Momentum update
        velocity = momentum * velocity + grad_tangent
        # Normalize velocity to prevent huge steps
        v_norm = np.linalg.norm(velocity, axis=1, keepdims=True)
        v_norm = np.where(v_norm < 1e-12, 1.0, v_norm)
        velocity = velocity / v_norm
        
        # Step
        pts_new = pts + lr * velocity
        
        # Retract to sphere
        pts_new = _normalize(pts_new)
        
        # Check for improvement, otherwise reduce learning rate
        energy_new, _ = _energy_and_grad(pts_new, s)
        if energy_new > energy + 1e-8:
            lr *= 0.9
            pts = pts_new # Still move, but smaller effective step next time? 
            # Actually, standard backtracking is better
            # For simplicity, just reduce lr and keep pts_new
        else:
            pts = pts_new
            # Slowly increase lr if stable? No, keep it safe.
            
        # Decay momentum slightly if stuck?
        # Let's keep it simple.
        
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_pts = None
    best_cr = np.inf
    
    # Initializations
    inits = []
    # 1. Fibonacci
    inits.append(_fibonacci_points(N, seed=0))
    
    # 2. Random starts
    rng = np.random.RandomState(42)
    for i in range(5):
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        inits.append(pts)
        
    # 3. Perturbed Fibonacci
    for i in range(3):
        base = _fibonacci_points(N, seed=0)
        noise = rng.randn(N, 3) * 0.1
        pts = base + noise
        pts = _normalize(pts)
        inits.append(pts)
        
    # Optimization loop
    # We will run energy minimization with different exponents s
    # Higher s focuses more on the nearest neighbors, which is what we want for covering radius.
    s_values = [10, 20, 50]
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        # Quick check of initial CR
        cr_init = _covering_radius(init_pts)
        if cr_init < best_cr:
            best_cr = cr_init
            best_pts = init_pts.copy()
            
        for s in s_values:
            if time.time() > deadline - 5.0:
                break
                
            try:
                opt_pts = _optimize_energy(init_pts, deadline, s=s, max_iter=2000)
                cr_opt = _covering_radius(opt_pts)
                if cr_opt < best_cr:
                    best_cr = cr_opt
                    best_pts = opt_pts
            except Exception:
                pass

    # Final refinement if time permits
    if best_pts is not None and time.time() < deadline - 5.0:
        # Try a very high s to polish the closest pairs
        try:
            opt_pts = _optimize_energy(best_pts, deadline, s=100.0, max_iter=1000, lr_init=0.005)
            cr_opt = _covering_radius(opt_pts)
            if cr_opt < best_cr:
                best_cr = cr_opt
                best_pts = opt_pts
        except Exception:
            pass

    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_points(N, seed=0)
        
    return best_pts
# EVOLVE-BLOCK-END
