"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates.
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    # Compute squared distances using dot products: |x_i - x_j|^2 = 2 - 2*x_i·x_j (for unit vectors)
    dots = pts @ pts.T  # (n, n)
    dist2 = 2.0 - 2.0 * dots
    np.fill_diagonal(dist2, 1.0)  # Avoid division by zero on diagonal
    
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of log(1/dist[i,j]) = -sum over i<j log(dist[i,j])
    # Use math.fsum for high precision as per evaluator
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -math.fsum(log_dists.tolist())
    
    # Gradient: dE/dx_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    # grad_i = - sum_j weight[i,j] * (x_i - x_j) = - (sum_j weight[i,j]*x_i) + sum_j weight[i,j]*x_j
    # = -x_i * sum_j weight[i,j] + sum_j weight[i,j]*x_j
    weight_sum = weight.sum(axis=1, keepdims=True)  # (n, 1)
    grad = -weight_sum * pts + weight @ pts
    
    return energy, grad

def _retract(pts):
    """Project points back onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _lbfgs_sphere(x0, deadline, maxiter=20000, ftol=1e-16, gtol=1e-14):
    """Run L-BFGS-B with manifold-aware gradient projection."""
    n = len(x0) // 3
    
    def obj_and_grad(x_flat):
        pts = x_flat.reshape(n, 3)
        pts = _retract(pts)
        energy, grad = _energy_and_grad(pts)
        # Project gradient to tangent space for correct unconstrained optimization
        proj = np.einsum('ij,ij->i', grad, pts)
        grad_tangent = grad - proj[:, np.newaxis] * pts
        return energy, grad_tangent.flatten()
    
    best_x = x0.copy()
    best_f, _ = obj_and_grad(x0)
    
    def callback(xk):
        nonlocal best_x, best_f
        if time.time() > deadline:
            return True
        f_k, _ = obj_and_grad(xk)
        if f_k < best_f:
            best_f = f_k
            best_x = xk.copy()
        return False
    
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxiter * 2},
            callback=callback
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except Exception:
        pass
    
    return _retract(best_x.reshape(n, 3))

def _polish(pts, deadline):
    """High-precision manifold gradient descent polish with adaptive step size."""
    n = pts.shape[0]
    lr = 1e-2
    f_curr, g_curr = _energy_and_grad(pts)
    
    for it in range(5000):
        if time.time() > deadline - 1.0:
            break
        
        # Project gradient to tangent space
        proj = np.einsum('ij,ij->i', g_curr, pts)
        g_tangent = g_curr - proj[:, np.newaxis] * pts
        
        # Adaptive step size with line search
        alpha = lr
        improved = False
        for _ in range(20):
            new_pts = _retract(pts - alpha * g_tangent)
            f_new, g_new = _energy_and_grad(new_pts)
            if f_new < f_curr:
                pts = new_pts
                f_curr = f_new
                g_curr = g_new
                lr = min(alpha * 1.2, 0.1)  # Allow slight increase if successful
                improved = True
                break
            alpha *= 0.4
        
        if not improved:
            # If no improvement, reduce learning rate
            lr *= 0.4
            if lr < 1e-16:
                break
        
        # Check convergence
        g_norm = np.linalg.norm(g_tangent)
        if g_norm < 1e-13:
            break
    
    return pts

def _ultra_polish(pts, deadline):
    """Final ultra-precise polish with very small steps."""
    n = pts.shape[0]
    lr = 1e-4
    f_curr, g_curr = _energy_and_grad(pts)
    
    for it in range(3000):
        if time.time() > deadline - 0.5:
            break
        
        # Project gradient to tangent space
        proj = np.einsum('ij,ij->i', g_curr, pts)
        g_tangent = g_curr - proj[:, np.newaxis] * pts
        
        # Small fixed step, check improvement
        new_pts = _retract(pts - lr * g_tangent)
        f_new, g_new = _energy_and_grad(new_pts)
        
        if f_new < f_curr:
            pts = new_pts
            f_curr = f_new
            g_curr = g_new
            lr = min(lr * 1.08, 1e-3)
        else:
            lr *= 0.4
            if lr < 1e-19:
                break
        
        if np.linalg.norm(g_tangent) < 1e-14:
            break
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate multiple diverse initial configurations
    inits = []
    
    # 1. Golden-angle spiral (Fibonacci)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    inits.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Random initialization with seed
    for seed in [0, 1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.normal(size=(N, 3))
        pts_rand = _retract(pts_rand)
        inits.append(pts_rand)
    
    # 3. Grid-based initialization (better distribution)
    # Use a spherical grid with slight perturbation
    n_theta = 10
    n_phi = 10
    theta = np.linspace(0, np.pi, n_theta)
    phi_grid = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    pts_grid = []
    for t in theta:
        for p in phi_grid:
            pts_grid.append([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
    pts_grid = np.array(pts_grid[:N])
    rng = np.random.default_rng(123)
    pts_grid += rng.normal(scale=0.05, size=pts_grid.shape)
    inits.append(_retract(pts_grid))
    
    # 4. Icosahedron-based initialization
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    rng = np.random.default_rng(999)
    ico_perturbed = ico + rng.normal(scale=0.1, size=ico.shape)
    ico_perturbed = _retract(ico_perturbed)
    
    try:
        points = [ico]
        for _ in range(3):
            new_points = []
            for face_idx in range(len(ico)):
                for _ in range(3):
                    u, v = rng.random(), rng.random()
                    if u + v > 1:
                        u, v = 1-u, 1-v
                    p = ico[face_idx] * (1-u-v) + ico[(face_idx+1)%12] * u + ico[(face_idx+2)%12] * v
                    new_points.append(p)
            points.append(_retract(np.array(new_points)))
        all_pts = np.vstack(points)
        idx = np.random.choice(len(all_pts), N, replace=False)
        inits.append(all_pts[idx])
    except Exception:
        inits.append(ico_perturbed)
    
    best_pts = None
    best_energy = float('inf')
    
    # Optimize each initial configuration
    for idx, pts_init in enumerate(inits):
        if time.time() > deadline - 25.0:
            break
        
        # Run first L-BFGS-B pass
        pts_opt = _lbfgs_sphere(pts_init.flatten(), deadline - 15.0)
        
        # Run second L-BFGS-B pass with tighter tolerances
        pts_opt = _lbfgs_sphere(pts_opt.flatten(), deadline - 8.0, maxiter=10000, ftol=1e-18, gtol=1e-16)
        
        # Polish with high-precision gradient descent
        pts_opt = _polish(pts_opt, deadline - 3.0)
        
        # Ultra polish
        pts_opt = _ultra_polish(pts_opt, deadline)
        
        f, _ = _energy_and_grad(pts_opt)
        if f < best_energy:
            best_energy = f
            best_pts = pts_opt
    
    # Final validation and normalization
    if best_pts is None:
        best_pts = inits[0]
    
    best_pts = _retract(best_pts)
    
    # Do a final polish on the best result if we have time
    if time.time() < deadline - 5.0:
        best_pts = _polish(best_pts, deadline - 2.0)
        best_pts = _ultra_polish(best_pts, deadline)
    
    best_pts = _retract(best_pts)
    
    # Ensure no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to spiral if points are too close
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _retract(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
