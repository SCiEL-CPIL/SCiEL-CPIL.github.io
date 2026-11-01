import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 115.0  # Safety margin
EPS = 1e-12

def _energy_and_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|)
    and grad is the gradient w.r.t. pts in R^{N x 3}.
    """
    # Pairwise differences: (N, N, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    # Squared distances
    dist2 = np.sum(diff * diff, axis=2)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    # Safe distances
    dist = np.sqrt(np.where(mask, np.maximum(dist2, EPS), 1.0))
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # We sum over all i!=j and divide by 2
    log_terms = np.where(mask, -np.log(dist), 0.0)
    energy = 0.5 * np.sum(log_terms)
    
    # Gradient:
    # d/dp_i log(1/|p_i - p_j|) = -(p_i - p_j) / |p_i - p_j|^2
    # grad_i = sum_{j!=i} (p_j - p_i) / |p_i - p_j|^2
    # Note: dist2 is |p_i - p_j|^2
    inv_dist2 = np.where(mask, 1.0 / np.maximum(dist2, EPS), 0.0)
    
    # grad[i] = sum_j (pts[j] - pts[i]) * inv_dist2[i, j]
    # = sum_j pts[j] * inv_dist2[i, j] - pts[i] * sum_j inv_dist2[i, j]
    # Let's compute the sum of weights for each i
    weight_sums = np.sum(inv_dist2, axis=1, keepdims=True)
    # Sum of pts[j] * inv_dist2[i, j]
    # This is inv_dist2 @ pts
    weighted_pts = inv_dist2 @ pts
    
    grad = weighted_pts - weight_sums * pts
    
    return energy, grad

def _tangent_grad(pts, grad):
    """Project Euclidean gradient onto the tangent space."""
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - radial * pts

def _retract(pts, step):
    """Move points along step and project back to sphere."""
    new_pts = pts + step
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < EPS, 1.0, norms)
    return new_pts / norms

def _optimize_sphere(pts0, max_iters=2000, tol=1e-10, deadline=None):
    """
    Riemannian Gradient Descent with Backtracking Line Search and momentum.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.01
    momentum = np.zeros_like(pts)
    beta = 0.9
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Check convergence
        if np.linalg.norm(g_t) < tol:
            break
            
        # Update momentum
        momentum = beta * momentum - lr * g_t
        
        # Backtracking line search
        alpha = 1.0
        improved = False
        for _ in range(30):
            new_pts = _retract(pts, alpha * momentum)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(momentum * momentum):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            momentum = 0.9 * momentum
            if lr < 1e-12:
                break
            continue
        
        # Update gradient
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Slowly decay learning rate
        lr = max(0.001, lr * 0.9995)
        
    return best_pts, best_energy

def _rcg_polish(pts, max_iters=500, tol=1e-12, deadline=None):
    """
    Riemannian Conjugate Gradient method for faster convergence.
    """
    pts = pts.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    # Initial direction
    d = -g_t.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        # Line search
        alpha = 1.0
        improved = False
        for _ in range(30):
            new_pts = _retract(pts, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            new_g_t = _tangent_grad(new_pts, new_grad)
            
            if new_energy < energy:
                pts = new_pts
                energy = new_energy
                g_t = new_g_t
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            d = -g_t.copy()
            continue
        
        # Update direction using Fletcher-Reeves
        old_g_t = g_t.copy()
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        
        beta_fr = np.sum(g_t * g_t) / np.sum(old_g_t * old_g_t) if np.sum(old_g_t * old_g_t) > EPS else 0.0
        d = -g_t + beta_fr * d
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    return best_pts, best_energy

def _fibonacci_spiral(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _golden_spiral_v2(n):
    """Generate n points using improved golden spiral with better distribution."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    z = 1.0 - 2.0 * (i + 0.5) / n
    theta = golden_angle * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _icosahedron_subdivide(n_target=100):
    """
    Generate points by subdividing an icosahedron and projecting to sphere.
    This gives a more uniform distribution than random.
    """
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Icosahedron faces (12 vertices, 20 faces)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    points = list(verts)
    
    # Add edge midpoints
    edge_mids = set()
    for face in faces:
        for i in range(3):
            j = (i + 1) % 3
            a, b = face[i], face[j]
            mid = (verts[a] + verts[b]) / 2.0
            mid = mid / np.linalg.norm(mid)
            mid_key = tuple(np.round(mid, 10))
            if mid_key not in edge_mids:
                edge_mids.add(mid_key)
                points.append(mid)
    
    # Add face centers
    for face in faces:
        center = (verts[face[0]] + verts[face[1]] + verts[face[2]]) / 3.0
        center = center / np.linalg.norm(center)
        points.append(center)
    
    points = np.array(points)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # If we have more than n_target, randomly sample
    if len(points) > n_target:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(points), n_target, replace=False)
        points = points[indices]
    
    # If we have fewer, pad with Fibonacci
    if len(points) < n_target:
        fib = _fibonacci_spiral(n_target - len(points))
        points = np.vstack([points, fib])
    
    return points[:n_target]

def _random_points(n, seed=42):
    """Generate n random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _perturbed_points(base_pts, sigma, seed=0):
    """Perturb base points by adding Gaussian noise and reprojecting."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*base_pts.shape) * sigma
    pts = base_pts + noise
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _polish_lbfgs(pts, max_iter=200, gtol=1e-14):
    """Polish using L-BFGS-B with tight tolerances."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        norms = np.linalg.norm(x_pts, axis=1, keepdims=True)
        norms = np.where(norms < EPS, 1.0, norms)
        x_pts = x_pts / norms
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-18, 'maxls': 50})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
            norms = np.where(norms < EPS, 1.0, norms)
            x_opt = x_opt / norms
            e_opt, _ = _energy_and_grad(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _energy_and_grad(pts)[0]

def _polish_trust_constr(pts, max_iter=100):
    """Polish using trust-constr method for potentially better local convergence."""
    def fun(x):
        x_pts = x.reshape(N, 3)
        norms = np.linalg.norm(x_pts, axis=1, keepdims=True)
        norms = np.where(norms < EPS, 1.0, norms)
        x_pts = x_pts / norms
        e, _ = _energy_and_grad(x_pts)
        return e
    
    def jac(x):
        x_pts = x.reshape(N, 3)
        norms = np.linalg.norm(x_pts, axis=1, keepdims=True)
        norms = np.where(norms < EPS, 1.0, norms)
        x_pts = x_pts / norms
        _, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun, x0, jac=jac, method='trust-constr',
                       options={'maxiter': max_iter, 'gtol': 1e-12, 'xtol': 1e-12})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
            norms = np.where(norms < EPS, 1.0, norms)
            x_opt = x_opt / norms
            e_opt, _ = _energy_and_grad(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _energy_and_grad(pts)[0]

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(('fib', _fibonacci_spiral(N)))
    
    # 2. Golden spiral v2
    inits.append(('gold_v2', _golden_spiral_v2(N)))
    
    # 3. Icosahedron-based
    inits.append(('icosa', _icosahedron_subdivide(N)))
    
    # 4. Random with different seeds
    for seed in [1, 2, 3, 4, 5]:
        inits.append((f'rand{seed}', _random_points(N, seed=seed)))
    
    # 5. Perturbed Fibonacci with different scales
    fib_base = _fibonacci_spiral(N)
    for sigma, seed in [(0.1, 10), (0.05, 11), (0.2, 12), (0.03, 13)]:
        inits.append((f'pert{sigma}_{seed}', _perturbed_points(fib_base, sigma, seed=seed)))
    
    # 6. Perturbed golden spiral
    gold_base = _golden_spiral_v2(N)
    for sigma, seed in [(0.1, 20), (0.05, 21)]:
        inits.append((f'pert_gold{sigma}_{seed}', _perturbed_points(gold_base, sigma, seed=seed)))
    
    best_pts = None
    best_energy = np.inf
    
    # Optimize each initialization
    for name, init_pts in inits:
        elapsed = time.time() - t_start
        if elapsed > DEADLINE_SECONDS - 20:
            break
        
        # Riemannian Gradient Descent with more iterations
        pts_opt, energy_opt = _optimize_sphere(init_pts, max_iters=800, tol=1e-12, deadline=deadline - 20)
        
        # RCG polish for faster convergence
        if time.time() < deadline - 18:
            pts_rcg, energy_rcg = _rcg_polish(pts_opt, max_iters=300, tol=1e-12, deadline=deadline - 15)
            if energy_rcg < energy_opt:
                pts_opt, energy_opt = pts_rcg, energy_rcg
        
        # L-BFGS-B Polish
        if time.time() < deadline - 15:
            pts_opt, energy_opt = _polish_lbfgs(pts_opt, max_iter=200, gtol=1e-14)
        
        # Trust-constr polish for additional improvement
        if time.time() < deadline - 10:
            pts_tc, energy_tc = _polish_trust_constr(pts_opt, max_iter=50)
            if energy_tc < energy_opt:
                pts_opt, energy_opt = pts_tc, energy_tc
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() < deadline - 5:
        # RCG polish first for fast convergence
        if time.time() < deadline - 8:
            pts_rcg, energy_rcg = _rcg_polish(best_pts, max_iters=200, tol=1e-14, deadline=deadline - 6)
            if energy_rcg < best_energy:
                best_pts, best_energy = pts_rcg, energy_rcg
        
        # Multiple polish passes with tightening tolerances
        for gtol in [1e-14, 1e-15, 1e-16]:
            if time.time() > deadline - 3:
                break
            best_pts, best_energy = _polish_lbfgs(best_pts, max_iter=300, gtol=gtol)
    
    # Final Check and Return
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < EPS, 1.0, norms)
    best_pts = best_pts / norms
    return best_pts
