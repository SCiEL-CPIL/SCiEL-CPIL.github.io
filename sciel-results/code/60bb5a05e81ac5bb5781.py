"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time

N = 150

def _normalize(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dist = np.sqrt(dist2[iu])
    dist = np.maximum(dist, 1e-15)
    return -np.sum(np.log(dist))

def _log_energy_grad(pts):
    """Compute Riemannian gradient of log energy on sphere.
    
    grad_riem[i] = (I - x_i x_i^T) * sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    
    # Euclidean gradient: grad[i] = sum_j (x_j - x_i) / |x_i - x_j|^2
    # = sum_j pts[j] / dist2[i,j] - pts[i] * sum_j 1/dist2[i,j]
    inv_d2 = 1.0 / dist2
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum = inv_d2 @ pts
    sum_inv = np.sum(inv_d2, axis=1)
    eucl_grad = weighted_sum - pts * sum_inv[:, np.newaxis]
    
    # Project to tangent space
    dot = np.sum(eucl_grad * pts, axis=1, keepdims=True)
    riem_grad = eucl_grad - dot * pts
    
    return riem_grad

def _exponential_map(pts, tangent):
    """Exponential map on sphere: exp_pts(v) = cos(|v|)*pts + sin(|v|)/|v| * v."""
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-15)
    cos_n = np.cos(norms)
    sin_over_n = np.sin(norms) / safe_norms
    return cos_n * pts + sin_over_n * tangent

def _icosahedron_points():
    """Generate 12 icosahedron vertices on unit sphere."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    pts = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1],
    ], dtype=float)
    return _normalize(pts)

def _subdivide_sphere(pts, target_n):
    """Grow a set of sphere points to target_n by repeatedly splitting the 
    point with the largest nearest-neighbor distance.
    
    This is a coarse-to-fine approach: start from symmetric points and 
    refine by splitting the most 'crowded' regions.
    """
    pts = pts.copy()
    n = pts.shape[0]
    
    if n >= target_n:
        return pts[:target_n]
    
    while n < target_n:
        # For each point, find its nearest neighbor distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, np.inf)
        
        # Nearest neighbor distance for each point
        nn_dist2 = np.min(dist2, axis=1)
        
        # Pick the point with the largest nearest-neighbor distance (most isolated)
        # This means it has the most "room" and splitting it will help most
        idx = np.argmax(nn_dist2)
        
        # Create a new point near this one, in a random tangent direction
        rng = np.random.RandomState(idx * 1000 + n)
        tangent = rng.randn(3)
        # Make tangent orthogonal to pts[idx]
        tangent = tangent - np.dot(tangent, pts[idx]) * pts[idx]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-15:
            tangent = np.array([1.0, 0.0, 0.0])
            tangent = tangent - np.dot(tangent, pts[idx]) * pts[idx]
            tangent_norm = np.linalg.norm(tangent)
        tangent = tangent / tangent_norm
        
        # Place new point at a small angular offset from pts[idx]
        # Use a moderate step so points don't overlap
        angle = 0.15  # ~8.6 degrees
        new_pt = _exponential_map(pts[idx:idx+1], angle * tangent.reshape(1,3))[0]
        
        pts = np.vstack([pts, new_pt.reshape(1,3)])
        n += 1
    
    return pts

def _riemannian_cg(pts, max_iter=300, tol=1e-12, time_limit=15.0):
    """Riemannian Conjugate Gradient using exponential map for sphere geometry."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy(current)
    grad = _log_energy_grad(current)
    
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for k in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Backtracking line search with exponential map
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad.copy()
        best_alpha = 0.0
        
        for _ in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy = _log_energy(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = _log_energy_grad(new_pts)
                best_alpha = alpha
            
            # Armijo-like condition
            if new_energy > energy - 1e-4 * alpha * np.sum(grad * d):
                break
            
            alpha *= 0.7
        
        if best_alpha == 0.0:
            break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Polak-Ribiere beta
        grad_diff = grad - prev_grad
        beta_num = np.sum(grad * grad_diff)
        beta_den = np.sum(prev_grad * prev_grad)
        if beta_den > 1e-30:
            beta = max(0.0, beta_num / beta_den)
            beta = min(beta, 1.0)
        else:
            beta = 0.0
        
        d_new = -grad
        d = d_new + beta * d
        prev_grad = grad.copy()
        
        # Ensure d is in tangent space
        dot = np.sum(d * current, axis=1, keepdims=True)
        d = d - dot * current
        
        # Restart if not descent direction
        if np.sum(grad * d) > 0:
            d = -grad
    
    return current, energy

def _riemannian_gd_momentum(pts, max_iter=500, tol=1e-10, time_limit=10.0):
    """Riemannian Gradient Descent with momentum and exponential map."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy(current)
    grad = _log_energy_grad(current)
    
    velocity = np.zeros_like(current)
    momentum = 0.0
    alpha = 0.3
    
    for k in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        if k >= 50:
            momentum = 0.4
        elif k >= 20:
            momentum = 0.3
        elif k >= 10:
            momentum = 0.15
        
        velocity = momentum * velocity - (1.0 - momentum) * grad
        
        # Backtracking line search
        step = alpha * velocity
        new_pts = _exponential_map(current, step)
        new_energy = _log_energy(new_pts)
        
        while new_energy > energy - 1e-4 * alpha * np.sum(velocity * grad):
            alpha *= 0.5
            step = alpha * velocity
            new_pts = _exponential_map(current, step)
            new_energy = _log_energy(new_pts)
            if alpha < 1e-15:
                break
        
        if alpha < 1e-15:
            break
        
        current = new_pts
        energy = new_energy
        grad = _log_energy_grad(current)
        
        if np.linalg.norm(grad) < tol:
            break
        
        if k % 20 == 0 and k > 0:
            alpha = min(alpha * 1.3, 1.0)
    
    return current, energy

def _lbfgs_polish(pts, max_iter=500, time_limit=10.0):
    """L-BFGS-B polish on flattened coordinates with projection."""
    start = time.time()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        e = _log_energy(p)
        g = _log_energy_grad(p)
        return e, g.flatten()
    
    res = minimize(fun_and_grad, pts.flatten(), method='L-BFGS-B', jac=True,
                   options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-13,
                            'maxfun': 5000, 'maxls': 40})
    
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    e_opt = _log_energy(x_opt)
    
    return x_opt, e_opt

def _perturb_and_optimize(pts, seed, time_limit=20.0):
    """Perturb points slightly and optimize. Used for diversity."""
    rng = np.random.RandomState(seed)
    perturbed = pts + rng.randn(N, 3) * 0.01
    perturbed = _normalize(perturbed)
    
    # GD + CG
    x1, e1 = _riemannian_gd_momentum(perturbed, max_iter=200, tol=1e-8, time_limit=time_limit*0.3)
    x2, e2 = _riemannian_cg(x1, max_iter=200, tol=1e-11, time_limit=time_limit*0.5)
    
    return x2, e2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_energy = np.inf
    best_points = None
    
    # Phase 1: Coarse-to-fine subdivision from icosahedron
    # Start with 12 icosahedron vertices, grow to 150
    ico_pts = _icosahedron_points()
    
    # Try multiple subdivision seeds for diversity
    subdivision_starts = []
    for seed in range(5):
        np.random.seed(seed)
        subdivided = _subdivide_sphere(ico_pts, N)
        subdivision_starts.append(subdivided)
    
    # Also try Fibonacci and random starts as fallbacks
    fib_i = np.arange(N) + 0.5
    fib_z = 1.0 - 2.0 * fib_i / N
    fib_phi = np.pi * (1.0 + np.sqrt(5.0)) * fib_i
    fib_r = np.sqrt(np.maximum(0.0, 1.0 - fib_z * fib_z))
    fib_pts = np.column_stack([fib_r * np.cos(fib_phi), fib_r * np.sin(fib_phi), fib_z])
    subdivision_starts.append(fib_pts)
    
    rng0 = np.random.RandomState(42)
    rand_pts = _normalize(rng0.randn(N, 3))
    subdivision_starts.append(rand_pts)
    
    # Phase 2: Optimize each start
    for idx, X0 in enumerate(subdivision_starts):
        if time.time() > deadline - 20.0:
            break
        
        remaining = deadline - 20.0 - time.time()
        time_budget = max(remaining / max(1, len(subdivision_starts) - idx), 5.0)
        
        try:
            # Step 1: Riemannian GD with momentum
            gd_time = min(time_budget * 0.3, 8.0)
            x1, e1 = _riemannian_gd_momentum(X0, max_iter=300, tol=1e-8, time_limit=gd_time)
            
            # Step 2: Riemannian CG with exponential map
            if time.time() < deadline - 15.0:
                cg_time = min(time_budget * 0.4, 12.0)
                x2, e2 = _riemannian_cg(x1, max_iter=300, tol=1e-11, time_limit=cg_time)
                if e2 < e1:
                    x1, e1 = x2, e2
            
            # Step 3: L-BFGS-B polish
            if time.time() < deadline - 10.0:
                lbfgs_time = min(time_budget * 0.3, 10.0)
                x3, e3 = _lbfgs_polish(x1, max_iter=400, time_limit=lbfgs_time)
                if e3 < e1:
                    x1, e1 = x3, e3
            
            if e1 < best_energy:
                best_energy = e1
                best_points = x1.copy()
                
        except Exception:
            pass
    
    # Phase 3: Perturbation-based diversity on best result
    if best_points is not None and time.time() < deadline - 15.0:
        for seed in range(4):
            if time.time() > deadline - 10.0:
                break
            try:
                xp, ep = _perturb_and_optimize(best_points, seed, time_limit=10.0)
                if ep < best_energy:
                    best_energy = ep
                    best_points = xp.copy()
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            # Riemannian CG for final convergence
            cg_time = min(8.0, deadline - 5.0 - time.time())
            x_final, e_final = _riemannian_cg(best_points, max_iter=300, tol=1e-13, time_limit=cg_time)
            if e_final < best_energy:
                best_points = x_final
                best_energy = e_final
            
            # One more L-BFGS-B pass
            if time.time() < deadline - 2.0:
                x_final2, e_final2 = _lbfgs_polish(best_points, max_iter=300, time_limit=5.0)
                if e_final2 < best_energy:
                    best_points = x_final2
                    best_energy = e_final2
        except Exception:
            pass
    
    # Ensure validity
    best_points = _normalize(best_points)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.sqrt(dist2[iu]).min()
    
    if min_dist < 1e-9 or not np.all(np.isfinite(best_points)):
        # Fall back to Fibonacci
        return fib_pts
    
    return best_points
# EVOLVE-BLOCK-END
