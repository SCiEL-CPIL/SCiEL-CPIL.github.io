"""Logarithmic energy minimization of 150 points on the unit sphere.
Uses Riemannian optimization with exponential map, multi-start, and high-precision polishing."""

import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    The gradient is projected to the tangent space of the sphere.
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist2, 1.0)
    
    # Compute energy
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())
    
    # Compute gradient of E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad_i = sum_j (x_i - x_j) / d_ij^2 = x_i * sum_j 1/d_ij^2 - sum_j x_j / d_ij^2
    row_sums = np.sum(inv_dist2, axis=1)
    grad = pts * row_sums[:, np.newaxis] - inv_dist2 @ pts
    
    # Project gradient to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _energy_only(pts):
    """Compute only the energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    return -math.fsum(np.log(dists).tolist())


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _exp_map(pts, v):
    """
    Exponential map on the sphere: move from pts along tangent vector v.
    x_new = cos(||v||) * x + sin(||v||)/||v|| * v
    """
    v_norms = np.linalg.norm(v, axis=1, keepdims=True)
    v_norms_safe = np.maximum(v_norms, 1e-15)
    cos_t = np.cos(v_norms)
    sin_t_over = np.sin(v_norms) / v_norms_safe
    # For very small v_norms, sin(t)/t ≈ 1
    sin_t_over = np.where(v_norms < 1e-15, 1.0, sin_t_over)
    return cos_t * pts + sin_t_over * v


def _log_map(pts, target):
    """
    Logarithmic map: find tangent vector v at pts such that exp_map(pts, v) = target.
    v = arccos(<pts, target>) / ||target - <pts,target>pts|| * (target - <pts,target>pts)
    """
    cos_theta = np.sum(pts * target, axis=1, keepdims=True)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    # tangent component
    tangent = target - cos_theta * pts
    tangent_norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_norms_safe = np.maximum(tangent_norms, 1e-15)
    v = (theta / tangent_norms_safe) * tangent
    v = np.where(tangent_norms < 1e-15, 0.0, v)
    return v


def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    """Generate points using a Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron (geodesic dome),
    then project to sphere.
    """
    rng = np.random.RandomState(seed)
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    edges = [(0,11),(0,5),(0,1),(0,7),(0,10),(1,5),(5,11),(11,10),(10,7),
             (7,1),(1,6),(6,10),(2,11),(2,5),(2,9),(2,4),(2,8),(3,5),
             (3,11),(3,4),(3,9),(3,6),(4,9),(5,4),(11,4),(10,7),(6,1),
             (8,4),(9,8),(8,5),(6,9),(7,2),(8,3)]
    
    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)])
    
    if len(verts) > n:
        idx = rng.choice(len(verts), size=n, replace=False)
        verts = verts[idx]
    
    return verts


def _random_points(n, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _riemannian_adam(pts0, max_iter=5000, lr_start=0.01, lr_end=0.0005,
                     beta1=0.9, beta2=0.999, eps=1e-8, deadline=None, check_interval=50):
    """
    Riemannian Adam optimization using exponential map updates.
    This respects sphere geometry naturally.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    
    def lr_schedule(t):
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t / max_iter))

    for t in range(1, max_iter + 1):
        if deadline is not None and (t % check_interval == 0 or t == 1):
            if time.time() > deadline:
                break

        energy, grad = _energy_and_grad(pts)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        
        lr = lr_schedule(t)
        
        # Riemannian update via exponential map
        v_tangent = -lr * update
        # Ensure v_tangent is in tangent space
        dot_v = np.sum(v_tangent * pts, axis=1, keepdims=True)
        v_tangent = v_tangent - dot_v * pts
        
        pts_new = _exp_map(pts, v_tangent)

        if not np.all(np.isfinite(pts_new)):
            break

        energy_new = _energy_only(pts_new)
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
            
        pts = pts_new

    return best_pts, best_energy


def _lbfgs_polish(pts, deadline, maxiter=1000):
    """
    L-BFGS-B polish using flat Cartesian coordinates with projection.
    """
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 30,
                                'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        return p_res, e_check
    except Exception:
        return pts, _energy_only(pts)


def _conjugate_gradient_polish(pts, deadline, max_iter=500):
    """
    Riemannian conjugate gradient polish on the sphere.
    Uses exponential map for updates and backtracking line search.
    """
    pts = _project_to_sphere(pts.copy())
    current_energy = _energy_only(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    p = np.zeros_like(pts)  # search direction
    prev_grad = None
    
    for k in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        energy, grad = _energy_and_grad(pts)
        current_energy = energy
        
        # Check convergence
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-13:
            break
        
        if k == 0:
            p = -grad
        else:
            # Fletcher-Reeves beta
            prev_grad_norm2 = np.sum(prev_grad * prev_grad)
            curr_grad_norm2 = np.sum(grad * grad)
            beta = curr_grad_norm2 / prev_grad_norm2 if prev_grad_norm2 > 0 else 0.0
            p = -grad + beta * p
            # Project p to tangent space
            dot_p = np.sum(p * pts, axis=1, keepdims=True)
            p = p - dot_p * pts
        
        prev_grad = grad.copy()
        
        # Backtracking line search along geodesic
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-14:
            break
        
        alpha = 1.0
        c1 = 1e-4
        rho = 0.5
        
        for _ in range(30):
            if time.time() > deadline - 2.0:
                break
            
            v_step = alpha * p
            new_pts = _exp_map(pts, v_step)
            new_energy = _energy_only(new_pts)
            
            # Armijo condition
            if new_energy <= current_energy + c1 * alpha * np.sum(grad * p):
                break
            
            alpha *= rho
        
        new_pts = _exp_map(pts, alpha * p)
        new_energy = _energy_only(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
        
        pts = new_pts
    
    return best_pts, best_energy


def _simulated_annealing(points, deadline, rng, max_iter=3000):
    """Simulated Annealing to escape local minima using Riemannian updates."""
    current_points = points.copy()
    current_energy = _energy_only(current_points)
    best_points = current_points.copy()
    best_energy = current_energy
    
    T = 0.3
    T_min = 1e-5
    cooling = 0.998
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        if T < T_min:
            break
            
        n_perturb = rng.randint(1, 8)
        idxs = rng.choice(N, n_perturb, replace=False)
        
        new_points = current_points.copy()
        for idx in idxs:
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, new_points[idx]) * new_points[idx]
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-15:
                tangent /= norm_t
            else:
                continue
            step_size = T * rng.uniform(0.05, 0.5)
            v = step_size * tangent
            new_points[idx] = _exp_map(new_points[idx:idx+1], v.reshape(1,3))[0]
        
        new_energy = _energy_only(new_points)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
        
        T *= cooling

    return best_points, best_energy


def _basin_hopping(pts, deadline, rng, n_restarts=20, max_iter=2000):
    """
    Basin hopping: perturb + local minimize, keep best.
    """
    current = pts.copy()
    current_energy = _energy_only(current)
    best = current.copy()
    best_energy = current_energy
    
    T = 0.1
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb
        n_perturb = rng.randint(3, 15)
        idxs = rng.choice(N, n_perturb, replace=False)
        perturbed = current.copy()
        for idx in idxs:
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, perturbed[idx]) * perturbed[idx]
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-15:
                tangent /= norm_t
            step = T * rng.uniform(0.1, 1.0)
            v = step * tangent
            perturbed[idx] = _exp_map(perturbed[idx:idx+1], v.reshape(1,3))[0]
        
        # Local minimize
        p_res, e_res = _riemannian_adam(perturbed, max_iter=max_iter, deadline=deadline - 5.0)
        
        if e_res < best_energy:
            best_energy = e_res
            best = p_res.copy()
        
        # Accept/reject
        delta = e_res - current_energy
        if delta < 0 or (T > 0 and rng.rand() < math.exp(-delta / T)):
            current = p_res.copy()
            current_energy = e_res
        
        T *= 0.95

    return best, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    inits.append(_fibonacci_spiral(N))
    inits.append(_geodesic_dome(N, seed=0))
    
    for seed in range(8):
        inits.append(_random_points(N, seed=seed))
        
    for k in range(5):
        perturbed = _fibonacci_spiral(N) + rng.randn(N, 3) * 0.3
        inits.append(_project_to_sphere(perturbed))
    
    for k in range(3):
        perturbed = _geodesic_dome(N, seed=k) + rng.randn(N, 3) * 0.2
        inits.append(_project_to_sphere(perturbed))

    best_points = inits[0].copy()
    best_energy = _energy_only(best_points)
    
    # Phase 1: Riemannian Adam on all inits
    adam_results = []
    for init in inits:
        if time.time() > deadline - 60.0:
            break
        p_res, e_res = _riemannian_adam(init, max_iter=3000, deadline=deadline - 60.0)
        adam_results.append((e_res, p_res))
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()

    adam_results.sort(key=lambda x: x[0])
    
    # Phase 2: Basin hopping from top candidates
    candidates = [p for e, p in adam_results[:4]]
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 20.0:
            break
        bh_rng = np.random.RandomState(i * 137 + 42)
        bh_points, bh_energy = _basin_hopping(cand, deadline, bh_rng, n_restarts=10, max_iter=1500)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_points = bh_points.copy()

    # Phase 3: Simulated annealing from best
    if time.time() < deadline - 15.0:
        sa_rng = np.random.RandomState(999)
        sa_points, sa_energy = _simulated_annealing(best_points, deadline, sa_rng, max_iter=2000)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_points = sa_points.copy()

    # Phase 4: L-BFGS-B polish
    if time.time() < deadline - 10.0:
        final_points, final_energy = _lbfgs_polish(best_points, deadline, maxiter=2000)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points.copy()

    # Phase 5: Riemannian conjugate gradient for high precision
    if time.time() < deadline - 3.0:
        cg_points, cg_energy = _conjugate_gradient_polish(best_points, deadline, max_iter=500)
        if cg_energy < best_energy:
            best_energy = cg_energy
            best_points = cg_points.copy()

    # Phase 6: Final L-BFGS-B with tighter tolerance
    if time.time() < deadline - 1.0:
        final_points, final_energy = _lbfgs_polish(best_points, deadline, maxiter=500)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points.copy()

    best_points = _project_to_sphere(best_points)

    # Safety checks
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_spiral(N)

    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_spiral(N)

    return best_points
# EVOLVE-BLOCK-END
