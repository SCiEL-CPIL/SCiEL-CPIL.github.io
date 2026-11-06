"""
Minimize logarithmic energy of 100 points on the unit sphere.
Strategy:
  1. Multiple diverse initializations (Fibonacci, jittered, random, icosahedron-based).
  2. Coarse L-BFGS optimization from each.
  3. Simulated annealing to escape local minima.
  4. Newton-Raphson polish with exact Hessian for machine-precision convergence.
"""
import numpy as np
import math
import time

N = 100

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient (projected to tangent space).
    Energy: sum_{i<j} log(1/|p_i - p_j|) = -sum_{i<j} log(d_ij)
    Gradient at p_i: sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2, projected to tangent.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)

    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask_upper]

    if np.any(d_upper < 1e-9):
        return 1e18, np.zeros_like(pts)

    energy = -math.fsum(math.log(di) for di in d_upper)

    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)

    grad = np.einsum('ij,ijk->ik', inv_d2, diff)

    radial = np.einsum('ij,ij->i', grad, pts)[:, np.newaxis]
    grad = grad - radial * pts

    return energy, grad

def _hessian_tangent(pts):
    """
    Compute the Hessian of the energy restricted to the tangent space.
    For each pair (i,j), the contribution to the Hessian at point i in the
    tangent space involves second derivatives of log(1/d_ij).
    
    On the sphere, the Riemannian Hessian in tangent direction u:
    Hess_sphere(u,u) = u^T H_R3 f u + corr_scalar
    where H_R3 = I/d - 3(p-q)(p-q)^T/d^3 and corr_scalar = -(1-q·p)/d^2
    """
    n = len(pts)
    H = np.zeros((n, 3, 3))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_i = pts[i]
            p_j = pts[j]
            diff_ij = p_i - p_j
            d2 = np.dot(diff_ij, diff_ij)
            d = math.sqrt(d2)
            if d < 1e-15:
                continue
            I_d = np.eye(3) / d
            outer = np.outer(diff_ij, diff_ij)
            H_R3 = I_d - 3.0 * outer / (d2 * d)
            dot_pj_pi = np.dot(p_j, p_i)
            corr = -(1.0 - dot_pj_pi) / d2
            H[i] += H_R3 + corr * np.eye(3)
    
    return H

def _newton_polish(pts0, max_iter=50, tol=1e-14):
    """
    Newton-Raphson method on the sphere using the Riemannian Hessian.
    At each step, compute full batch update: for all points simultaneously,
    solve the coupled system using the block-diagonal Hessian approximation.
    """
    pts = _project_to_sphere(pts0.copy())
    n = len(pts)
    
    for it in range(max_iter):
        energy, grad = _energy_and_grad(pts)
        if np.linalg.norm(grad) < tol:
            break
        
        H = _hessian_tangent(pts)
        
        # Compute tangent-space steps for all points
        deltas = np.zeros_like(pts)
        for i in range(n):
            p = pts[i]
            g = grad[i]
            Hi = H[i]
            
            if abs(p[0]) < 0.9:
                e1 = np.cross(p, [1.0, 0.0, 0.0])
            else:
                e1 = np.cross(p, [0.0, 1.0, 0.0])
            e1 = e1 / np.linalg.norm(e1)
            e2 = np.cross(p, e1)
            
            H2 = np.array([
                [np.dot(e1, Hi @ e1), np.dot(e1, Hi @ e2)],
                [np.dot(e2, Hi @ e1), np.dot(e2, Hi @ e2)]
            ])
            g2 = np.array([np.dot(g, e1), np.dot(g, e2)])
            
            try:
                delta2 = np.linalg.solve(H2 + 1e-12 * np.eye(2), -g2)
            except np.linalg.LinAlgError:
                delta2 = np.linalg.lstsq(H2 + 1e-12 * np.eye(2), -g2, rcond=None)[0]
            
            deltas[i] = delta2[0] * e1 + delta2[1] * e2
        
        # Full batch line search
        best_alpha = 1.0
        best_e = energy
        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]:
            new_pts_candidate = _exp_map(pts, deltas * alpha)
            e_candidate, _ = _energy_and_grad(new_pts_candidate)
            if e_candidate < best_e:
                best_e = e_candidate
                best_alpha = alpha
        
        pts = _exp_map(pts, deltas * best_alpha)
        pts = _project_to_sphere(pts)
        
        e_new, g_new = _energy_and_grad(pts)
        if abs(e_new - energy) < 1e-15 and np.linalg.norm(g_new) < tol:
            break
    
    return pts, _energy_and_grad(pts)[0]

def _exp_map_single(p, v):
    """Exponential map at point p with tangent vector v."""
    r = np.linalg.norm(v)
    if r < 1e-15:
        return p.copy()
    cos_r = math.cos(r)
    sin_r_over_r = math.sin(r) / r
    return p * cos_r + v * sin_r_over_r

def _exp_map(pts, tangent):
    """Exponential map on the unit sphere for all points."""
    result = np.zeros_like(pts)
    for i in range(len(pts)):
        result[i] = _exp_map_single(pts[i], tangent[i])
    return result

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, jitter=0.0):
    """Generate Fibonacci spiral points on unit sphere with optional jitter."""
    rng = np.random.RandomState(7)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        noise = rng.randn(n, 3) * jitter
        noise = noise - np.einsum('ij,ij->i', noise, pts)[:, np.newaxis] * pts
        pts = _project_to_sphere(pts + noise)
    return pts

def _random_sphere(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_sphere(n):
    """
    Generate points starting from icosahedron vertices and filling in.
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    pts = verts.copy()
    rng = np.random.RandomState(123)
    
    while len(pts) < n:
        candidates = rng.randn(500, 3)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        
        best_idx = 0
        best_min_dist = -1
        for c in candidates:
            dists = np.linalg.norm(pts - c, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = c
        
        pts = np.vstack([pts, best_idx.reshape(1, 3)])
    
    return pts[:n]

def _lbfgs_optimize(pts0, maxiter=3000, ftol=1e-12, gtol=1e-8):
    """L-BFGS optimization with projection onto sphere."""
    from scipy.optimize import minimize
    
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _energy_and_grad(pts)
        return g.flatten()

    try:
        result = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 10,
                     'ftol': ftol, 'gtol': gtol, 'm': 20}
        )
        best_pts = _project_to_sphere(result.x.reshape(-1, 3))
        best_energy = _energy_and_grad(best_pts)[0]
        return best_pts, best_energy
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def _simulated_annealing(pts0, n_iterations=2000, T_start=0.05, T_end=0.001, seed=42):
    """
    Simulated annealing on the sphere to escape local minima.
    Uses adaptive step size and occasional larger jumps.
    """
    rng = np.random.RandomState(seed)
    pts = pts0.copy()
    current_energy = _energy_and_grad(pts)[0]
    best_pts = pts.copy()
    best_energy = current_energy
    
    for it in range(n_iterations):
        alpha = it / max(1, n_iterations - 1)
        T = T_start * (T_end / T_start) ** alpha
        
        # Adaptive noise: larger early, smaller late; occasional big jump
        base_scale = 0.02 * (1.0 - 0.7 * alpha)
        if rng.random() < 0.05:
            noise_scale = base_scale * 3.0
        else:
            noise_scale = base_scale
        
        noise = rng.randn(N, 3) * noise_scale
        noise = noise - np.einsum('ij,ij->i', noise, pts)[:, np.newaxis] * pts
        new_pts = _project_to_sphere(pts + noise)
        
        new_energy, _ = _energy_and_grad(new_pts)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or (T > 1e-10 and rng.random() < math.exp(-delta_e / T)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
    
    return best_pts, best_energy

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    best_pts = None
    best_energy = np.inf

    # More diverse initializations
    initializations = []
    initializations.append(_fibonacci_sphere(N))
    initializations.append(_fibonacci_sphere(N, jitter=0.02))
    initializations.append(_fibonacci_sphere(N, jitter=0.05))
    initializations.append(_icosahedron_sphere(N))
    
    seeds = [42, 123, 777, 999, 2023, 555, 888, 1337, 31415, 27182, 16180, 98765]
    for s in seeds:
        initializations.append(_random_sphere(N, s))

    # Phase 1: Coarse L-BFGS from each initialization
    for i, pts0 in enumerate(initializations):
        if time.time() > deadline - 50.0:
            break
        
        pts_opt, e_opt = _lbfgs_optimize(pts0, maxiter=1500, ftol=1e-12, gtol=1e-7)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()

    if best_pts is None:
        return _fibonacci_sphere(N)

    # Phase 2: Simulated annealing to escape local minima
    time_left = deadline - time.time()
    if time_left > 35.0:
        n_iter = min(2500, int(time_left * 40))
        for sa_seed in [42, 123, 777, 999]:
            if time.time() > deadline - 25.0:
                break
            pts_sa, e_sa = _simulated_annealing(
                best_pts, n_iterations=n_iter, T_start=0.04, T_end=0.0003, seed=sa_seed
            )
            if e_sa < best_energy:
                best_energy = e_sa
                best_pts = pts_sa.copy()
            
            # Quick L-BFGS after SA to refine
            if time.time() < deadline - 25.0:
                pts_opt, e_opt = _lbfgs_optimize(best_pts, maxiter=1000, ftol=1e-13, gtol=1e-8)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_pts = pts_opt.copy()

    # Phase 3: Newton polish for machine-precision convergence
    time_left = deadline - time.time()
    if time_left > 15.0:
        pts_newton, e_newton = _newton_polish(best_pts, max_iter=30, tol=1e-14)
        if e_newton < best_energy:
            best_energy = e_newton
            best_pts = pts_newton.copy()

    # Phase 4: Final high-precision L-BFGS
    time_left = deadline - time.time()
    if time_left > 10.0:
        pts_final, e_final = _lbfgs_optimize(best_pts, maxiter=3000, ftol=1e-15, gtol=1e-12)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final.copy()

    # Phase 5: Final Newton polish
    time_left = deadline - time.time()
    if time_left > 5.0:
        pts_newton2, e_newton2 = _newton_polish(best_pts, max_iter=20, tol=1e-15)
        if e_newton2 < best_energy:
            best_energy = e_newton2
            best_pts = pts_newton2.copy()

    best_pts = _project_to_sphere(best_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                best_pts[j] = _project_to_sphere(best_pts[j] + 0.01 * np.random.randn(3))
    
    return best_pts
