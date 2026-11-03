"""Logarithmic energy minimization of 150 points on the unit sphere.
Uses a multi-start strategy with Velocity-Verlet, Adam, and L-BFGS-B polishing."""

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
    Gradient: grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    The gradient is projected to the tangent space of the sphere.
    """
    n = pts.shape[0]
    # Compute pairwise distances squared
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist2, 1.0)
    
    # Compute energy
    # We only need the upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())
    
    # Compute gradient
    # grad_i = sum_j (x_j - x_i) / dist_ij^2
    # Note: 1/dist2 is symmetric
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad = inv_dist2 @ pts - pts * row_sums
    # row_sums[i] = sum_j inv_dist2[i,j]
    row_sums = np.sum(inv_dist2, axis=1)
    grad = inv_dist2 @ pts
    grad -= pts * row_sums[:, np.newaxis]
    
    # Project gradient to tangent space
    # tangent_grad = grad - (grad . x_i) x_i
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
    then project to sphere. Provides a different structure than Fibonacci spiral.
    """
    rng = np.random.RandomState(seed)
    # Start with icosahedron vertices
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Edges of icosahedron
    edges = [(0,11),(0,5),(0,1),(0,7),(0,10),(1,5),(5,11),(11,10),(10,7),
             (7,1),(1,6),(6,10),(2,11),(2,5),(2,9),(2,4),(2,8),(3,5),
             (3,11),(3,4),(3,9),(3,6),(4,9),(5,4),(11,4),(10,7),(6,1),
             (8,4),(9,8),(8,5),(6,9),(7,2),(8,3)]
    
    # Recursively subdivide until we have enough points
    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)])
    
    # If we have more than n, randomly select n
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


def _adam_optimize(pts0, max_iter=3000, lr_start=0.01, lr_end=0.001,
                   beta1=0.9, beta2=0.999, eps=1e-8, deadline=None, check_interval=20):
    """
    Adam optimization on the sphere with learning rate decay.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    
    # Cosine decay for learning rate
    def lr_schedule(t):
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t / max_iter))

    for t in range(1, max_iter + 1):
        if deadline is not None and (t % check_interval == 0 or t == 1):
            if time.time() > deadline:
                break

        energy, grad = _energy_and_grad(pts)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        
        lr = lr_schedule(t)
        
        # Step in tangent direction
        pts_new = pts - lr * update
        pts_new = _project_to_sphere(pts_new)

        if not np.all(np.isfinite(pts_new)):
            break

        energy_new = _energy_only(pts_new)
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
            
        pts = pts_new

    return best_pts, best_energy


def _riemannian_cg(pts0, deadline, maxiter=200, tol=1e-8):
    """
    Riemannian Conjugate Gradient method on the sphere.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    e, g = _energy_and_grad(pts)
    d = -g
    
    prev_g = g
    prev_d = d
    
    for k in range(maxiter):
        if time.time() > deadline - 0.5:
            break
        
        # Line search (Backtracking)
        alpha = 0.1
        e_new = float('inf')
        pts_new = pts
        for _ in range(50):
            cand = pts + alpha * d
            cand = _project_to_sphere(cand)
            e_cand = _energy_only(cand)
            if e_cand < e:
                e_new = e_cand
                pts_new = cand
                break
            alpha *= 0.5
            if alpha < 1e-10:
                break
        
        if not np.isfinite(e_new) or e_new >= e:
            # If line search failed, try smaller step or break
            break
            
        pts = pts_new
        e = e_new
        
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
            
        # Update gradient
        _, g_new = _energy_and_grad(pts)
        
        # Project gradient to tangent space (already done in _energy_and_grad)
        
        # Fletcher-Reeves beta
        beta = np.sum(g_new * g_new) / np.sum(prev_g * prev_g)
        
        # Parallel transport of d (approx by projection)
        # Simple projection for sphere
        dot_pd = np.sum(pts * prev_d, axis=1, keepdims=True)
        prev_d_tan = prev_d - dot_pd * pts
        
        d_new = -g_new + beta * prev_d_tan
        
        # Project d_new to tangent space
        dot_nd = np.sum(pts * d_new, axis=1, keepdims=True)
        d_new = d_new - dot_nd * pts
        
        prev_g = g_new
        prev_d = d_new
        g = g_new
        d = d_new
        
    return best_pts, best_energy


def _basin_hopping(pts0, deadline, rng, n_kicks=20, kick_scale=0.1):
    """
    Basin Hopping: Random kick + RCG minimize.
    """
    best_pts = pts0.copy()
    best_energy = _energy_only(pts0)
    
    for _ in range(n_kicks):
        if time.time() > deadline - 5.0:
            break
            
        # Kick
        p_kick = best_pts + rng.randn(N, 3) * kick_scale
        p_kick = _project_to_sphere(p_kick)
        
        # Minimize
        p_min, e_min = _riemannian_cg(p_kick, deadline, maxiter=100)
        
        if e_min < best_energy:
            best_energy = e_min
            best_pts = p_min.copy()
            
    return best_pts, best_energy


def _lbfgs_polish(pts, deadline, maxiter=500):
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
                                'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        if e_check < _energy_only(pts):
            return p_res
    except Exception:
        pass
    return pts


def _lbfgs_tight_polish(pts, deadline, maxiter=1000):
    """
    Ultra-tight L-BFGS-B polish.
    """
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 50,
                                'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 60})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        if e_check < _energy_only(pts):
            return p_res
    except Exception:
        pass
    return pts


def _local_perturb_restart(pts, rng, perturb_scale=0.05, n_perturb=10):
    """
    Create perturbed versions of pts by randomly moving n_perturb points.
    Returns list of perturbed point arrays.
    """
    results = []
    for _ in range(5):
        p = pts.copy()
        idx = rng.choice(N, size=n_perturb, replace=False)
        noise = rng.randn(n_perturb, 3) * perturb_scale
        p[idx] += noise
        p = _project_to_sphere(p)
        results.append(p)
    return results


def _gram_surrogate_energy_and_grad(pts, eps=1e-4):
    """
    Compute a smooth surrogate energy based on the Gram matrix.
    Uses log-sum-exp to create a smooth approximation that avoids singularities.
    
    The idea: for each pair (i,j), we have dist2_ij = 2(1 - G_ij) where G_ij = x_i . x_j.
    We approximate log(dist) smoothly using a soft formulation.
    
    Surrogate: E_sur = sum_{i<j} log(sqrt(2(1-G_ij)) + eps)
    This is smooth everywhere and approximates the true energy when eps is small.
    
    Gradient w.r.t. x_i: sum_{j!=i} (x_j - x_i) / (dist2_ij + eps^2) * (1/dist_ij)
    which simplifies to: sum_{j!=i} (x_j - x_i) / (dist2_ij + eps^2) * 1/sqrt(dist2_ij)
    But more carefully, d/dx_i log(sqrt(dist2_ij + eps^2)) = (x_j - x_i) / (dist2_ij + eps^2)
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Add epsilon to avoid singularity
    dist2_safe = dist2 + eps * eps
    np.fill_diagonal(dist2_safe, 1.0 + eps * eps)
    
    # Compute surrogate energy
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2_safe[iu])
    energy = -math.fsum(np.log(dists).tolist())
    
    # Compute gradient: d/dx_i log(sqrt(dist2_ij + eps^2)) 
    # = d/dx_i 0.5 * log(dist2_ij + eps^2)
    # = 0.5 * (d(dist2_ij)/dx_i) / (dist2_ij + eps^2)
    # = 0.5 * 2*(x_j - x_i) / (dist2_ij + eps^2)
    # = (x_j - x_i) / (dist2_ij + eps^2)
    
    inv_dist2_safe = 1.0 / dist2_safe
    np.fill_diagonal(inv_dist2_safe, 0.0)
    
    row_sums = np.sum(inv_dist2_safe, axis=1)
    grad = inv_dist2_safe @ pts
    grad -= pts * row_sums[:, np.newaxis]
    
    # Project to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _riemannian_cg_surrogate(pts0, deadline, maxiter=300, tol=1e-8, eps=1e-4):
    """
    Riemannian CG using the smooth Gram-matrix surrogate energy.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)  # Track true energy
    
    e, g = _gram_surrogate_energy_and_grad(pts, eps)
    d = -g
    
    prev_g = g
    prev_d = d
    
    for k in range(maxiter):
        if time.time() > deadline - 0.5:
            break
        
        # Line search (Backtracking with Armijo condition)
        alpha = 0.1
        e_new = float('inf')
        pts_new = pts
        for _ in range(50):
            cand = pts + alpha * d
            cand = _project_to_sphere(cand)
            e_cand, _ = _gram_surrogate_energy_and_grad(cand, eps)
            if e_cand < e:
                e_new = e_cand
                pts_new = cand
                break
            alpha *= 0.5
            if alpha < 1e-10:
                break
        
        if not np.isfinite(e_new) or e_new >= e:
            break
            
        pts = pts_new
        e = e_new
        
        # Track true energy
        e_true = _energy_only(pts)
        if e_true < best_energy:
            best_energy = e_true
            best_pts = pts.copy()
            
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
            
        # Update gradient
        _, g_new = _gram_surrogate_energy_and_grad(pts, eps)
        
        # Fletcher-Reeves beta
        prev_g_sq = np.sum(prev_g * prev_g)
        if prev_g_sq < 1e-20:
            beta = 0.0
        else:
            beta = np.sum(g_new * g_new) / prev_g_sq
        
        # Parallel transport of d
        dot_pd = np.sum(pts * prev_d, axis=1, keepdims=True)
        prev_d_tan = prev_d - dot_pd * pts
        
        d_new = -g_new + beta * prev_d_tan
        
        # Project d_new to tangent space
        dot_nd = np.sum(pts * d_new, axis=1, keepdims=True)
        d_new = d_new - dot_nd * pts
        
        prev_g = g_new
        prev_d = d_new
        g = g_new
        d = d_new
        
    return best_pts, best_energy


def _lbfgs_surrogate_polish(pts, deadline, maxiter=500, eps=1e-4):
    """
    L-BFGS-B polish using the smooth surrogate energy.
    """
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _gram_surrogate_energy_and_grad(p, eps)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 30,
                                'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        e_orig = _energy_only(pts)
        if e_check < e_orig:
            return p_res
    except Exception:
        pass
    return pts


def _annealed_surrogate_optimize(pts0, deadline, maxiter=500, eps_start=1e-2, eps_end=1e-5):
    """
    Annealed surrogate optimization: start with large eps (smooth landscape),
    gradually decrease to refine the solution.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    n_phases = 5
    phase_iters = maxiter // n_phases
    
    for phase in range(n_phases):
        eps = eps_start * (eps_end / eps_start) ** (phase / max(n_phases - 1, 1))
        
        phase_deadline = deadline - 1.0
        if time.time() > phase_deadline:
            break
        
        pts, e = _riemannian_cg_surrogate(pts, phase_deadline, maxiter=phase_iters, eps=eps)
        
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
        
        # Polish with L-BFGS using current eps
        if time.time() < phase_deadline - 2.0:
            pts_polished = _lbfgs_surrogate_polish(pts, phase_deadline - 2.0, maxiter=100, eps=eps)
            e_polished = _energy_only(pts_polished)
            if e_polished < best_energy:
                best_energy = e_polished
                best_pts = pts_polished.copy()
                pts = pts_polished
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    # Define multiple initializations
    # 1. Fibonacci spiral
    init1 = _fibonacci_spiral(N)
    # 2. Fibonacci spiral with different twist
    init2 = _fibonacci_spiral(N, twist=1.5)
    # 3. Random points
    init3 = _random_points(N, seed=123)
    # 4. Random points different seed
    init4 = _random_points(N, seed=456)
    # 5. Perturbed Fibonacci
    rng = np.random.RandomState(999)
    init5 = _project_to_sphere(_fibonacci_spiral(N) + rng.randn(N, 3) * 0.1)
    # 6. Geodesic dome
    init6 = _geodesic_dome(N, seed=7)

    inits = [init1, init2, init3, init4, init5, init6]
    
    best_pts_global = None
    best_energy_global = float('inf')

    # Phase 1: Multi-start with surrogate annealing (30s)
    time_per_restart = 15.0
    num_restarts = 4
    
    for i in range(num_restarts):
        current_deadline = start_time + (i + 1) * time_per_restart
        if current_deadline > deadline - 60.0:
            current_deadline = deadline - 60.0
            
        pts_start = inits[i % len(inits)]
        
        # Use annealed surrogate optimization
        if time.time() < current_deadline - 2.0:
            pts_cg, e_cg = _annealed_surrogate_optimize(pts_start, current_deadline, maxiter=200)
        else:
            pts_cg, e_cg = pts_start, _energy_only(pts_start)
            
        # Standard L-BFGS polish
        if time.time() < current_deadline - 0.5:
            pts_lbfgs = _lbfgs_polish(pts_cg, current_deadline - 0.5, maxiter=200)
            e_lbfgs = _energy_only(pts_lbfgs)
        else:
            pts_lbfgs, e_lbfgs = pts_cg, e_cg
            
        if e_lbfgs < best_energy_global:
            best_energy_global = e_lbfgs
            best_pts_global = pts_lbfgs.copy()

    # Phase 2: Basin hopping with surrogate (30s)
    if best_pts_global is not None and time.time() < deadline - 30.0:
        bh_deadline = deadline - 30.0
        rng_bh = np.random.RandomState(42)
        
        best_pts_bh = best_pts_global.copy()
        best_energy_bh = best_energy_global
        
        n_kicks = 20
        for _ in range(n_kicks):
            if time.time() > bh_deadline - 3.0:
                break
                
            # Kick
            p_kick = best_pts_bh + rng_bh.randn(N, 3) * 0.05
            p_kick = _project_to_sphere(p_kick)
            
            # Minimize using surrogate
            p_min, e_min = _annealed_surrogate_optimize(p_kick, bh_deadline - 3.0, maxiter=100)
            
            if e_min < best_energy_bh:
                best_energy_bh = e_min
                best_pts_bh = p_min.copy()
        
        # Final polish
        if time.time() < bh_deadline - 1.0:
            best_pts_bh = _lbfgs_polish(best_pts_bh, bh_deadline - 1.0, maxiter=200)
            best_energy_bh = _energy_only(best_pts_bh)
        
        if best_energy_bh < best_energy_global:
            best_energy_global = best_energy_bh
            best_pts_global = best_pts_bh.copy()

    # Phase 3: Final high-precision polish (remaining time)
    remaining_deadline = deadline - 2.0
    
    if best_pts_global is None:
        best_pts_global = _fibonacci_spiral(N)
        best_energy_global = _energy_only(best_pts_global)
        
    # Tight L-BFGS with more iterations
    if time.time() < remaining_deadline - 3.0:
        best_pts_global = _lbfgs_tight_polish(best_pts_global, remaining_deadline - 3.0, maxiter=1500)
        best_energy_global = _energy_only(best_pts_global)
        
    # One more standard L-BFGS with different settings
    if time.time() < remaining_deadline - 0.5:
        best_pts_global = _lbfgs_polish(best_pts_global, remaining_deadline - 0.5, maxiter=300)
        best_energy_global = _energy_only(best_pts_global)

    best_pts_global = _project_to_sphere(best_pts_global)

    # Validation
    if not np.all(np.isfinite(best_pts_global)):
        return _fibonacci_spiral(N)

    diff = best_pts_global[:, np.newaxis, :] - best_pts_global[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_spiral(N)

    return best_pts_global

# EVOLVE-BLOCK-END
