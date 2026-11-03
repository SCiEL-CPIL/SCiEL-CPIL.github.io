"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time

_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of -sum log(d_ij) with respect to points
    # d/dx_i [-log(d_ij)] = -(x_i - x_j) / d_ij^2
    # Sum over j != i: -sum_j (x_i - x_j)/d_ij^2 = -sum_j x_i/d_ij^2 + sum_j x_j/d_ij^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = -sum_j W_ij * x_i + sum_j W_ij * x_j = -W.sum(axis=1)*x_i + W @ x
    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)
    
    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _validity_check(pts):
    """Check if points are valid: finite, nonzero, on sphere, min distance > 1e-9."""
    if not np.all(np.isfinite(pts)):
        return False
    if pts.shape != (N, 3):
        return False
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-15):
        return False
    # Check min pairwise distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    if min_dist < 1e-9:
        return False
    return True


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _gradient_descent(pts, deadline, max_iter=5000, lr_init=0.01, lr_min=1e-12):
    """Adaptive gradient descent with backtracking line search and Riemannian exponential map."""
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    lr = lr_init
    n = N
    c1 = 0.1  # Armijo parameter
    rho = 0.5  # backtracking factor
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-13:
            break
        
        # Backtracking line search
        step = lr
        found = False
        for bt in range(20):
            # Riemannian exponential map: move along geodesic
            v = -step * g  # negative gradient direction
            v_norms = np.linalg.norm(v, axis=1, keepdims=True)
            v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
            v_unit = v / v_norms_safe
            cos_t = np.cos(v_norms)
            sin_t = np.sin(v_norms)
            new_pts = current * cos_t + v_unit * sin_t
            new_pts = _project(new_pts)
            
            new_e = _energy(new_pts)
            
            # Armijo condition: f(x + step * g) <= f(x) + c1 * step * ||g||^2
            if new_e <= e + c1 * step * g_norm * g_norm:
                found = True
                break
            step *= rho
        
        if not found:
            # If no acceptable step found, try a very small step
            step = max(lr * rho**20, 1e-15)
            v = -step * g
            v_norms = np.linalg.norm(v, axis=1, keepdims=True)
            v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
            v_unit = v / v_norms_safe
            cos_t = np.cos(v_norms)
            sin_t = np.sin(v_norms)
            new_pts = current * cos_t + v_unit * sin_t
            new_pts = _project(new_pts)
            new_e = _energy(new_pts)
        
        if new_e < best_e - 1e-15:
            current = new_pts
            best_e = new_e
            best_pts = current.copy()
            no_improve_count = 0
            lr = min(step * 1.5, 0.1)  # increase step size
        elif new_e < best_e:
            current = new_pts
            no_improve_count += 1
            lr = step
        else:
            lr = step * 0.5
            no_improve_count += 1
            if lr < lr_min or no_improve_count > 50:
                break
    
    return best_pts, best_e


def _langevin_verlet(pts, deadline, rng, max_iter=2000, dt=0.01, gamma=0.5, T_init=0.5, T_min=1e-5):
    """
    Langevin dynamics with Velocity Verlet integrator on the sphere using Riemannian updates.
    """
    current = pts.copy()
    velocity = np.zeros_like(current)
    best_e = _energy(current)
    best_pts = current.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Cooling schedule
        T = T_init * (T_min / T_init) ** (it / max_iter)
        
        # 1. Half step velocity update with force
        e, g = _energy_grad(current)
        F = -g
        velocity += 0.5 * dt * F
        
        # Project velocity to tangent space
        dot_v = np.sum(velocity * current, axis=1, keepdims=True)
        velocity = velocity - dot_v * current
        
        # 2. Full step position update using exponential map
        v = dt * velocity
        v_norms = np.linalg.norm(v, axis=1, keepdims=True)
        v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
        v_unit = v / v_norms_safe
        cos_t = np.cos(v_norms)
        sin_t = np.sin(v_norms)
        new_pts = current * cos_t + v_unit * sin_t
        new_pts = _project(new_pts)
        
        # 3. Compute new force
        new_e, new_g = _energy_grad(new_pts)
        new_F = -new_g
        
        # 4. Half step velocity update with new force
        velocity += 0.5 * dt * new_F
        
        # 5. Apply friction and noise (Langevin kick)
        # Project velocity to tangent space of new_pts
        dot_v = np.sum(velocity * new_pts, axis=1, keepdims=True)
        velocity = velocity - dot_v * new_pts
        
        # Damping
        velocity *= (1.0 - gamma * dt)
        
        # Noise
        if T > T_min:
            noise_scale = np.sqrt(2.0 * gamma * T * dt)
            noise = noise_scale * rng.standard_normal((N, 3))
            # Project noise to tangent space
            dot_n = np.sum(noise * new_pts, axis=1, keepdims=True)
            noise = noise - dot_n * new_pts
            velocity += noise
        
        # Update best
        if new_e < best_e:
            best_e = new_e
            best_pts = new_pts.copy()
            current = new_pts.copy()
        else:
            # Revert if energy increased significantly
            if new_e > best_e + 0.1:
                current = best_pts.copy()
                velocity = np.zeros_like(current)
    
    return best_pts, best_e


def _local_optimize(pts, maxiter=500, ftol=1e-16, gtol=1e-14):
    """Local optimization using L-BFGS-B with sphere constraint handled by projection."""
    def obj(x):
        pts_local = _project(x.reshape(N, 3))
        e, g = _energy_grad(pts_local)
        return e, g.flatten()
    
    try:
        res = minimize(obj, pts.flatten(), method='L-BFGS-B', 
                      jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                                         'maxls': 50, 'maxfun': 10000})
        best_pts = _project(res.x.reshape(N, 3))
        if _validity_check(best_pts):
            return best_pts
    except Exception:
        pass
    return pts


def _refine(pts, budget_seconds):
    """Multi-stage refinement with decreasing perturbation sizes."""
    start = time.time()
    current = pts.copy()
    current_energy = _energy(current)
    
    # Stage 1: Coarse local optimization
    if time.time() - start > budget_seconds * 0.7:
        return current
    
    current = _local_optimize(current, maxiter=200, ftol=1e-14, gtol=1e-12)
    current_energy = _energy(current)
    
    # Stage 2: Perturb and re-optimize
    rng = np.random.default_rng(42)
    for scale in [0.05, 0.01, 0.005, 0.001]:
        if time.time() - start > budget_seconds * 0.85:
            break
        for attempt in range(3):
            perturbed = current + rng.normal(0, scale, size=(N, 3))
            perturbed = _project(perturbed)
            if not _validity_check(perturbed):
                continue
            optimized = _local_optimize(perturbed, maxiter=300, ftol=1e-15, gtol=1e-13)
            e = _energy(optimized)
            if e < current_energy:
                current = optimized
                current_energy = e
    
    # Stage 3: Ultra-precise polish
    if time.time() - start < budget_seconds * 0.9:
        current = _local_optimize(current, maxiter=1000, ftol=1e-18, gtol=1e-15)
        if _validity_check(current):
            current_energy = _energy(current)
    
    return current


def construct():
    deadline = _DEADLINE - _SAFETY
    
    # Generate diverse initial configurations
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_spiral(N))
    
    # 2. Random uniform on sphere
    rng = np.random.default_rng(123)
    for _ in range(5):
        pts = _random_sphere(rng)
        starts.append(pts)
    
    # 3. Clustered starts (some points close together)
    for _ in range(3):
        pts = _random_sphere(rng)
        # Create some clusters
        for c in range(3):
            center = pts[c] / np.linalg.norm(pts[c])
            for j in range(c + 1, min(c + 5, N)):
                pts[j] = center + rng.normal(0, 0.1, size=3)
        pts = _project(pts)
        starts.append(pts)
    
    # 4. Grid-based starts
    for lat_div, lon_div in [(8, 12), (10, 10)]:
        lats = np.linspace(-np.pi/2 + np.pi/(2*lat_div), np.pi/2 - np.pi/(2*lat_div), lat_div)
        lons = np.linspace(0, 2*np.pi, lon_div, endpoint=False)
        pts_list = []
        for lat in lats:
            for lon in lons:
                x = np.cos(lat) * np.cos(lon)
                y = np.cos(lat) * np.sin(lon)
                z = np.sin(lat)
                pts_list.append([x, y, z])
        pts = np.array(pts_list[:N])
        if len(pts) == N:
            starts.append(_project(pts))
    
    # Evaluate and refine each start
    best_pts = starts[0]
    best_energy = _energy(best_pts)
    
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
        if not _validity_check(start_pts):
            continue
        
        e0 = _energy(start_pts)
        if e0 < best_energy:
            best_pts = start_pts
            best_energy = e0
        
        # Refine
        refined = _refine(start_pts, budget_seconds=25.0)
        e = _energy(refined)
        if e < best_energy:
            best_pts = refined
            best_energy = e
    
    # Use gradient descent for further refinement
    if time.time() < deadline - 10.0:
        gd_pts, gd_e = _gradient_descent(best_pts, deadline - 5.0, max_iter=3000, lr_init=0.005)
        if gd_e < best_energy:
            best_pts = gd_pts
            best_energy = gd_e
    
    # Use Langevin dynamics for exploration
    if time.time() < deadline - 10.0:
        lv_pts, lv_e = _langevin_verlet(best_pts, deadline - 5.0, rng, max_iter=1000)
        if lv_e < best_energy:
            best_pts = lv_pts
            best_energy = lv_e
    
    # Final polish on the best
    if time.time() < deadline - 5.0:
        final = _local_optimize(best_pts, maxiter=2000, ftol=1e-18, gtol=1e-16)
        if _validity_check(final):
            best_pts = final
    
    # Ensure valid output
    best_pts = _project(best_pts)
    if not _validity_check(best_pts):
        # Fallback to Fibonacci
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
