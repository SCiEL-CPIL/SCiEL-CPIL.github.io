"""Optimize logarithmic energy of 150 points on the unit sphere.

Strategy:
  - Multi-start local optimization (Fibonacci spiral, geodesic dome, random, clustered).
  - Basin hopping with Riemannian (exponential-map) perturbations that respect sphere geometry.
  - Riemannian L-BFGS polish: optimize in tangent space, then map back via exponential map.
  - Barzilai-Borwein adaptive step-size gradient descent for fast local convergence.
  - Final high-precision Riemannian polish pass.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150
DEADLINE = 148.0  # seconds, with 2s safety margin before 150s limit

# Precompute upper-triangular pair indices for fast energy/gradient evaluation
_IU = np.triu_indices(N, k=1)
_PAIR_I = _IU[0]
_PAIR_J = _IU[1]


def _energy_and_grad(pts):
    """Compute logarithmic energy and tangent-space gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    where d_ij is chordal distance.
    
    Tangent gradient: grad_i = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2, projected to tangent.
    Uses precomputed pair indices for speed.
    """
    # Compute pairwise differences for upper-triangular pairs only
    diff = pts[_PAIR_I] - pts[_PAIR_J]
    dist2 = np.einsum('ij,ij->i', diff, diff)
    dists = np.sqrt(dist2)
    energy = -math.fsum(np.log(dists).tolist())

    # inv_dist2 for each pair
    inv_d2 = 1.0 / dist2
    # diff_vec = x_i - x_j; contribution to grad_i is +inv_d2 * diff_vec
    #              contribution to grad_j is -inv_d2 * diff_vec
    weighted_diff = inv_d2[:, np.newaxis] * diff
    grad = np.zeros((N, 3))
    np.add.at(grad, _PAIR_I, weighted_diff)
    np.add.at(grad, _PAIR_J, -weighted_diff)

    # Project to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    return energy, grad


def _energy_only(pts):
    """Compute only the energy (fsum for precision). Uses precomputed pair indices."""
    diff = pts[_PAIR_I] - pts[_PAIR_J]
    dist2 = np.einsum('ij,ij->i', diff, diff)
    dists = np.sqrt(dist2)
    return -math.fsum(np.log(dists).tolist())


def _project_to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _exp_map_tangent(pts, tangent):
    """
    Exponential map on the sphere: move pts along tangent vectors.
    For a unit vector x and tangent v, the exponential map is
        exp_x(v) = cos(|v|) x + sin(|v|) (v/|v|)
    which is equivalent to rotating x by angle |v| about the axis v/|v|.
    This respects the sphere geometry exactly.
    """
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    safe_norm = np.where(norm > 1e-15, norm, 1.0)
    u = tangent / safe_norm
    c = np.cos(norm)
    s = np.sin(norm)
    out = c * pts + s * u
    return _project_to_sphere(out)


def _fibonacci_sphere(n, offset=0.0):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron.
    Provides a different structure than the Fibonacci spiral.
    """
    rng = np.random.RandomState(seed)
    
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    points = list(verts)
    
    # Subdivide until we have enough points
    while len(points) < n:
        new_faces = []
        mid_cache = {}
        
        def midpoint(p1, p2):
            key = (min(p1, p2), max(p1, p2))
            if key not in mid_cache:
                mid = (points[p1] + points[p2]) / 2.0
                mid = mid / np.linalg.norm(mid)
                mid_cache[key] = len(points)
                points.append(mid)
            return mid_cache[key]
        
        for face in faces:
            a, b, c = face
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        
        faces = new_faces
    
    return np.array(points[:n])


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _bb_gradient_descent(pts, deadline, maxiter=5000):
    """
    Barzilai-Borwein step-size gradient descent in tangent space.
    Alternates between BB1 and BB2 step sizes for fast local convergence.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Initial gradient
    e0, g0 = _energy_and_grad(pts)
    if not np.all(np.isfinite(g0)):
        return best_pts, best_energy
    
    g_prev = g0.copy()
    x_prev = pts.copy()
    alpha = 0.01  # initial step
    
    for it in range(maxiter):
        if time.time() >= deadline:
            break
        
        # Compute BB step size
        if it > 0:
            dx = pts - x_prev
            dg = g0 - g_prev
            s_dot_s = np.sum(dx * dx)
            s_dot_y = np.sum(dx * dg)
            y_dot_y = np.sum(dg * dg)
            
            if s_dot_s > 1e-30 and y_dot_y > 1e-30:
                if it % 2 == 0:
                    # BB1: alpha = s^T s / s^T y
                    alpha_bb = s_dot_s / max(abs(s_dot_y), 1e-30)
                else:
                    # BB2: alpha = s^T y / y^T y
                    alpha_bb = max(s_dot_y, 1e-30) / y_dot_y
                # Clamp step size
                alpha = np.clip(alpha_bb, 1e-8, 1.0)
            else:
                alpha *= 0.5
        
        # Gradient step in tangent space
        step = -alpha * g0
        new_pts = _exp_map_tangent(pts, step)
        new_pts = _project_to_sphere(new_pts)
        
        e_new, g_new = _energy_and_grad(new_pts)
        
        if not np.all(np.isfinite(e_new)) or not np.all(np.isfinite(g_new)):
            alpha *= 0.5
            continue
        
        if e_new < best_energy:
            best_energy = e_new
            best_pts = new_pts.copy()
        
        # Accept or backtrack
        if e_new <= e0 + 1e-4 * alpha * np.sum(g0 * step):
            x_prev = pts.copy()
            g_prev = g0.copy()
            pts = new_pts
            e0 = e_new
            g0 = g_new
        else:
            # Backtrack
            alpha *= 0.5
            if alpha < 1e-10:
                break
        
        # Check convergence
        gnorm = np.linalg.norm(g0)
        if gnorm < 1e-12:
            break
    
    return best_pts, best_energy


def _riemannian_lbfgs_polish(pts, deadline, maxiter=2000, ftol=1e-16, gtol=1e-13):
    """
    Polish a configuration using L-BFGS-B on tangent-space coordinates,
    followed by Barzilai-Borwein gradient descent for fine-tuning.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    def fun(t):
        tv = t.reshape(N, 3)
        dot = np.sum(tv * pts, axis=1, keepdims=True)
        tv = tv - dot * pts
        new_pts = _exp_map_tangent(pts, tv)
        e, g = _energy_and_grad(new_pts)
        return e, g.flatten()

    t0 = np.zeros(3 * N)
    try:
        res = minimize(fun, t0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol,
                                'gtol': gtol, 'maxls': 50})
        if time.time() < deadline:
            tv = res.x.reshape(N, 3)
            dot = np.sum(tv * pts, axis=1, keepdims=True)
            tv = tv - dot * pts
            new_pts = _exp_map_tangent(pts, tv)
            new_energy = _energy_only(new_pts)
            if new_energy < best_energy and np.all(np.isfinite(new_pts)):
                best_energy = new_energy
                best_pts = new_pts.copy()
    except Exception:
        pass

    # Barzilai-Borwein gradient descent polish
    if time.time() < deadline - 1:
        try:
            bb_pts, bb_energy = _bb_gradient_descent(best_pts, deadline, maxiter=3000)
            if bb_energy < best_energy:
                best_energy = bb_energy
                best_pts = bb_pts.copy()
        except Exception:
            pass

    return best_pts, best_energy


def _optimize_single(pts0, max_time):
    """Optimize a single starting configuration using Riemannian L-BFGS-B.
    
    Uses exponential map to respect sphere geometry exactly.
    """
    start_time = time.time()
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Phase 1: Initial Riemannian L-BFGS polish
    phase1_deadline = start_time + max_time * 0.7
    if time.time() < phase1_deadline:
        try:
            pts_opt, e_opt = _riemannian_lbfgs_polish(pts, phase1_deadline, maxiter=2000)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Basin hopping with Riemannian perturbations
    if time.time() < start_time + max_time * 0.9:
        T = 0.1
        T_min = 1e-6
        cooling = 0.99
        
        while time.time() < start_time + max_time * 0.9 and T > T_min:
            # Pick 1-3 random points to perturb simultaneously
            n_move = np.random.randint(1, 4)
            indices = np.random.choice(N, size=n_move, replace=False)
            
            new_pts = best_pts.copy()
            for idx in indices:
                rand_dir = np.random.standard_normal(3)
                rand_dir -= np.dot(rand_dir, best_pts[idx]) * best_pts[idx]
                norm = np.linalg.norm(rand_dir)
                if norm > 1e-15:
                    rand_dir /= norm
                step_size = T * 0.2 * np.sqrt(n_move)
                tangent = step_size * rand_dir
                new_pts[idx] = _exp_map_tangent(best_pts[idx:idx+1], tangent.reshape(1,3))[0]
            
            new_e = _energy_only(new_pts)
            delta_e = new_e - best_energy
            
            if delta_e < 0 or np.random.random() < math.exp(-delta_e / T):
                best_pts = new_pts
                best_energy = new_e
                # Quick polish after accepting
                if time.time() < start_time + max_time * 0.9:
                    try:
                        pts_polish, e_polish = _riemannian_lbfgs_polish(best_pts, 
                                                                        time.time() + 1.0, 
                                                                        maxiter=200)
                        if e_polish < best_energy:
                            best_energy = e_polish
                            best_pts = pts_polish.copy()
                    except Exception:
                        pass
            
            T *= cooling
    
    # Phase 3: Final high-precision polish (L-BFGS + BB gradient descent)
    if time.time() < start_time + max_time - 2:
        try:
            pts_final, e_final = _riemannian_lbfgs_polish(best_pts, 
                                                          start_time + max_time - 1, 
                                                          maxiter=3000, gtol=1e-14)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = pts_final.copy()
        except Exception:
            pass
    
    return _project_to_sphere(best_pts), best_energy


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Geodesic dome
    starts.append(_geodesic_dome(N, seed=42))
    
    # 3. Perturbed Fibonacci (different noise levels)
    for noise_scale in [0.03, 0.1, 0.2]:
        fib = _fibonacci_sphere(N)
        noise = noise_scale * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))
    
    # 4. Random points
    for _ in range(4):
        starts.append(_random_sphere(N, rng))
    
    # 5. Clustered starts (to explore different basins)
    for _ in range(2):
        centers = _random_sphere(10, rng)
        pts = []
        for c in centers:
            local = c + 0.1 * rng.standard_normal((15, 3))
            pts.append(_project_to_sphere(local))
        starts.append(np.vstack(pts))
    
    # Optimize each start
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Optimize all starts with equal time allocation
    n_starts = len(starts)
    phase1_deadline = start_time + (DEADLINE - start_time) * 0.65
    time_per_opt = (phase1_deadline - start_time) / n_starts
    time_per_opt = max(4.0, min(15.0, time_per_opt))
    
    for i, s in enumerate(starts):
        if time.time() > phase1_deadline - 5:
            break
        
        remaining = phase1_deadline - time.time()
        if remaining < 4:
            break
        
        opt_time = min(time_per_opt, remaining - 2)
        if opt_time < 3:
            break
        
        try:
            pts_opt, e_opt = _optimize_single(s, opt_time)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            continue
    
    # Phase 2: Final high-precision polish on the best result
    if best_pts is not None and time.time() < DEADLINE - 5:
        remaining = DEADLINE - time.time() - 3
        if remaining > 5:
            try:
                pts_final, e_final = _riemannian_lbfgs_polish(best_pts, 
                                                              time.time() + remaining, 
                                                              maxiter=4000, gtol=1e-14)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pts = pts_final.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    assert best_pts.shape == (N, 3)
    assert np.all(np.isfinite(best_pts))
    assert np.all(np.linalg.norm(best_pts, axis=1) > 1e-15)
    
    # Check no two points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    iu = np.triu_indices(N, k=1)
    min_dists = dists[iu]
    for idx in np.where(min_dists < 1e-9)[0]:
        i, j = iu[0][idx], iu[1][idx]
        best_pts[j] += 1e-8 * rng.standard_normal(3)
        best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
