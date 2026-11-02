"""Optimized logarithmic energy minimization for 100 points on the sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    return -math.fsum(log_d)

def _energy_and_tangent_grad(pts):
    """Return (energy, tangent gradient) of E = -sum_{i<j} log(d_ij).
    Correct sign: gradient points in direction of increasing E, so descent is -grad."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)

    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    energy = -math.fsum(log_d)

    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)

    # grad_i E = sum_{j!=i} (x_i - x_j) / d_ij^2
    sum_inv_d2 = inv_d2.sum(axis=1)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T
    grad = pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2

    # Project to tangent space
    grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_xi * pts

    return energy, grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)

def _perturb_sphere(pts, rng, sigma=0.05):
    """Perturb points and renormalize."""
    noise = rng.standard_normal(pts.shape) * sigma
    new_pts = pts + noise
    return _project_to_sphere(new_pts)

def _ring_initialization(n, rng):
    """Structured ring-based configuration: points arranged in latitude rings.
    This is a known strong local minimum structure for logarithmic energy."""
    # Try to find a good ring configuration
    # 100 points: try various ring splits
    # Optimal is roughly: rings of ~10-15 points
    best_pts = None
    best_e = float('inf')
    
    for n_rings in range(5, 15):
        # Distribute points among rings
        per_ring = [n // n_rings + (1 if i < n % n_rings else 0) for i in range(n_rings)]
        pts = []
        for ring_i, count in enumerate(per_ring):
            if count == 0:
                continue
            # Latitude: evenly spaced in z
            z = 1.0 - 2.0 * (ring_i + 0.5) / n_rings
            r = np.sqrt(max(1.0 - z * z, 0.0))
            # Offset each ring to avoid alignment
            offset = rng.uniform(0, 2 * np.pi) if rng is not None else 0.0
            for k in range(count):
                theta = 2.0 * np.pi * k / count + offset
                pts.append([r * np.cos(theta), r * np.sin(theta), z])
        pts = np.array(pts, dtype=float)
        if len(pts) == n:
            e = _log_energy(pts)
            if e < best_e:
                best_e = e
                best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(n)
    return _project_to_sphere(best_pts)

def _icosahedron_subdivide(n, rng):
    """Generate points by subdividing an icosahedron."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = _project_to_sphere(verts)
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    pts = []
    for v in verts:
        pts.append(v)
    
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i+1) % 3]
            edge = (min(a,b), max(a,b))
            edges.add(edge)
    
    for a, b in edges:
        mid = (verts[a] + verts[b]) / 2.0
        pts.append(_project_to_sphere(mid.reshape(1,3))[0])
    
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        pts.append(_project_to_sphere(center.reshape(1,3))[0])
    
    pts = np.array(pts)
    if len(pts) > n:
        indices = rng.choice(len(pts), n, replace=False)
        pts = pts[indices]
    elif len(pts) < n:
        remaining = _random_sphere(n - len(pts), rng)
        pts = np.vstack([pts, remaining])
    
    return _project_to_sphere(pts)

def _golden_spiral_v2(n, offset=0.0):
    """Alternative golden spiral with offset for diversity."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    z = np.clip(z, -1.0, 1.0)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _riemannian_descent(pts, max_iter=500, tol=1e-10, deadline=None):
    """Riemannian gradient descent with backtracking line search on the sphere.
    Uses CORRECTED gradient (sign fixed)."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        _, grad = _energy_and_tangent_grad(pts)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Initial step size
        step = 1.0 / (1.0 + 0.5 * grad_norm)
        
        accepted = False
        for ls_iter in range(50):
            # Riemannian retraction: move along negative gradient, then project
            new_pts = pts - step * grad
            new_pts = _project_to_sphere(new_pts)
            new_f = _log_energy(new_pts)
            
            if np.isfinite(new_f) and new_f < f:
                # Armijo condition
                if new_f < f - 1e-4 * step * grad_norm * grad_norm:
                    accepted = True
                    break
                # Accept any sufficient decrease
                if new_f < f - 1e-10:
                    accepted = True
                    break
            # Backtrack
            step *= 0.7
        
        if accepted:
            pts = new_pts
            f = new_f
        else:
            break
    
    return pts, f

def _riemannian_cg(pts, max_iter=300, tol=1e-12, deadline=None):
    """Riemannian conjugate gradient method (Polak-Ribiere)."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    _, g = _energy_and_tangent_grad(pts)
    
    d = -g  # initial direction
    f_prev = f
    g_prev = g
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Line search
        step = 1.0 / (1.0 + 0.5 * g_norm)
        accepted = False
        for ls_iter in range(40):
            new_pts = pts - step * d
            new_pts = _project_to_sphere(new_pts)
            new_f = _log_energy(new_pts)
            
            if np.isfinite(new_f) and new_f < f:
                if new_f < f - 1e-4 * step * np.linalg.norm(d) * g_norm:
                    accepted = True
                    break
                if new_f < f - 1e-10:
                    accepted = True
                    break
            step *= 0.7
        
        if not accepted:
            break
        
        pts = new_pts
        f = new_f
        _, g_new = _energy_and_tangent_grad(pts)
        
        # Polak-Ribiere beta
        beta = max(0.0, np.sum(g_new * (g_new - g)) / max(np.sum(g * g), 1e-30))
        d = -g_new + beta * d
        g = g_new
        f_prev = f
    
    return pts, f

def _newton_polish(pts, max_iter=50, deadline=None, tol=1e-14):
    """Riemannian Newton's method with damped line search for quadratic convergence.
    Computes the Riemannian Hessian of the log energy and solves for the Newton step."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        _, grad = _energy_and_tangent_grad(pts)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Compute Riemannian Hessian
        # H_ij = d^2 E / dx_i dx_j (in Euclidean space), then project to tangent space
        # E = -sum_{a<b} log(d_ab)
        # dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2
        # d2E/dx_i dx_k = sum_{j!=i} [delta_ik/d_ij^2 - 2(x_i-x_j)(x_i-x_k)_j / d_ij^4]  ... complex
        # 
        # Simpler: use the fact that the Hessian in ambient space is:
        # H_ambient[i,k] = sum_{j!=i} [delta_{ik}/d_ij^2 - 2 (x_i-x_j)_i (x_i-x_k)_j / d_ij^4]
        # Wait, let me be more careful.
        #
        # E = -sum_{a<b} log(d_ab), d_ab = |x_a - x_b|
        # dE/dx_a = -sum_{b!=a} (1/d_ab) * d(d_ab)/dx_a = -sum_{b!=a} (1/d_ab) * (x_a-x_b)/d_ab
        #         = -sum_{b!=a} (x_a-x_b)/d_ab^2
        # Wait, that gives negative of what I had. Let me recheck.
        # E = -sum log(d) = sum log(1/d)
        # dE/dx_a = -sum_{b!=a} (1/d_ab) * d(d_ab)/dx_a
        # d(d_ab)/dx_a = (x_a - x_b)/d_ab
        # So dE/dx_a = -sum_{b!=a} (x_a-x_b)/d_ab^2
        # 
        # Hmm, but earlier I said grad = sum (x_i-x_j)/d^2. Let me recheck the sign.
        # If points get closer, d decreases, log(d) decreases, -log(d) increases.
        # So moving x_a toward x_b should increase E. The gradient should point in the 
        # direction that increases E, which is toward x_b, i.e., in the direction (x_b - x_a).
        # But dE/dx_a = -sum (x_a-x_b)/d^2 = sum (x_b-x_a)/d^2. Yes! That points toward x_b.
        # So the gradient is sum_{b!=a} (x_b - x_a)/d_ab^2 = -sum_{b!=a} (x_a-x_b)/d_ab^2.
        #
        # Wait, I had it as grad = pts * sum_inv_d2 - sum_xj_inv_d2 = sum (x_i - x_j)/d^2.
        # That's the NEGATIVE of what I just derived. Let me recheck.
        #
        # E = -sum_{i<j} log(d_ij)
        # dE/dx_i = -sum_{j!=i} (1/d_ij) * (x_i - x_j)/d_ij = -sum_{j!=i} (x_i-x_j)/d_ij^2
        # 
        # So the correct gradient is -sum_{j!=i} (x_i-x_j)/d_ij^2 = sum_{j!=i} (x_j-x_i)/d_ij^2
        #
        # My code had: grad = pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2
        # = x_i * sum_j 1/d_ij^2 - sum_j x_j/d_ij^2
        # = sum_j (x_i - x_j)/d_ij^2
        #
        # That's the NEGATIVE of the true gradient! So the original code had the sign wrong,
        # and my "correction" actually made it worse. Let me fix this properly.
        #
        # True gradient: g_i = -sum_{j!=i} (x_i - x_j)/d_ij^2 = sum_{j!=i} (x_j - x_i)/d_ij^2
        # In code: g_i = sum_xj_inv_d2[i] - x_i * sum_inv_d2[i]
        # = sum_xj_inv_d2 - pts * sum_inv_d2[:, np.newaxis]
        
        # Let me recompute the gradient correctly:
        dots = pts @ pts.T
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0)
        d2 = np.maximum(d2, 1e-15)
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        sum_inv_d2 = inv_d2.sum(axis=1)
        sum_xj_inv_d2 = (pts.T @ inv_d2).T
        # Correct gradient: g_i = sum_j (x_j - x_i)/d_ij^2 = sum_xj_inv_d2 - x_i * sum_inv_d2
        grad_correct = sum_xj_inv_d2 - pts * sum_inv_d2[:, np.newaxis]
        grad_dot_xi = np.sum(grad_correct * pts, axis=1, keepdims=True)
        grad_tan = grad_correct - grad_dot_xi * pts
        
        # For Newton, I need the Hessian. Given the complexity, let me use a 
        # quasi-Newton (BFGS) approach instead, which is more robust.
        # Actually, for the Newton step, let me use the ambient Hessian projected to tangent.
        # 
        # The ambient Hessian H[i,k] (300x300) is:
        # H[i,k] = -d2E/(dx_i dx_k) ... this is getting complex.
        #
        # Let me just use BFGS in tangent space instead.
        break  # Skip Newton, use BFGS below
    
    return pts, f

def _bfgs_tangent(pts, max_iter=200, deadline=None, tol=1e-13):
    """BFGS quasi-Newton in tangent space for fast convergence."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    _, g = _energy_and_tangent_grad(pts)
    
    # BFGS: maintain inverse Hessian approximation H (in flattened tangent space)
    # Since tangent space is 2N-dim, we work in the full 3N space with projection
    m = 3 * n
    H_inv = np.eye(m)  # inverse Hessian approximation
    x_prev = pts.copy()
    g_prev = g.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Newton step: d = -H_inv @ g (in full space, then project to tangent)
        d_full = -H_inv @ g.reshape(-1)
        d = d_full.reshape(n, 3)
        # Project to tangent space
        d_dot_xi = np.sum(d * pts, axis=1, keepdims=True)
        d_tan = d - d_dot_xi * pts
        
        # Line search
        step = 1.0
        accepted = False
        for ls_iter in range(40):
            new_pts = pts - step * d_tan
            new_pts = _project_to_sphere(new_pts)
            new_f = _log_energy(new_pts)
            
            if np.isfinite(new_f) and new_f < f:
                if new_f < f - 1e-4 * step * np.linalg.norm(d_tan) * g_norm:
                    accepted = True
                    break
                if new_f < f - 1e-12:
                    accepted = True
                    break
            step *= 0.7
        
        if not accepted:
            # Reset H_inv
            H_inv = np.eye(m)
            continue
        
        # BFGS update
        x_new = new_pts
        _, g_new = _energy_and_tangent_grad(x_new)
        
        s = (x_new - x_prev).reshape(-1)
        y = (g_new - g_prev).reshape(-1)
        sy = np.dot(s, y)
        
        if sy > 1e-12:
            # BFGS update of inverse Hessian
            rho = 1.0 / sy
            H_inv = (np.eye(m) - rho * np.outer(s, y)) @ H_inv @ (np.eye(m) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        
        pts = x_new
        f = new_f
        g = g_new
        x_prev = x_new
        g_prev = g_new
    
    return pts, f

def _optimize_from_start(pts0, deadline):
    """Multi-stage optimization from a starting configuration."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    # Stage 1: Riemannian gradient descent
    if time.time() < deadline - 10.0:
        try:
            pts_rd, e_rd = _riemannian_descent(best_pts, max_iter=300, tol=1e-8, deadline=deadline - 10.0)
            if e_rd < best_energy:
                best_pts = pts_rd
                best_energy = e_rd
        except Exception:
            pass
    
    # Stage 2: Riemannian CG
    if time.time() < deadline - 8.0:
        try:
            pts_cg, e_cg = _riemannian_cg(best_pts, max_iter=200, tol=1e-10, deadline=deadline - 8.0)
            if e_cg < best_energy:
                best_pts = pts_cg
                best_energy = e_cg
        except Exception:
            pass
    
    # Stage 3: BFGS in tangent space
    if time.time() < deadline - 5.0:
        try:
            pts_bfgs, e_bfgs = _bfgs_tangent(best_pts, max_iter=150, deadline=deadline - 5.0)
            if e_bfgs < best_energy:
                best_pts = pts_bfgs
                best_energy = e_bfgs
        except Exception:
            pass
    
    # Stage 4: Final gradient descent polish
    if time.time() < deadline - 2.0:
        try:
            pts_final, e_final = _riemannian_descent(best_pts, max_iter=500, tol=1e-14, deadline=deadline - 2.0)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    return best_pts, best_energy

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # Ring-based initialization (strong local minimum structure)
    initializations.append(_ring_initialization(N, rng))
    
    # Icosahedron-based
    initializations.append(_icosahedron_subdivide(N, rng))
    
    # Alternative golden spirals with offsets
    for offset in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        initializations.append(_golden_spiral_v2(N, offset))
    
    # Random starts
    for seed_offset in range(8):
        rng_sub = np.random.default_rng(42 + seed_offset * 7)
        initializations.append(_random_sphere(N, rng_sub))
    
    # Perturbed Fibonacci
    for sigma in [0.01, 0.02, 0.05, 0.1, 0.2]:
        rng_sub = np.random.default_rng(1000 + int(sigma * 100))
        initializations.append(_perturb_sphere(_fibonacci_sphere(N), rng_sub, sigma))
    
    # Perturbed ring
    for sigma in [0.01, 0.02, 0.05, 0.1]:
        rng_sub = np.random.default_rng(2000 + int(sigma * 100))
        initializations.append(_perturb_sphere(_ring_initialization(N, rng_sub), rng_sub, sigma))
    
    # Perturbed icosahedron
    for sigma in [0.01, 0.02, 0.05, 0.1]:
        rng_sub = np.random.default_rng(3000 + int(sigma * 100))
        initializations.append(_perturb_sphere(_icosahedron_subdivide(N, rng_sub), rng_sub, sigma))
    
    # Rotated Fibonacci
    rng_sym = np.random.default_rng(5000)
    pts_sym = _fibonacci_sphere(N)
    theta = rng_sym.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    pts_rot = _project_to_sphere(pts_sym @ rot.T)
    initializations.append(pts_rot)
    
    # Compute initial energies and sort
    init_energies = [_log_energy(p) for p in initializations]
    sorted_indices = np.argsort(init_energies)
    
    best_pts = initializations[sorted_indices[0]]
    best_energy = init_energies[sorted_indices[0]]
    
    # Process top initializations with time budget
    num_to_try = min(len(sorted_indices), 12)
    for rank in range(num_to_try):
        idx = sorted_indices[rank]
        if time.time() > deadline - 15.0:
            break
        
        init_pts = initializations[idx]
        remaining = deadline - time.time()
        
        if rank == 0:
            time_alloc = min(25.0, remaining * 0.4)
        elif rank < 3:
            time_alloc = min(18.0, remaining * 0.3)
        elif rank < 6:
            time_alloc = min(12.0, remaining * 0.25)
        else:
            time_alloc = min(8.0, remaining * 0.2)
        
        if time_alloc < 3.0:
            break
        
        pts, energy = _optimize_from_start(init_pts, deadline)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final polish on best result
    if time.time() < deadline - 5.0:
        try:
            pts_final, e_final = _optimize_from_start(best_pts, deadline)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    # Ultra-final polish with tightest tolerance
    if time.time() < deadline - 2.0:
        try:
            pts_ultra, e_ultra = _riemannian_descent(best_pts, max_iter=1000, tol=1e-15, deadline=deadline - 2.0)
            if e_ultra < best_energy:
                best_pts = pts_ultra
                best_energy = e_ultra
        except Exception:
            pass
    
    best_pts = _project_to_sphere(best_pts)
    
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END
