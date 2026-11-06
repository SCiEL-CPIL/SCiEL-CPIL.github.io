"""
Minimize logarithmic energy of 100 points on the unit sphere.
Strategy:
  1. Multiple diverse initializations (Fibonacci, random, structured).
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
    
    The Hessian of f_ij(p_i) = -log|p_i - p_j| at p_i, restricted to tangent space,
    for a tangent direction u (u·p_i = 0):
    H_ij(u,u) = u^T [I/d_ij^2 - 3(p_i-p_j)(p_i-p_j)^T/d_ij^4] u + correction for sphere curvature
    
    More precisely, on the sphere, the Hessian of g(p) = -log|p - q| at p (tangent):
    For tangent vector u at p:
    H(u,u) = u·u/d - 3(u·(p-q))^2/d^3 + u·u/d  ... let me derive carefully.
    
    Actually, let's use the formula: for f(p) = -log|p-q| on R^3,
    grad f = (p-q)/|p-q|^2
    Hessian in R^3: I/d - 3(p-q)(p-q)^T/d^3 where d = |p-q|
    
    On the sphere, the Riemannian Hessian in tangent direction u:
    Hess f(u,u) = u^T H_R3 f u + <grad f, II(u,u)>
    where II(u,u) = -p (second fundamental form of unit sphere)
    So: Hess_sphere(u,u) = u^T H_R3 f u + <grad f, -p>
    
    u^T H_R3 f u = u^T (I/d - 3(p-q)(p-q)^T/d^3) u
    <grad f, -p> = -< (p-q)/d^2, p> = -(1 - q·p)/d^2
    
    So Hess_sphere(u,u) = |u|^2/d - 3(u·(p-q))^2/d^3 - (1 - q·p)/d^2
    
    For the full Hessian matrix in tangent space, we need to express this as a 2x2 matrix
    for each point. We'll build it by applying to basis vectors.
    """
    n = len(pts)
    # We'll compute the Hessian as a 3x3 matrix per point, then project to tangent.
    # H_total[i] = sum_{j!=i} H_ij
    # where H_ij is the Hessian contribution from pair (i,j) at point i.
    
    # For efficiency, compute all pairwise contributions.
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
            # H_R3 = I/d - 3*diff*diff^T/d^3
            I_d = np.eye(3) / d
            outer = np.outer(diff_ij, diff_ij)
            H_R3 = I_d - 3.0 * outer / (d2 * d)
            # Correction: -<grad f, p_i> where grad f = diff_ij/d^2
            # <grad f, p_i> = <diff_ij, p_i>/d^2 = (1 - p_j·p_i)/d^2
            dot_pj_pi = np.dot(p_j, p_i)
            corr = -(1.0 - dot_pj_pi) / d2
            # Full Hessian in R^3 coordinates: H_R3 + corr * I (since II(u,u) = -p, and we add <grad f, -p> * I_u... 
            # Actually the correction is a scalar added to the quadratic form: Hess_sphere(u,u) = u^T H_R3 u + corr_scalar
            # where corr_scalar = -<grad f, p_i> = -(1-p_j·p_i)/d^2
            # So the effective Hessian matrix is H_R3 + corr_scalar * I
            H[i] += H_R3 + corr * np.eye(3)
    
    # Now project each 3x3 Hessian to tangent space.
    # For point i, tangent space is {u : u·p_i = 0}.
    # The Riemannian Hessian as a bilinear form on tangent space:
    # H_tan(u,v) = u^T H[i] v for u,v tangent to p_i.
    # We represent it as a 2x2 matrix in a local orthonormal basis, but for Newton
    # we work in R^3 and constrain to tangent.
    
    return H

def _newton_polish(pts0, max_iter=50, tol=1e-14):
    """
    Newton-Raphson method on the sphere using the Riemannian Hessian.
    At each step, solve H_tan * delta = -grad_tan, then move via exponential map.
    """
    pts = _project_to_sphere(pts0.copy())
    n = len(pts)
    
    for it in range(max_iter):
        energy, grad = _energy_and_grad(pts)
        if np.linalg.norm(grad) < tol:
            break
        
        # Compute Hessian (this is expensive: O(n^2) matrix operations)
        H = _hessian_tangent(pts)
        
        # For each point, solve the 2x2 tangent-space linear system.
        # Build orthonormal basis for tangent space at each point.
        new_pts = np.zeros_like(pts)
        for i in range(n):
            p = pts[i]
            g = grad[i]
            Hi = H[i]
            
            # Build orthonormal basis for tangent space at p
            if abs(p[0]) < 0.9:
                e1 = np.cross(p, [1, 0, 0])
            else:
                e1 = np.cross(p, [0, 1, 0])
            e1 = e1 / np.linalg.norm(e1)
            e2 = np.cross(p, e1)
            
            # Hessian in tangent basis: 2x2
            H2 = np.array([
                [np.dot(e1, Hi @ e1), np.dot(e1, Hi @ e2)],
                [np.dot(e2, Hi @ e1), np.dot(e2, Hi @ e2)]
            ])
            g2 = np.array([np.dot(g, e1), np.dot(g, e2)])
            
            # Solve H2 * delta2 = -g2
            try:
                delta2 = np.linalg.solve(H2 + 1e-12 * np.eye(2), -g2)
            except np.linalg.LinAlgError:
                delta2 = np.linalg.lstsq(H2 + 1e-12 * np.eye(2), -g2, rcond=None)[0]
            
            # Step size: use line search to ensure descent
            delta_tangent = delta2[0] * e1 + delta2[1] * e2
            
            # Damped Newton with backtracking line search
            best_alpha = 1.0
            best_e = energy
            for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
                new_p = _exp_map(p, delta_tangent * alpha)
                # Evaluate energy at new point (approximate: just check this point)
                # For efficiency, do a full energy eval
                new_pts_candidate = pts.copy()
                new_pts_candidate[i] = new_p
                e_candidate, _ = _energy_and_grad(new_pts_candidate)
                if e_candidate < best_e:
                    best_e = e_candidate
                    best_alpha = alpha
            
            new_pts[i] = _exp_map(p, delta_tangent * best_alpha)
        
        new_pts = _project_to_sphere(new_pts)
        
        # Check convergence
        e_new, g_new = _energy_and_grad(new_pts)
        if abs(e_new - energy) < 1e-15 and np.linalg.norm(g_new) < tol:
            pts = new_pts
            break
        pts = new_pts
    
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

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

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
    At each step, propose a random perturbation, accept if energy decreases
    or with probability exp(-delta_E / T).
    """
    rng = np.random.RandomState(seed)
    pts = pts0.copy()
    current_energy = _energy_and_grad(pts)[0]
    best_pts = pts.copy()
    best_energy = current_energy
    
    for it in range(n_iterations):
        # Linear cooling
        alpha = it / max(1, n_iterations - 1)
        T = T_start * (T_end / T_start) ** alpha
        
        # Propose random perturbation
        noise_scale = 0.02 * (1.0 - 0.5 * alpha)
        noise = rng.randn(N, 3) * noise_scale
        # Make noise tangent
        noise = noise - np.einsum('ij,ij->i', noise, pts)[:, np.newaxis] * pts
        new_pts = _project_to_sphere(pts + noise)
        
        new_energy, _ = _energy_and_grad(new_pts)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-15)):
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
    deadline = start_time + 145.0  # 5s buffer before 150s limit

    best_pts = None
    best_energy = np.inf

    # 1. Diverse Initializations
    initializations = []
    initializations.append(_fibonacci_sphere(N))
    initializations.append(_icosahedron_sphere(N))
    
    seeds = [42, 123, 777, 999, 2023, 555, 888, 1337, 31415, 27182]
    for s in seeds:
        initializations.append(_random_sphere(N, s))

    # 2. Coarse L-BFGS Optimization Phase
    for i, pts0 in enumerate(initializations):
        if time.time() > deadline - 60.0:
            break
        
        pts_opt, e_opt = _lbfgs_optimize(pts0, maxiter=2000, ftol=1e-12, gtol=1e-8)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()

    if best_pts is None:
        return _fibonacci_sphere(N)

    # 3. Simulated Annealing Phase
    time_left = deadline - time.time()
    if time_left > 40.0:
        n_iter = min(3000, int(time_left * 50))
        for sa_seed in [42, 123, 777]:
            if time.time() > deadline - 30.0:
                break
            pts_sa, e_sa = _simulated_annealing(
                best_pts, n_iterations=n_iter, T_start=0.03, T_end=0.0005, seed=sa_seed
            )
            if e_sa < best_energy:
                best_energy = e_sa
                best_pts = pts_sa.copy()
        
        # Re-optimize after SA
        if time.time() < deadline - 30.0:
            pts_opt, e_opt = _lbfgs_optimize(best_pts, maxiter=2000, ftol=1e-13, gtol=1e-9)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()

    # 4. Newton-Raphson Polish Phase
    time_left = deadline - time.time()
    if time_left > 20.0:
        pts_newton, e_newton = _newton_polish(best_pts, max_iter=30, tol=1e-14)
        if e_newton < best_energy:
            best_energy = e_newton
            best_pts = pts_newton.copy()

    # 5. Final L-BFGS polish
    time_left = deadline - time.time()
    if time_left > 10.0:
        pts_final, e_final = _lbfgs_optimize(best_pts, maxiter=5000, ftol=1e-15, gtol=1e-12)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final.copy()

    # 6. Final Newton polish
    time_left = deadline - time.time()
    if time_left > 5.0:
        pts_newton2, e_newton2 = _newton_polish(best_pts, max_iter=20, tol=1e-15)
        if e_newton2 < best_energy:
            best_energy = e_newton2
            best_pts = pts_newton2.copy()

    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: no points closer than 1e-9
    d2 = np.einsum('ij,ij->i', best_pts - best_pts[0], best_pts - best_pts[0])
    # Check all pairs
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them
                best_pts[j] = _project_to_sphere(best_pts[j] + 0.01 * np.random.randn(3))
    
    return best_pts
