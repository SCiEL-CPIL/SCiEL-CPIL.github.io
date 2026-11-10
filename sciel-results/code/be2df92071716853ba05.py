import numpy as np
import time
import math

N = 150

def _energy(points):
    """Compute logarithmic energy using math.fsum for precision."""
    n = points.shape[0]
    # Compute pairwise distances
    # dist_sq[i, j] = ||x_i - x_j||^2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    d = dist[iu[0], iu[1]]
    
    # Calculate log(1/d) terms
    # We must be careful with d=0, but points are on sphere and distinct.
    # If d is very small, log(1/d) is large positive.
    terms = np.log(1.0 / d)
    
    # Use math.fsum for exact summation as per task description
    return math.fsum(terms.tolist())

def _grad_and_energy(points):
    """Compute log energy E = sum_{i<j} log(1/|xi-xj|) and gradient on R3.
    dE/dxi = - sum_{j!=i} (xi - xj) / |xi - xj|^2
    """
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(n, k=1)
    # Use fsum for energy to match evaluator
    d = dist[iu[0], iu[1]]
    energy = math.fsum(np.log(1.0 / d).tolist())

    # grad[i] = - sum_{j!=i} (xi - xj) / dist_ij^2
    coeff = np.zeros_like(dist_sq)
    mask = ~np.eye(n, dtype=bool)
    coeff[mask] = -1.0 / dist_sq[mask]

    # grad[i] = sum_j coeff[i,j] * (xi - xj)
    #         = xi * sum_j coeff[i,j] - sum_j coeff[i,j]*xj
    row_sums = coeff.sum(axis=1)
    grad = points * row_sums[:, np.newaxis] - (coeff @ points)

    return energy, grad

def _project_to_sphere(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _tangent(v, x):
    dot = np.sum(v * x, axis=1, keepdims=True)
    return v - dot * x

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _halton_sequence(n, d=3, skip=1):
    """Generate Halton sequence points on the sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(skip, n + skip):
        p = [halton(i, b) for b in [2, 3, 5]]
        # Map [0,1]^3 to sphere
        # Use a simple mapping: x = 2p[0]-1, y = 2p[1]-1, z = 2p[2]-1
        # Then normalize. This is not uniform on sphere surface but a decent start.
        # Better: Use spherical coords from Halton
        # u = 2*p[0] - 1 (cos theta)
        # phi = 2*pi*p[1]
        u = 2.0 * p[0] - 1.0
        phi = 2.0 * math.pi * p[1]
        x = np.sqrt(1.0 - u*u) * np.cos(phi)
        y = np.sqrt(1.0 - u*u) * np.sin(phi)
        z = u
        points.append([x, y, z])
    return np.array(points)

def _lbfgs_optimize(points, max_iter=800, ftol=1e-16, gtol=1e-14, deadline=None):
    """Riemannian L-BFGS via scipy on the sphere (projected)."""
    from scipy.optimize import minimize

    def objective_and_grad(x_flat):
        pts = x_flat.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, g = _grad_and_energy(pts)
        g_t = _tangent(g, pts)
        return e, g_t.reshape(-1)

    try:
        result = minimize(
            objective_and_grad,
            points.reshape(-1),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 60,
                     'iprint': -1}
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            cand = _project_to_sphere(result.x.reshape(-1))
            e = _energy(cand)
            return cand, e
    except Exception:
        pass
    return points, _energy(points)

def _riemannian_cg(points, max_iter=400, tol=1e-13, deadline=None):
    """Riemannian conjugate gradient with backtracking on the sphere."""
    X = _project_to_sphere(points.copy())
    n = X.shape[0]

    e, g = _grad_and_energy(X)
    g_t = _tangent(g, X)

    if np.linalg.norm(g_t) < tol:
        return X, e

    d = -g_t
    g_old_t = g_t

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Backtracking line search along geodesic (projected) direction
        # Use Armijo condition for better stability
        best_alpha = 1e-6
        best_e = e
        best_X = X.copy()
        alpha = 1.0
        c1 = 1e-4
        for _ in range(40):
            X_new = _project_to_sphere(X + alpha * d)
            e_new = _energy(X_new)
            if e_new < e + c1 * alpha * np.sum(g_t * d):
                best_e = e_new
                best_alpha = alpha
                best_X = X_new
                break
            alpha *= 0.5
            if alpha < 1e-12:
                break
        else:
            # If no improvement, try smaller step or break
            if best_e >= e:
                 # Try a tiny step
                 X_new = _project_to_sphere(X + 1e-8 * d)
                 e_new = _energy(X_new)
                 if e_new < e:
                     best_e = e_new
                     best_alpha = 1e-8
                     best_X = X_new
                 else:
                     break

        X = best_X
        e = best_e

        e_new, g_new = _grad_and_energy(X)
        g_new_t = _tangent(g_new, X)
        if np.linalg.norm(g_new_t) < tol:
            break

        beta = (np.sum(g_new_t * g_new_t) / (np.sum(g_old_t * g_old_t) + 1e-30))
        # Powell restart
        if np.sum(g_new_t * g_old_t) > 0:
            beta = 0.0
        d = -g_new_t + beta * d
        d = _tangent(d, X)
        g_old_t = g_new_t

    return X, e

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # ---- Build diverse initial configurations ----
    inits = []
    # Fibonacci spiral is a strong baseline
    inits.append(_fibonacci_spiral(N))
    # Halton sequence for better uniformity
    inits.append(_halton_sequence(N))
    # Random restarts
    for s in range(8):
        inits.append(_random_points(N, seed=100 + s * 17))
    
    # Add some rotated Fibonacci spirals
    for s in range(4):
        rng = np.random.RandomState(200 + s)
        A = rng.randn(3, 3)
        Q, R = np.linalg.qr(A)
        d = np.sign(np.diag(R))
        Q = Q * d
        fib = _fibonacci_spiral(N)
        inits.append(_project_to_sphere(fib @ Q.T))

    best_points = None
    best_energy = np.inf

    # ---- Phase 1: L-BFGS on each init ----
    # Prioritize the best initial guesses (Fibonacci, Halton)
    # We have ~14 inits. 140s / 14 ~ 10s per init.
    # L-BFGS is fast.
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        try:
            cand, cand_energy = _lbfgs_optimize(
                init_pts, max_iter=500, ftol=1e-15, gtol=1e-13, deadline=deadline
            )
            cand = _project_to_sphere(cand)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass

    # ---- Phase 2: Riemannian CG polish on the best candidate ----
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            cand, cand_energy = _riemannian_cg(best_points, max_iter=500, tol=1e-14, deadline=deadline)
            cand = _project_to_sphere(cand)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass

    # ---- Phase 3: Final tight L-BFGS polish ----
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            cand, cand_energy = _lbfgs_optimize(best_points, max_iter=800, ftol=1e-16, gtol=1e-15, deadline=deadline)
            cand = _project_to_sphere(cand)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass

    # ---- Phase 4: Final CG polish ----
    if best_points is not None and time.time() < deadline - 2.0:
        try:
            cand, cand_energy = _riemannian_cg(best_points, max_iter=300, tol=1e-15, deadline=deadline)
            cand = _project_to_sphere(cand)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass

    # ---- Fallback ----
    if best_points is None:
        best_points = _fibonacci_spiral(N)

    best_points = _project_to_sphere(best_points)

    # ---- Validation: ensure no points closer than 1e-9, finite, nonzero ----
    rng = np.random.RandomState(0)
    for _ in range(200):
        ok = True
        for i in range(N):
            if not np.all(np.isfinite(best_points[i])) or np.linalg.norm(best_points[i]) < 1e-12:
                ok = False
                break
        if not ok:
            best_points = _project_to_sphere(best_points + rng.randn(N, 3) * 1e-6)
            continue
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_points[i] - best_points[j])
                if d < 1e-9:
                    best_points[j] += rng.randn(3) * 1e-7
                    ok = False
                    break
            if not ok:
                break
        best_points = _project_to_sphere(best_points)
        if ok:
            break

    return best_points
