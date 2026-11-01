"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
import math
import random

DEADLINE_FRACTION = 0.90  # safety margin
TIME_LIMIT = 150.0


def _energy_and_grad(X):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    and its gradient w.r.t. X (shape (N,3)).
    Uses np.sum for speed in the optimization loop; the evaluator uses math.fsum for final scoring.
    """
    dots = X @ X.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)

    iu = np.triu_indices(N, 1)
    log_d2 = np.log(d2[iu])
    E = -0.5 * np.sum(log_d2)

    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    row_sums = inv_d2.sum(axis=1)
    term2 = inv_d2 @ X
    grad = X * row_sums[:, None] - term2

    return E, grad


def _retract(X, step):
    """Riemannian retraction: move along tangent direction, then project back to sphere."""
    step_dot_X = np.sum(step * X, axis=1, keepdims=True)
    tangent_step = step - step_dot_X * X
    X_new = X + tangent_step
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    X_new = X_new / norms
    return X_new


def _fibonacci_sphere(n, seed_offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral with optional phase offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed_offset
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedron_based(n, rng):
    """
    Generate n points by subdividing an icosahedron and projecting to sphere.
    This gives a more uniform base distribution than Fibonacci for some configurations.
    """
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Icosahedron vertices (12 points)
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    # Icosahedron faces (20 triangles)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]

    # Subdivide each face to get ~n total points
    # We need roughly n points total; each face gets n/20 points
    pts_per_face = max(1, n // 20)
    all_pts = []

    for fi, (a, b, c) in enumerate(faces):
        va, vb, vc = verts[a], verts[b], verts[c]
        for i in range(pts_per_face):
            for j in range(pts_per_face - i):
                k = pts_per_face - i - j
                if i + j + k == 0:
                    continue
                # Barycentric coordinates
                t1 = i / pts_per_face
                t2 = j / pts_per_face
                # Point on triangle
                p = (1 - t1 - t2) * va + t1 * vb + t2 * vc
                # Add small random perturbation for diversity
                p = p + rng.normal(0, 0.01, size=3)
                pn = np.linalg.norm(p)
                if pn > 1e-15:
                    all_pts.append(p / pn)

    # If we have more points than needed, subsample
    if len(all_pts) > n:
        idx = rng.choice(len(all_pts), size=n, replace=False)
        all_pts = [all_pts[i] for i in idx]
    elif len(all_pts) < n:
        # Pad with random points
        while len(all_pts) < n:
            p = rng.normal(0, 1, size=3)
            pn = np.linalg.norm(p)
            all_pts.append(p / pn)

    result = np.array(all_pts[:n])
    # Normalize
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return result / norms


def _random_sphere(n, rng):
    """Generate n random points on the sphere."""
    theta = rng.uniform(0, 2 * np.pi, size=n)
    cos_phi = rng.uniform(-1, 1, size=n)
    sin_phi = np.sqrt(1 - cos_phi ** 2)
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)
    z = cos_phi
    return np.column_stack([x, y, z])


def _optimize_lbfgs(X0, max_iter=5000, tol=1e-13, deadline=None):
    """
    L-BFGS optimization on the sphere using retraction with Wolfe line search.
    Increased memory to 30, tighter tolerance.
    """
    X = X0.copy()
    n = X.shape[0]

    m = 30
    s_list = []
    y_list = []

    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        g_flat = g.ravel()
        g_norm = np.linalg.norm(g_flat)
        if g_norm < tol:
            break

        if len(s_list) == 0:
            direction = -g
        else:
            q = g_flat.copy()
            alphas = []
            rhos = []

            for i in range(len(s_list) - 1, -1, -1):
                s_i = s_list[i]
                y_i = y_list[i]
                sy = np.dot(s_i, y_i)
                if abs(sy) < 1e-15:
                    rho_i = 0.0
                else:
                    rho_i = 1.0 / sy
                rhos.append(rho_i)
                alpha_i = rho_i * np.dot(s_i, q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i

            r = q

            for i in range(len(s_list)):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[len(s_list) - 1 - i]
                alpha_i = alphas[len(s_list) - 1 - i]
                beta_i = rho_i * np.dot(y_i, r)
                r = r + (alpha_i - beta_i) * s_i

            direction = r.reshape(n, 3)

        # Wolfe line search
        g_flat_dir = np.dot(g_flat, direction.ravel())
        if g_flat_dir >= 0:
            direction = -g
            g_flat_dir = np.dot(g_flat, direction.ravel())

        c1 = 1e-4
        c2 = 0.9
        step_size = 1.0

        X_trial = _retract(X, step_size * direction)
        E_trial, g_trial = _energy_and_grad(X_trial)

        max_ls = 40
        ls_success = False
        for ls in range(max_ls):
            if deadline is not None and time.time() > deadline:
                break
            armijo_ok = E_trial <= E + c1 * step_size * g_flat_dir
            g_trial_dir = np.dot(g_trial.ravel(), direction.ravel())
            curvature_ok = g_trial_dir >= c2 * g_flat_dir

            if armijo_ok:
                ls_success = True
                break

            step_size *= 0.5
            X_trial = _retract(X, step_size * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)

        if step_size < 1e-15:
            break

        if not ls_success and E_trial > E:
            step_size = 0.01
            X_trial = _retract(X, step_size * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)
            if E_trial > E:
                break

        s_vec = (X_trial - X).ravel()
        y_vec = (g_trial - g).ravel()

        sy = np.dot(s_vec, y_vec)
        if sy > 1e-12:
            s_list.append(s_vec)
            y_list.append(y_vec)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)

        X = X_trial
        E = E_trial
        g = g_trial

        if E < best_E:
            best_E = E
            best_X = X.copy()

    return best_X, best_E


def _polish(X0, max_iter=10000, tol=1e-14, deadline=None):
    """
    High-precision conjugate gradient polish with adaptive step size.
    """
    X = X0.copy()
    n = X.shape[0]
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    step_size = 1e-3
    beta = 0.0
    prev_g = g.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        g_flat = g.ravel()
        g_norm = np.linalg.norm(g_flat)
        if g_norm < tol:
            break
        
        # Conjugate gradient direction
        if beta == 0.0 or it == 0:
            direction = -g
        else:
            direction = -g + beta * (direction + g)
        
        # Project to tangent space
        dir_dot_X = np.sum(direction * X, axis=1, keepdims=True)
        direction = direction - dir_dot_X * X
        
        # Backtracking line search
        E_new = None
        X_trial_final = None
        g_new = None
        
        for ls in range(40):
            if deadline is not None and time.time() > deadline:
                break
            X_trial = _retract(X, step_size * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)
            
            if E_trial < E:
                E_new = E_trial
                g_new = g_trial
                X_trial_final = X_trial
                break
            step_size *= 0.5
        
        if E_new is None:
            step_size *= 0.1
            if step_size < 1e-20:
                break
            beta = 0.0
            continue
        
        # Update beta (Polak-Ribiere)
        g_dot_prev = np.dot(g.ravel(), prev_g.ravel())
        prev_g_norm_sq = np.dot(prev_g.ravel(), prev_g.ravel())
        if prev_g_norm_sq > 1e-30:
            beta = max(0.0, g_dot_prev / prev_g_norm_sq)
        
        prev_g = g.copy()
        X = X_trial_final
        E = E_new
        g = g_new
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
        
        # Adapt step size
        if E_new < E - 1e-16:
            step_size = min(step_size * 1.3, 1e-2)
    
    return best_X, best_E


def _high_precision_polish(X0, max_iter=5000, tol=1e-15, deadline=None):
    """
    Very tight tolerance polish for final convergence.
    """
    X = X0.copy()
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()

    step_size = 1e-4

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        g_flat = g.ravel()
        g_norm = np.linalg.norm(g_flat)
        if g_norm < tol:
            break

        direction = -g / g_norm

        for ls in range(50):
            if deadline is not None and time.time() > deadline:
                break
            X_trial = _retract(X, step_size * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)

            if E_trial < E:
                X = X_trial
                E = E_trial
                g = g_trial
                if E < best_E:
                    best_E = E
                    best_X = X.copy()
                step_size = min(step_size * 1.2, 1e-3)
                break
            step_size *= 0.5

        if step_size < 1e-25:
            break

    return best_X, best_E


def _steepest_descent_polish(X0, max_iter=10000, tol=1e-16, deadline=None):
    """
    Ultra-tight steepest descent polish with adaptive step size.
    Used as the final refinement pass.
    """
    X = X0.copy()
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()

    step_size = 1e-5

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        g_flat = g.ravel()
        g_norm = np.linalg.norm(g_flat)
        if g_norm < tol:
            break

        # Project gradient to tangent space
        g_dot_X = np.sum(g * X, axis=1, keepdims=True)
        tangent_g = g - g_dot_X * X
        tg_norm = np.linalg.norm(tangent_g.ravel())
        if tg_norm < tol:
            break

        direction = -tangent_g / tg_norm

        for ls in range(60):
            if deadline is not None and time.time() > deadline:
                break
            X_trial = _retract(X, step_size * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)

            if E_trial < E - 1e-18:
                X = X_trial
                E = E_trial
                g = g_trial
                if E < best_E:
                    best_E = E
                    best_X = X.copy()
                step_size = min(step_size * 1.15, 1e-4)
                break
            step_size *= 0.5

        if step_size < 1e-30:
            break

    return best_X, best_E


def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + TIME_LIMIT * DEADLINE_FRACTION

    rng = np.random.RandomState(42)

    # Generate diverse starting points
    starts = []

    # Fibonacci with different phase offsets
    for offset in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        starts.append(_fibonacci_sphere(N, seed_offset=offset))

    # Icosahedron-based starts
    for seed in [1, 2, 3]:
        rng_ico = np.random.RandomState(seed)
        starts.append(_icosahedron_based(N, rng_ico))

    # Random starts
    for _ in range(4):
        starts.append(_random_sphere(N, rng))

    # Perturbed Fibonacci
    for _ in range(4):
        base = _fibonacci_sphere(N)
        noise = rng.normal(0, 0.08, size=(N, 3))
        perturbed = base + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / np.maximum(norms, 1e-15)
        starts.append(perturbed)

    # Optimize each start and keep the best
    best_pts = None
    best_energy = float('inf')

    for i, start in enumerate(starts):
        if deadline is not None and time.time() > deadline:
            break

        # Allocate time per start with a minimum floor
        remaining = deadline - time.time()
        num_remaining = max(1, len(starts) - i)
        time_per_start = max(remaining / num_remaining, 2.0)
        start_deadline = time.time() + min(time_per_start, remaining)

        # L-BFGS optimization
        pts, e = _optimize_lbfgs(start, max_iter=4000, tol=1e-13, deadline=start_deadline)

        # Polish
        pts, e = _polish(pts, max_iter=3000, tol=1e-14, deadline=start_deadline)

        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()

    # Final high-precision polish on the best result
    if best_pts is not None:
        best_pts, best_energy = _high_precision_polish(best_pts, max_iter=5000, tol=1e-15, deadline=deadline)
        # Final ultra-tight steepest descent
        best_pts, best_energy = _steepest_descent_polish(best_pts, max_iter=10000, tol=1e-16, deadline=deadline)

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms

    # Verify validity
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)

    # Check minimum distance
    min_d = 1e18
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_d:
                min_d = d
    if min_d < 1e-9:
        return _fibonacci_sphere(N)

    return best_pts
# EVOLVE-BLOCK-END
