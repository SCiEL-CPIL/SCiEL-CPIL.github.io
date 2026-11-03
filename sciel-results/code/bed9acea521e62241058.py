"""Logarithmic energy minimization of 150 points on the unit sphere.

Strategy (combined):
  - Multi-start local optimization: Fibonacci spiral, geodesic dome, random,
    anisotropic and perturbed Fibonacci starts (diversity of basins).
  - Basin hopping with Riemannian perturbations: random rotations (exponential
    map) that respect the sphere geometry, replacing Euclidean projection noise.
  - Riemannian L-BFGS polish: optimize in tangent space (removes the sphere
    constraint exactly), then map back via the exponential map.
  - Final high-precision polish pass.

All work is bounded by an explicit wall-clock deadline with a safety margin.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Energy and gradient
# ---------------------------------------------------------------------------
def _energy_and_grad(pts):
    """Logarithmic energy and tangent-space gradient.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    grad_i (Cartesian) = sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2, projected to tangent.
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)

    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())

    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    # grad_i = sum_j (x_i - x_j)/|x_i-x_j|^2 = x_i*sum_j inv - sum_j inv*x_j
    row_sums = np.sum(inv_dist2, axis=1)
    grad = pts * row_sums[:, np.newaxis] - inv_dist2 @ pts
    # Project to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    return energy, grad


def _energy_only(pts):
    """Compute only the energy (fsum for precision)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    return -math.fsum(np.log(dists).tolist())


def _project_to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


# ---------------------------------------------------------------------------
# Riemannian (exponential map) utilities
# ---------------------------------------------------------------------------
def _exp_map_tangent(pts, tangent):
    """Exponential map on the sphere: move pts along tangent vectors.
    exp_x(v) = cos(|v|) x + sin(|v|) (v/|v|), a rotation by angle |v| about v/|v|.
    """
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    safe_norm = np.where(norm > 1e-15, norm, 1.0)
    u = tangent / safe_norm
    c = np.cos(norm)
    s = np.sin(norm)
    out = c * pts + s * u
    return _project_to_sphere(out)


def _log_map_tangent(pts, target):
    """Logarithmic map: tangent vector at pts that maps to target on the sphere."""
    cos_theta = np.sum(pts * target, axis=1, keepdims=True)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    safe_sin = np.where(sin_theta > 1e-12, sin_theta, 1.0)
    v = (theta / safe_sin) * (target - cos_theta * pts)
    v = np.where((theta > 1e-12)[:, np.newaxis], v, 0.0)
    return v


# ---------------------------------------------------------------------------
# Riemannian L-BFGS polish (optimizes in tangent space)
# ---------------------------------------------------------------------------
def _riemannian_lbfgs_polish(pts, deadline, maxiter=400, ftol=1e-16, gtol=1e-13):
    """Polish a configuration using L-BFGS-B on tangent-space coordinates."""
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
                                'gtol': gtol, 'maxls': 30})
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
    return best_pts, best_energy


# ---------------------------------------------------------------------------
# Cartesian L-BFGS polish (fallback / final)
# ---------------------------------------------------------------------------
def _lbfgs_polish(pts, deadline, maxiter=500, ftol=1e-16, gtol=1e-14):
    """L-BFGS-B polish using flat Cartesian coordinates with projection."""
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol,
                                'gtol': gtol, 'maxls': 30})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_res = _energy_only(p_res)
        if e_res < best_energy and np.all(np.isfinite(p_res)):
            best_energy = e_res
            best_pts = p_res.copy()
    except Exception:
        pass
    return best_pts, best_energy


# ---------------------------------------------------------------------------
# Initial configurations
# ---------------------------------------------------------------------------
def _fibonacci_spiral(n, offset=0.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _geodesic_dome(n, seed=0):
    """Generate points via recursive subdivision of an icosahedron, then
    relax to equal spacing. Provides a different structure than Fibonacci."""
    # Icosahedron vertices
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    verts = _project_to_sphere(verts)

    def subdivide(tri):
        """Subdivide a triangle into 4, projecting midpoints to sphere."""
        a, b, c = tri
        ab = _project_to_sphere((a + b) / 2.0)
        bc = _project_to_sphere((b + c) / 2.0)
        ca = _project_to_sphere((c + a) / 2.0)
        return [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]

    # Build initial triangle list
    edges = []
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.5:  # edge threshold for icosahedron
                edges.append((i, j))
    # Build faces (triangles) from edges
    faces = []
    for i in range(len(verts)):
        nbrs = [j for (a, b) in edges if (a == i and b == j) or (a == j and b == i)]
        for j in range(len(nbrs)):
            for k in range(j + 1, len(nbrs)):
                if (verts[nbrs[j]] - verts[nbrs[k]]).dot(verts[nbrs[j]] - verts[nbrs[k]]) < 1e-6:
                    faces.append((i, nbrs[j], nbrs[k]))

    # Subdivide until we have enough points
    current_faces = [[verts[i], verts[j], verts[k]] for (i, j, k) in faces]
    while True:
        # count unique vertices
        pts_set = set()
        for tri in current_faces:
            for v in tri:
                pts_set.add(tuple(np.round(v, 12)))
        if len(pts_set) >= n:
            break
        new_faces = []
        for tri in current_faces:
            new_faces.extend(subdivide(tri))
        current_faces = new_faces
        if len(current_faces) > 200000:
            break

    # Collect unique points
    pts_list = []
    seen = set()
    for tri in current_faces:
        for v in tri:
            key = tuple(np.round(v, 12))
            if key not in seen:
                seen.add(key)
                pts_list.append(v)
    pts = np.array(pts_list)
    # Take first n
    if len(pts) >= n:
        pts = pts[:n]
    else:
        pts = np.vstack([pts, _random_sphere(n - len(pts), seed)])
    pts = _project_to_sphere(pts)
    return pts


# ---------------------------------------------------------------------------
# Riemannian perturbation (random rotations)
# ---------------------------------------------------------------------------
def _riemannian_perturb(pts, step, rng):
    """Perturb pts by random tangent displacements (random rotations)."""
    tangent = rng.randn(N, 3) * step
    # Project to tangent
    dot = np.sum(tangent * pts, axis=1, keepdims=True)
    tangent = tangent - dot * pts
    return _exp_map_tangent(pts, tangent)


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Diverse starting points
    starts = [_fibonacci_spiral(N)]
    try:
        starts.append(_geodesic_dome(N))
    except Exception:
        pass
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        starts.append(_random_sphere(N, seed))
    for scale in [0.8, 1.2, 0.6, 1.5]:
        Xf = _fibonacci_spiral(N)
        Xs = Xf.copy()
        Xs[:, 0] *= scale
        starts.append(_project_to_sphere(Xs))
    Xf = _fibonacci_spiral(N)
    rng0 = np.random.RandomState(999)
    noise = rng0.randn(N, 3) * 0.1
    starts.append(_project_to_sphere(Xf + noise))

    best_X = None
    best_E = np.inf

    for i, X0 in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 12.0:
            break

        X_opt, E_opt = _lbfgs_polish(X0, deadline - 8.0)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()

        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 8.0:
            break

    if best_X is None:
        best_X = _fibonacci_spiral(N)
        best_E = _energy_only(best_X)

    # Basin hopping with Riemannian perturbations on the best config
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 20.0:
        rng = np.random.RandomState(12345)
        step = 0.15
        best_X, best_E = _riemannian_lbfgs_polish(best_X, deadline - 6.0)
        for it in range(200):
            elapsed = time.time() - start_time
            remaining = deadline - elapsed
            if remaining < 12.0:
                break
            cand = _riemannian_perturb(best_X, step, rng)
            cand, cand_E = _lbfgs_polish(cand, deadline - 8.0)
            if cand_E < best_E:
                best_E = cand_E
                best_X = cand.copy()
                # cool down slightly
                step = max(step * 0.95, 0.05)
            else:
                step = min(step * 1.05, 0.3)

    # Final Riemannian polish for high precision
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 6.0:
        best_X, best_E = _riemannian_lbfgs_polish(best_X, deadline - 2.0)

    # Ensure validity: unit sphere, no points too close, finite
    best_X = _project_to_sphere(best_X)
    if not np.all(np.isfinite(best_X)):
        best_X = _fibonacci_spiral(N)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < 1e-9:
                rng = np.random.RandomState(i * 1000 + j)
                best_X[j] += rng.randn(3) * 1e-8
                best_X[j] = _project_to_sphere(best_X[j])

    return best_X
# EVOLVE-BLOCK-END
