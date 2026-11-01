"""Lennard-Jones 55-atom global minimum: exact Mackay icosahedron with multi-radius and full 3-D refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
def _lj_energy(pos):
    """Total LJ energy 4*sum(r^-12 - r^-6)."""
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    pair_e = 4.0 * (inv_r6 * inv_r6 - inv_r6)
    return 0.5 * np.sum(pair_e)

def _lj_energy_grad(pos):
    """Energy and gradient via vectorised pair forces."""
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]  # pos_i - pos_j
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)

    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 * inv_r6
    pair_e = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(pair_e, 0.0)
    energy = 0.5 * np.sum(pair_e)

    # force on atom i in direction (pos_i - pos_j): factor * (pos_i - pos_j)
    # grad_i = sum_j -24*2*... wait compute analytically:
    # E = sum 4(r^-12 - r^-6); dE/dri = 4(-12 r^-13 * ri/r + 6 r^-7 * ri/r)
    # using d(r2)/dri = 2*ri, and chain: dE/dri = (dE/dr2)*(d r2/dri) = factor2 * 2*ri
    # dE/dr2 = 4*(-6 r2^-7 + 3 r2^-4)
    # so gradient contribution from pair (i,j) on i: 8*(-6 r2^-7 + 3 r2^-4) * (pos_i - pos_j)
    # but each pair counted twice, so total grad_i = sum_{j != i} 4*(-12 r2^-7 + 6 r2^-4) * (pos_i - pos_j)
    factor2 = 4.0 * (-12.0 * inv_r2 ** 6 + 6.0 * inv_r2 ** 3)
    np.fill_diagonal(factor2, 0.0)
    grad = np.einsum('ij,ijk->ik', factor2, diff)
    return energy, grad

def _icosahedron_unit_directions():
    """12 unit vectors to icosahedron vertices (centrosymmetric set)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v

def _face_centers_unit(v):
    """12 face-center unit directions for a regular icosahedron with vertices v (unit)."""
    # Build 20 triangular faces by nearest-neighbor triples.
    n = len(v)
    dist = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    # For each vertex i, find two nearest neighbours j,k such that j,k are also neighbours of each other
    faces = []
    used = set()
    for i in range(n):
        order = np.argsort(dist[i])
        j = order[0]
        k = order[1]
        # ensure j,k are close (edge length)
        if dist[j, k] < 1.5 * dist[i, j]:
            for perm in [(i, j, k), (i, k, j)]:
                key = tuple(sorted(perm))
                if key not in used:
                    # Check if the three form a face: all pairwise distances are the same edge length
                    d_ij = dist[i, j]
                    d_ik = dist[i, k]
                    d_jk = dist[j, k]
                    if abs(d_ij - d_ik) < 1e-12 and abs(d_ik - d_jk) < 1e-12:
                        used.add(key)
                        faces.append(perm)
    # For each face, centroid direction
    centers = []
    for (a, b, c) in faces:
        cent = v[a] + v[b] + v[c]
        norm = np.linalg.norm(cent)
        if norm > 1e-12:
            centers.append(cent / norm)
    # Deduplicate directions (up to sign? face centres should be distinct)
    centers = np.array(centers)
    # If we got 20, keep only 12? Actually regular icosahedron has 20 faces. The Mackay icosahedron's
    # outer shell of 12 atoms corresponds to face centers of the icosahedron? There are 20 faces, but
    # only 12 distinct radial directions in an icosahedral cluster? We'll just use all face centroids;
    # to get exactly 12 outer atoms we need the 12 vertices mirrored. Instead, in Mackay icosahedra,
    # the 12 outer atoms lie along the same axes as the 12 inner shell atoms.
    # Thus the true geometry uses the same 12 directions for the outer shell.
    # We'll return the 12 vertex directions as the outer shell directions.
    return v

def _edge_midpoint_unit_dirs(v):
    """30 unit directions toward edge midpoints of the icosahedron (unit vertices v)."""
    n = len(v)
    dist = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    edge_len = np.min(dist[dist > 1e-12])
    dirs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(dist[i, j] - edge_len) < 1e-10:
                mid = (v[i] + v[j]) / 2.0
                nm = np.linalg.norm(mid)
                if nm > 1e-12:
                    dirs.append(mid / nm)
    return np.array(dirs)

def _build_clusters(v_dirs, mid_dirs, radii):
    """Build positions for a 55-atom cluster with given directions and shell radii (lengths 12,30,12)."""
    center = np.zeros((1, 3))
    shell1 = v_dirs * radii[0]
    shell2 = mid_dirs * radii[1]
    shell3 = v_dirs * radii[2]
    return np.vstack([center, shell1, shell2, shell3])

def _optimize_radii(v_dirs, mid_dirs, seed_radii, deadline):
    """Low-dim optimisation of three shell radii."""
    x0 = np.array(seed_radii, dtype=float)
    best_e = None
    best_r = None
    def obj(x):
        if np.any(x <= 0):
            return 1e12
        pos = _build_clusters(v_dirs, mid_dirs, x)
        return _lj_energy(pos)
    def objgrad(x):
        if np.any(x <= 0):
            return 1e12, np.ones(3)
        pos = _build_clusters(v_dirs, mid_dirs, x)
        e, g = _lj_energy_grad(pos)
        # gradient of e w.r.t x: shell1 and shell3 share v_dirs; shell2 uses mid_dirs
        g_center = g[0]
        g1 = np.mean(g[1:13], axis=0)
        g2 = np.mean(g[13:43], axis=0)
        g3 = np.mean(g[43:55], axis=0)
        # d pos1 / d r1 = v_dirs; chain: de/dr1 = sum_i v_dir_i . g_i
        # since g1 is mean, de/dr1 = 12 * np.dot(v_dirs, g1, axis=1) average? Use actual sum.
        de_dr1 = np.sum(g[1:13] * v_dirs, axis=1)
        de_dr2 = np.sum(g[13:43] * mid_dirs, axis=1)
        de_dr3 = np.sum(g[43:55] * v_dirs, axis=1)
        return e, np.concatenate([de_dr1, de_dr2, de_dr3])

    result = minimize(objgrad, x0, jac=True, method='L-BFGS-B',
                       bounds=[(0.5, None), (0.5, None), (0.5, None)],
                       options={'maxiter': 500, 'gtol': 1e-12, 'maxls': 200})
    if result.success and result.fun < (best_e if best_e is not None else np.inf):
        best_e = result.fun
        best_r = result.x.copy()
    return best_r, best_e, time.time()

def _check_min_dist(pos, min_d=0.5):
    diff = pos[:, None, :] - pos[None, :, :]
    d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(d, np.inf)
    return np.min(d)

def _enforce_min_dist(pos, min_d=0.5):
    if _check_min_dist(pos, min_d) >= min_d:
        return pos
    # uniform scale
    d = _check_min_dist(pos, min_d)
    if d < 1e-12:
        return pos
    return pos * (min_d / d + 1e-6)

def _full_relax(pos, deadline, tight=1e-12):
    """Full 3D L-BFGS refinement with adaptive stopping."""
    def f(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_grad(p)
        return e, g.ravel()
    x0 = pos.ravel()
    best_e = _lj_energy(pos)
    best_x = x0.copy()
    stages = [
        {'maxiter': 3000, 'gtol': 1e-10, 'maxls': 300, 'ftol': 1e-15},
        {'maxiter': 5000, 'gtol': 1e-12, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-14, 'maxls': 1000, 'ftol': 1e-17},
    ]
    for opts in stages:
        if time.time() > deadline - 3.0:
            break
        try:
            res = minimize(f, best_x, jac=True, method='L-BFGS-B', options=opts)
            if res.success and res.fun < best_e:
                best_e = res.fun
                best_x = res.x.copy()
        except Exception:
            break
    p = best_x.reshape(-1, 3)
    p = _enforce_min_dist(p)
    # quick final gradient-based polish if time remains
    if time.time() < deadline - 1.5:
        try:
            res = minimize(f, p.ravel(), jac=True, method='L-BFGS-B',
                           options={'maxiter': 1000, 'gtol': 1e-15, 'maxls': 500, 'ftol': 1e-18})
            if res.success:
                p2 = res.x.reshape(-1, 3)
                if _lj_energy(p2) < best_e:
                    p = _enforce_min_dist(p2)
        except Exception:
            pass
    return p, _lj_energy(p)

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin for 150 s wall clock

    v_dirs = _icosahedron_unit_directions()
    mid_dirs = _edge_midpoint_unit_dirs(v_dirs)

    # Seed radii close to known Mackay icosahedron optimum
    r1, r2, r3 = 1.055, 1.635, 1.879
    radii, e_rad, t1 = _optimize_radii(v_dirs, mid_dirs, [r1, r2, r3], deadline)

    # If radii optimisation failed, fall back to seed
    if radii is None:
        radii = np.array([r1, r2, r3])

    pos = _build_clusters(v_dirs, mid_dirs, radii)
    pos = _enforce_min_dist(pos)

    # Full 3-D relaxation from this low-symmetry-breaking starting point
    if time.time() < deadline - 5.0:
        pos, e_full = _full_relax(pos, deadline, tight=1e-14)

    # Final safety: if minimum distance is violated, restore best valid configuration
    if not _check_min_dist(pos, 0.5):
        pos = _build_clusters(v_dirs, mid_dirs, radii)
        pos = _enforce_min_dist(pos)
        if time.time() < deadline - 2.0:
            pos, _ = _full_relax(pos, deadline)
        if not _check_min_dist(pos, 0.5):
            pos = _enforce_min_dist(pos)

    return pos
# EVOLVE-BLOCK-END
