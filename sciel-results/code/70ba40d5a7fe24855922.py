"""LJ 55-atom cluster: concentric-shell Mackay icosahedron with analytic radius
optimization, then L-BFGS polishing + stochastic multi-start refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

def _lj_energy_and_grad(pos):
    """Compute Lennard-Jones energy and gradient for a set of positions."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r2_safe = np.maximum(r2, 1e-12)

    r6 = r2_safe ** (-3)
    r12 = r6 ** 2

    # E = 4 * sum_{i<j} (r^-12 - r^-6); matrix double-counts -> E = 2*sum(r12-r6)
    E = 2.0 * np.sum(r12 - r6)

    # dE/dpos_i = 24 * sum_{j!=i} (2*r2^-7 - r2^-4) * (pos_i - pos_j)
    coeff = 24.0 * (2.0 * r2_safe ** (-7) - r2_safe ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)

    return E, grad


def _icosahedron_frames():
    """Return vertices (12), edge-midpoint directions (30), face-center directions (20)
    of a regular icosahedron, all normalized to unit length. Edge/face sets are
    derived from the vertex adjacency so the geometry is exact."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    # adjacency via nearest-neighbor distance
    d = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_d = d.min()
    tol = min_d * 1.01
    edges = [(i, j) for i in range(12) for j in range(i + 1, 12) if d[i, j] < tol]
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)

    edge_dirs = np.array([(verts[i] + verts[j]) / 2.0 for i, j in edges])
    edge_dirs /= np.linalg.norm(edge_dirs, axis=1, keepdims=True)

    faces = set()
    for i in range(12):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and i in adj[k]:
                    faces.add(tuple(sorted((i, j, k))))
    face_dirs = np.array([(verts[a] + verts[b] + verts[c]) / 3.0 for a, b, c in faces])
    face_dirs /= np.linalg.norm(face_dirs, axis=1, keepdims=True)

    return verts, edge_dirs, face_dirs


def _mackay_unit_dirs():
    """Unit-direction set for the 55-atom Mackay icosahedron:
    1 center + 12 shell1 + 30 shell2 + 12 shell3 (a symmetric 12-subset of 20 faces)."""
    verts, edge_dirs, face_dirs = _icosahedron_frames()
    # pick 12 face centers: the 20 faces come in 10 antipodal pairs; take one from
    # each pair -> 10, then we need 12. Instead, select 12 that are mutually
    # non-antipodal and well spread. Simplest robust choice: take faces whose
    # centroid has positive first coordinate, else first 12.
    # Use antipodal-pair selection to get a balanced 10, then add 2 more spread ones.
    chosen = []
    used = set()
    for idx in range(face_dirs.shape[0]):
        if idx in used:
            continue
        # find antipodal partner
        ant = None
        for jdx in range(face_dirs.shape[0]):
            if jdx != idx and not np.any(np.isclose(face_dirs[idx], face_dirs[jdx])):
                if np.dot(face_dirs[idx], face_dirs[jdx]) < -0.999:
                    ant = jdx; break
        chosen.append(idx); used.add(idx)
        if ant is not None:
            used.add(ant)
    chosen = chosen[:12]
    if len(chosen) < 12:
        for idx in range(face_dirs.shape[0]):
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) == 12:
                break
    face_subset = face_dirs[chosen[:12]]
    return np.vstack([verts, edge_dirs, face_subset])  # 54 directions (excl. center)


def _optimize_shell_radii(unit_dirs, max_iter=200):
    """Optimize the 3 shell radii (r1, r2, r3) to minimize LJ energy of the
    concentric-shell Mackay (center at origin fixed). Returns optimal radii."""
    r10 = 1.0
    r20 = r10 * np.sqrt(5.0) / 2.0
    r30 = r10 * np.sqrt(5.0 / 3.0)

    def shell_energy(radii):
        r1, r2, r3 = radii
        s1 = unit_dirs[:12] * r1
        s2 = unit_dirs[12:42] * r2
        s3 = unit_dirs[42:] * r3
        pos = np.vstack([np.zeros((1, 3)), s1, s2, s3])
        E, _ = _lj_energy_and_grad(pos)
        return E

    res = minimize(shell_energy, np.array([r10, r20, r30]), method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
    return res.x


def _build_mackay(radii, n_atoms=55):
    """Build the 55-atom Mackay from unit directions and 3 radii."""
    unit_dirs = _mackay_unit_dirs()
    r1, r2, r3 = radii
    s1 = unit_dirs[:12] * r1
    s2 = unit_dirs[12:42] * r2
    s3 = unit_dirs[42:] * r3
    return np.vstack([np.zeros((1, 3)), s1, s2, s3])[:n_atoms]


def _cubic_lattice_fragment(n_atoms=55, spacing=1.12):
    side = int(np.ceil(n_atoms ** (1 / 3)))
    pts = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                pts.append([i, j, k])
    cx = (side - 1) / 2.0
    pts.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cx)**2 + (p[2]-cx)**2)
    return np.array(pts[:n_atoms], dtype=float) * spacing


def _ensure_min_dist(pos, min_d=0.5):
    """Nudge any pair closer than min_d apart (light pass)."""
    pos = pos.copy()
    for _ in range(3):
        moved = False
        for i in range(pos.shape[0]):
            for j in range(i + 1, pos.shape[0]):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    if dist < 1e-9:
                        d = np.array([1.0, 0.0, 0.0])
                    push = (min_d - dist) / dist * d
                    pos[i] += 0.5 * push
                    pos[j] -= 0.5 * push
                    moved = True
        if not moved:
            break
    return pos


def _relax(pos, deadline, maxiter=3000):
    """L-BFGS local relaxation with time guard."""
    try:
        res = minimize(_lj_energy_and_grad, pos, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'gtol': 1e-9,
                                'ftol': 1e-12, 'maxfun': 50000})
        if time.time() > deadline:
            return None
        if res.success or np.isfinite(res.fun):
            return res.x, res.fun
    except Exception:
        pass
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin under 150s kill

    best_pos = None
    best_energy = np.inf

    # --- Stage 1: analytic radius-optimized Mackay icosahedron (cheap, near-exact) ---
    try:
        unit_dirs = _mackay_unit_dirs()
        radii = _optimize_shell_radii(unit_dirs, max_iter=150)
        mackay = _build_mackay(radii, 55)
        mackay = _ensure_min_dist(mackay)
        out = _relax(mackay, deadline)
        if out is not None:
            best_pos, best_energy = out
    except Exception:
        pass

    # --- Stage 2: stochastic multi-start refinement around the best Mackay ---
    np.random.seed(123)
    base = best_pos if best_pos is not None else _cubic_lattice_fragment(55, 1.12)

    for scale in [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]:
        if time.time() > deadline:
            break
        for trial in range(2):
            if time.time() > deadline:
                break
            pert = base + np.random.randn(55, 3) * scale
            pert = _ensure_min_dist(pert)
            out = _relax(pert, deadline, maxiter=2000)
            if out is not None:
                pos, e = out
                if e < best_energy:
                    best_energy = e
                    best_pos = pos
                    base = pos  # refine around improving solution

    # --- Stage 3: fallback cubic lattice if nothing worked ---
    if best_pos is None or best_energy > -100:
        cubic = _cubic_lattice_fragment(55, 1.12)
        out = _relax(cubic, deadline)
        if out is not None and out[1] < best_energy:
            best_energy = out[1]
            best_pos = out[0]
        if best_pos is None:
            best_pos = cubic

    if best_pos is None:
        return _cubic_lattice_fragment(55, 1.12)

    return best_pos

# EVOLVE-BLOCK-END
