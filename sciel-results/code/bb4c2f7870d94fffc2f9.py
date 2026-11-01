"""Lennard-Jones 55-atom cluster: Mackay icosahedron with optimized shell radii + basin hopping."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
LJ_MIN = 2.0 ** (1.0 / 6.0)  # 1.122462... pair-potential minimum distance

# EVOLVE-BLOCK-START

def lj_energy(pos):
    """Total LJ energy 4*sum_{i<j}(r^-12 - r^-6)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    return 2.0 * (r12 - r6).sum()

def lj_energy_grad(pos):
    """LJ energy and gradient (each pair counted once)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 2.0 * (r12 - r6).sum()
    # dE/dpos_i = 24 * sum_{j!=i} (2*r^-14 - r^-8) * (pos_i - pos_j)
    # in r2: r^-14 = r2^-7, r^-8 = r2^-4
    coeff = 24.0 * (2.0 * r2 ** (-7) - r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def _icosahedron_frames():
    """Return 12 unit vertex vectors, 30 unit edge-midpoint vectors, 12 unit face-center vectors."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    dot_adj = 1.0 / (phi * phi)  # dot product of adjacent unit vertices
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(np.dot(verts[i], verts[j]) - dot_adj) < 1e-6:
                edges.append((i, j))

    edge_mids = np.array([(verts[i] + verts[j]) / 2.0 for i, j in edges])
    edge_mids = edge_mids / np.linalg.norm(edge_mids, axis=1, keepdims=True)

    # Faces: triangles of mutually adjacent vertices (20 faces).
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    faces = set()
    for i in range(12):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and i in adj[k]:
                    faces.add(tuple(sorted((i, j, k))))
    face_list = list(faces)[:20]
    face_c = np.array([(verts[a] + verts[b] + verts[c]) / 3.0 for a, b, c in face_list])
    face_c = face_c / np.linalg.norm(face_c, axis=1, keepdims=True)
    return verts, edge_mids, face_c

def make_mackay_icosahedron(n_atoms=55, r1=LJ_MIN, r_edge=LJ_MIN*1.75, r_face=LJ_MIN*1.45):
    """Two-shell Mackay icosahedron: 1 + 12 + 30 + 12 = 55 atoms."""
    verts, edge_mids, face_c = _icosahedron_frames()
    atoms = [np.zeros(3)]
    atoms.extend(list(verts * r1))
    atoms.extend(list(edge_mids * r_edge))
    atoms.extend(list(face_c * r_face))
    pos = np.array(atoms[:n_atoms])
    pos -= np.mean(pos, axis=0)
    return pos

def make_cubic_lattice():
    """Cubic lattice fragment at the LJ minimum spacing."""
    side = int(np.ceil(N ** (1.0 / 3.0)))
    pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    c = (side - 1) / 2.0
    pts.sort(key=lambda p: (p[0] - c) ** 2 + (p[1] - c) ** 2 + (p[2] - c) ** 2)
    return np.array(pts[:N], dtype=float) * LJ_MIN

def _valid(pos):
    """Check minimum-pair-distance >= 0.5 and finiteness."""
    if not np.all(np.isfinite(pos)):
        return False
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1e18)
    return np.min(r2) >= 0.5 ** 2

def try_relax(positions, max_iter=3000):
    """L-BFGS relaxation. Returns (pos, energy)."""
    best_energy, best_pos = lj_energy(positions), positions.copy()
    try:
        res = minimize(
            lj_energy_grad, positions, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-9,
                     'maxfun': max_iter*2, 'maxls': 20},
        )
        if np.all(np.isfinite(res.fun)) and res.fun < best_energy:
            best_energy, best_pos = res.fun, res.x.copy()
    except Exception:
        pass
    return best_pos, best_energy

def basin_hopping(pos, n_steps=20, kick_scale=0.3, rng=None):
    """Simple basin hopping: kick + relax, accept if lower energy."""
    if rng is None:
        rng = np.random.default_rng()
    current_pos = pos.copy()
    current_energy = lj_energy(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    for _ in range(n_steps):
        # Random kick
        kick = rng.normal(0.0, kick_scale, current_pos.shape)
        new_pos = current_pos + kick
        if not _valid(new_pos):
            continue
        # Relax
        new_pos, new_energy = try_relax(new_pos, max_iter=1000)
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
    
    return best_pos, best_energy

def construct():
    """Return Cartesian coordinates (N,3) of the 55-atom LJ cluster."""
    start = time.time()
    deadline = start + 135.0   # hard wall-clock guard

    def remaining():
        return deadline - time.time()

    best_pos, best_energy = None, np.inf
    rng = np.random.default_rng(42)

    def consider(pos, do_bh=False, bh_steps=0):
        nonlocal best_pos, best_energy
        if pos is None or not _valid(pos):
            return
        pos, e = try_relax(pos)
        if e < best_energy:
            best_energy, best_pos = e, pos.copy()
        if do_bh and bh_steps > 0 and remaining() > 20.0:
            pos_bh, e_bh = basin_hopping(pos, n_steps=bh_steps, rng=rng)
            if e_bh < best_energy:
                best_energy, best_pos = e_bh, pos_bh.copy()

    # 1) Mackay icosahedron with various shell radii
    radii_configs = [
        (LJ_MIN, LJ_MIN*1.75, LJ_MIN*1.45),
        (LJ_MIN, LJ_MIN*1.70, LJ_MIN*1.40),
        (LJ_MIN, LJ_MIN*1.80, LJ_MIN*1.50),
        (LJ_MIN*0.98, LJ_MIN*1.72, LJ_MIN*1.42),
        (LJ_MIN*1.02, LJ_MIN*1.78, LJ_MIN*1.48),
    ]
    for r1, r_edge, r_face in radii_configs:
        if remaining() < 25.0:
            break
        try:
            pos = make_mackay_icosahedron(N, r1, r_edge, r_face)
            consider(pos, do_bh=True, bh_steps=10)
        except Exception:
            pass

    # 2) Cubic lattice fragment
    if remaining() > 25.0:
        try:
            consider(make_cubic_lattice())
        except Exception:
            pass

    # 3) Random perturbations of the best Mackay geometry
    mackay = make_mackay_icosahedron(N)
    for amp in (0.05, 0.15, 0.30, 0.50):
        if remaining() < 18.0:
            break
        pert = mackay + rng.normal(0.0, amp, mackay.shape)
        consider(pert)

    # 4) Basin hopping from the best found so far
    if best_pos is not None and remaining() > 30.0:
        try:
            pos_bh, e_bh = basin_hopping(best_pos, n_steps=30, kick_scale=0.4, rng=rng)
            if e_bh < best_energy:
                best_energy, best_pos = e_bh, pos_bh.copy()
        except Exception:
            pass

    # 5) Additional basin hopping from Mackay with larger kicks
    if remaining() > 25.0:
        try:
            pos_bh, e_bh = basin_hopping(mackay, n_steps=20, kick_scale=0.5, rng=rng)
            if e_bh < best_energy:
                best_energy, best_pos = e_bh, pos_bh.copy()
        except Exception:
            pass

    if best_pos is not None:
        return best_pos
    return make_cubic_lattice()

# EVOLVE-BLOCK-END
