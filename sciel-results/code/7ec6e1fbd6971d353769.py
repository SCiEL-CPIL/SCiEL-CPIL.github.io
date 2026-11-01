"""Lennard-Jones 55-atom cluster: build a Mackay icosahedron and polish with L-BFGS."""
import time
import numpy as np
from scipy.optimize import minimize

N = 55
_START = time.time()
_DEADLINE = 150.0 - 15.0  # safety margin before the 150 s wall-clock kill


def _lj_energy(pos):
    """Total LJ energy 4*sum(r^-12 - r^-6) over all pairs (vectorised)."""
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    return 4.0 * np.sum(inv_r6 * (inv_r6 - 1.0))


def _lj_energy_grad(pos):
    """Energy and gradient for L-BFGS."""
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 * inv_r6
    # per-pair contribution to gradient: dE/dr2 = 4*(-12 r^-14 + 6 r^-8) = 4*(-12 inv_r2^6 + 6 inv_r2^3)
    # dE/dpos_i = 2 * (dE/dr2) * (pos_i - pos_j)  summed over j
    dEdr2 = 4.0 * (-12.0 * inv_r12 + 6.0 * inv_r6)
    dEdr2 = dEdr2[:, None] * diff
    grad = 2.0 * np.sum(dEdr2, axis=1)
    energy = 4.0 * np.sum(inv_r6 * (inv_r6 - 1.0))
    return energy, grad


def _mulliken_icosahedron(scale_inner, scale_outer):
    """Two-shell Mackay icosahedron.
    Inner shell: 12 icosahedral vertices at radius R1.
    Outer shell: 12 vertices at R2 + 30 face-centers (unit) at R2.
    Center: 1 atom.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 icosahedral vertices (unit)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    # 30 face centers: each face is (v_i, v_j, v_k) with the three mutually-adjacent vertices.
    # Build from pairs of vertices whose dot product is 1/phi (adjacent), then the face normal.
    face_centers = []
    seen = set()
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            d = verts[i] @ verts[j]
            if abs(d - 1.0 / phi) < 1e-9:
                # find the third vertex k adjacent to both i and j
                for k in range(12):
                    if k == i or k == j:
                        continue
                    if abs(verts[i] @ verts[k] - 1.0 / phi) < 1e-9 and abs(verts[j] @ verts[k] - 1.0 / phi) < 1e-9:
                        # face normal = normalized (vi + vj + vk)
                        n = verts[i] + verts[j] + verts[k]
                        n = n / np.linalg.norm(n)
                        key = tuple(np.round(n, 6))
                        if key not in seen:
                            seen.add(key)
                            face_centers.append(n)
                        break
    face_centers = np.array(face_centers)
    assert len(face_centers) == 30, f"expected 30 face centers, got {len(face_centers)}"

    center = np.zeros(3)
    inner = verts * scale_inner
    outer_verts = verts * scale_outer
    outer_faces = face_centers * scale_outer
    pos = np.vstack([center[None, :], inner, outer_verts, outer_faces])
    assert pos.shape[0] == N, f"expected {N} atoms, got {pos.shape[0]}"
    return pos


def _relax(pos):
    """L-BFGS local relaxation; returns (pos, energy)."""
    try:
        res = minimize(_lj_energy_grad, pos, method='L-BFGS-B',
                       jac=True, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
        return res.x, float(res.fun)
    except Exception:
        return pos, _lj_energy(pos)


def construct():
    global _START
    best_pos = None
    best_energy = np.inf

    # Candidate (inner, outer) radius pairs. The LJ pair minimum is r = 2^(1/6) ~ 1.1225.
    # For a Mackay icosahedron the outer shell radius is roughly 2 * inner.
    candidates = [
        (1.1225, 2.245),
        (1.12, 2.24),
        (1.10, 2.20),
        (1.15, 2.30),
        (1.05, 2.10),
        (1.20, 2.40),
        (1.00, 2.00),
        (1.1225, 2.10),
        (1.1225, 2.35),
        (1.1225, 2.50),
        (1.08, 2.16),
        (1.16, 2.32),
    ]

    for (ri, ro) in candidates:
        if time.time() > _DEADLINE:
            break
        pos = _mulliken_icosahedron(ri, ro)
        # Enforce minimum separation of 0.5 by nudging if needed (shouldn't trigger for good scales)
        pos, e = _relax(pos)
        # Validate separation
        diff = pos[:, None, :] - pos[None, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        if np.min(r2) < 0.25 - 1e-9:
            continue
        e = _lj_energy(pos)
        if e < best_energy:
            best_energy = e
            best_pos = pos.copy()

    # Final high-precision polish on the best found
    if best_pos is not None and time.time() < _DEADLINE:
        try:
            res = minimize(_lj_energy_grad, best_pos, method='L-BFGS-B',
                           jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12})
            cand = res.x
            diff = cand[:, None, :] - cand[None, :, :]
            r2 = np.einsum('ijk,ijk->ij', diff, diff)
            np.fill_diagonal(r2, np.inf)
            if np.min(r2) >= 0.25 - 1e-9:
                best_pos = cand
        except Exception:
            pass

    if best_pos is None:
        # Fallback: seed-like cubic lattice (should not happen)
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12

    return best_pos
