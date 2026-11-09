"""Sutton-Chen cluster global-minima search: diverse structural prototypes + soft/hard relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = time.time() + 285.0


def _remaining():
    return _DEADLINE - time.time()


def _valid(pos):
    if pos is None or not np.all(np.isfinite(pos)):
        return False
    if len(pos) < 2:
        return True
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min() > 0.35


def _fcc(n):
    pts = []
    for i in range(-20, 21):
        for j in range(-20, 21):
            for k in range(-20, 21):
                pts.append([i, j, k])
    pts = np.array(pts, dtype=float)
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _bcc(n):
    pts = []
    for i in range(-20, 21):
        for j in range(-20, 21):
            for k in range(-20, 21):
                pts.append([i, j, k])
                pts.append([(i + 0.5), (j + 0.5), (k + 0.5)])
    pts = np.array(pts, dtype=float)
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _hcp(n):
    pts = []
    a = 1.0
    h = np.sqrt(2.0 / 3.0) * a
    for layer in range(-15, 16):
        z = layer * h
        if layer % 3 == 0:
            off = (0.0, 0.0)
        elif layer % 3 == 1:
            off = (a / 2.0, a / (2 * np.sqrt(3.0)))
        else:
            off = (0.0, a / np.sqrt(3.0))
        for i in range(-15, 16):
            for j in range(-15, 16):
                x = i * a + (a / 2.0 if j % 2 else 0.0) + off[0]
                y = j * a * np.sqrt(3.0) / 2.0 + off[1]
                pts.append([x, y, z])
    pts = np.array(pts, dtype=float)
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _icosahedral(n):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v1 = np.array([0.0, 1.0, t])
    v2 = np.array([1.0, t, 0.0])
    v3 = np.array([t, 0.0, 1.0])
    verts = []
    for v in (v1, v2, v3):
        for s in (1, -1):
            verts.append(v * s)
            verts.append(v[[1, 0, 2]] * s)
            verts.append(v[[2, 1, 0]] * s)
    verts = np.array(verts, dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1)[:, None]

    # second shell: edge midpoints (12) + face centers (20)
    verts2 = []
    # face centers: cross products of adjacent vertices (20 faces)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 0.5:  # adjacent -> edge; midpoint
                verts2.append((verts[i] + verts[j]) / 2.0)
    # face centers via triples
    used = set()
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            for k in range(j + 1, len(verts)):
                a, b, c = verts[i], verts[j], verts[k]
                da = np.linalg.norm(a - b)
                db = np.linalg.norm(b - c)
                dc = np.linalg.norm(c - a)
                if da < 0.5 and db < 0.5 and dc < 0.5:
                    fc = (a + b + c) / 3.0
                    fc = fc / np.linalg.norm(fc)
                    key = tuple(np.round(fc, 3))
                    if key not in used:
                        used.add(key)
                        verts2.append(fc)
    verts2 = np.array(verts2, dtype=float)

    # third shell: icosahedral vertices scaled by sqrt(5)
    verts3 = verts * np.sqrt(5.0)

    all_pts = np.vstack([verts, verts2, verts3])
    r = np.linalg.norm(all_pts, axis=1)
    idx = np.argsort(r)[:n]
    return all_pts[idx]


def _relax_soft_hard(pos, potential, kind):
    """Soft then hard relaxation; returns (energy, positions)."""
    e0, p0 = sutton_chen_energy_gradient(pos, kind)
    if not _valid(p0):
        return np.inf, pos
    # hard relaxation
    e1, p1 = relax(pos, potential, kind)
    if _valid(p1) and e1 < e0:
        return e1, p1
    return e0, p0


def construct(instance):
    n = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")

    prototypes = {
        "fcc": _fcc,
        "hcp": _hcp,
        "bcc": _bcc,
        "ico": _icosahedral,
    }
    scales = [0.85, 1.0, 1.15, 1.3, 1.5]

    best_e = np.inf
    best_pos = None

    for name, gen in prototypes.items():
        if _remaining() < 20.0:
            break
        for s in scales:
            if _remaining() < 15.0:
                break
            try:
                pos = gen(n) * s
            except Exception:
                continue
            if not _valid(pos):
                continue
            try:
                e, p = _relax_soft_hard(pos, potential, kind)
            except Exception:
                continue
            if _valid(p) and e < best_e:
                best_e = e
                best_pos = p

    # fallback if nothing valid
    if best_pos is None:
        pos = _fcc(n)
        e, best_pos = relax(pos, potential, kind)
        best_e = e

    return best_pos
# EVOLVE-BLOCK-END
