"""Lennard-Jones 55-atom cluster: build a Mackay icosahedron and polish it."""
import time
import numpy as np

N = 55
DEADLINE = 135.0  # safety margin below the 150 s kill

# icosahedron vertex set (12)
_PHI = (1.0 + 5.0 ** 0.5) / 2.0
_ICO_VERTS = np.array([
    [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
    [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
    [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1],
], dtype=float)
_ICO_VERTS /= np.linalg.norm(_ICO_VERTS[0])  # unit radius


def _lj_energy(pos):
    """Total Lennard-Jones energy 4*sum(r^-12 - r^-6) over all pairs."""
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, 1.0)
    inv = 1.0 / r2
    return 4.0 * np.sum(inv ** 3 - inv ** 2)


def _build_mackay(r1, r2):
    """Mackay icosahedron: 1 centre + 12 inner shell + 42 outer shell."""
    pts = [np.zeros(3)]
    pts += [r1 * v for v in _ICO_VERTS]
    # 42 outer-shell directions: 12 (5-fold) + 30 (3-fold)
    outer = []
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            outer.append(_ICO_VERTS[i] + _ICO_VERTS[j])
    for i in range(12):
        for j in range(i + 1, 12):
            outer.append(_ICO_VERTS[i] + _ICO_VERTS[j])
    outer = np.array(outer, dtype=float)
    outer /= np.linalg.norm(outer, axis=1, keepdims=True)
    pts += [r2 * v for v in outer]
    return np.array(pts, dtype=float)


def _try_relax(pos, t_start, t_end):
    """Run L-BFGS relaxation with a deadline guard; return best positions."""
    try:
        from science_chemistry import relax
    except Exception:
        return pos
    best = pos
    best_e = _lj_energy(pos)
    try:
        out = relax(pos)
        if out is not None and len(out) == N and np.all(np.isfinite(out)):
            e = _lj_energy(out)
            if e < best_e:
                best, best_e = out, e
        return best
    except Exception:
        return best


def construct():
    t0 = time.time()
    # Candidate radii near the known Mackay minimum; relax each.
    candidates = [
        _build_mackay(1.05, 2.10),
        _build_mackay(1.06, 2.12),
        _build_mackay(1.04, 2.08),
        _build_mackay(1.07, 2.15),
        _build_mackay(1.05, 2.15),
        _build_mackay(1.02, 2.05),
    ]
    best = candidates[0]
    best_e = _lj_energy(best)

    for c in candidates:
        if time.time() > DEADLINE:
            break
        r = _try_relax(c, t0, DEADLINE)
        e = _lj_energy(r)
        if e < best_e:
            best, best_e = r, e

    # Final high-precision polish on the best result if time remains.
    if time.time() < DEADLINE - 10:
        r = _try_relax(best, t0, DEADLINE)
        e = _lj_energy(r)
        if e < best_e:
            best = r

    # Enforce minimum separation of 0.5 (defensive; should already hold).
    if time.time() < DEADLINE:
        d = best[:, None, :] - best[None, :, :]
        r2 = np.einsum('ijk,ijk->ij', d, d)
        np.fill_diagonal(r2, 1.0)
        if np.min(r2) < 0.25:
            # gently nudge any too-close pair apart
            for _ in range(200):
                r2 = np.einsum('ijk,ijk->ij', best[:, None, :] - best[None, :, :],
                               best[:, None, :] - best[None, :, :])
                np.fill_diagonal(r2, 1.0)
                if np.min(r2) >= 0.25:
                    break
                i, j = np.unravel_index(np.argmin(r2), r2.shape)
                if i == j:
                    break
                v = best[i] - best[j]
                nrm = np.linalg.norm(v)
                if nrm < 1e-8:
                    v = np.array([1.0, 0, 0]); nrm = 1.0
                v /= nrm
                push = (0.5 - nrm) / 2.0 + 1e-4
                best[i] += push * v
                best[j] -= push * v
    return best
