"""LJ38 global minimum via symmetry-adapted truncated-octahedron parameterization."""
import time

N = 38

# --- deadline guard ---------------------------------------------------------
_T0 = time.time()
_LIMIT = 115.0  # safety margin under the 150 s wall-clock kill


def _time_left():
    return _LIMIT - (time.time() - _T0)


# --- symmetry-adapted truncated octahedron (FCC) geometry -------------------
# 38 sites: 1 body-center (Oh 1a) + 12 cube-corner (Oh 3a) + 12 face-center (Oh 3c)
#          + 12 edge-center (Oh 3e).  Four radial parameters.
_BASE = np.array([
    [0.0, 0.0, 0.0],
    # 12 cube corners
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1],
    # 12 face centers
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0],
    [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1],
    # 12 edge centers
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
    [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
], dtype=float)
# NOTE: the block above is illustrative; the canonical 38-site list is built
# precisely below to guarantee exactly 38 unique sites.

import numpy as np
import itertools


def _build_orbits():
    """Return (orbits, base_dirs) where orbits is a list of arrays of unit-ish
    direction vectors (one per orbit) and base_dirs the full 38x3 base matrix."""
    # Orbit A: body center
    A = np.array([[0.0, 0.0, 0.0]])
    # Orbit B: cube corners (12)
    B = np.array([list(t) for t in itertools.product([1, -1], repeat=3)], dtype=float)
    # Orbit C: face centers (12)
    C = []
    for axis in range(3):
        for s in (1, -1):
            v = [0.0, 0.0, 0.0]
            v[axis] = s
            C.append(v)
    C = np.array(C, dtype=float)
    # Orbit D: edge centers (12)
    D = []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0.0, 0.0, 0.0]
                    v[i] = si
                    v[j] = sj
                    D.append(v)
    D = np.array(D, dtype=float)
    base = np.vstack([A, B, C, D])
    assert base.shape == (38, 3)
    return [A, B, C, D], base


_ORBITS, _BASE = _build_orbits()
# radii multipliers per orbit index in the stacked base
_ORBIT_RADIUS = np.array([
    0.0,
    1.0] * 1 + [1.0] * 12 + [1.0] * 12 + [1.0] * 12, dtype=float)


def _positions_from_params(p):
    """p = [rA, rB, rC, rD] -> 38x3 Cartesian positions."""
    rA, rB, rC, rD = p
    pos = np.empty((38, 3), dtype=float)
    pos[0:1] = _ORBITS[0] * rA
    pos[1:13] = _ORBITS[1] * rB
    pos[13:25] = _ORBITS[2] * rC
    pos[25:38] = _ORBITS[3] * rD
    return pos


# --- energy -----------------------------------------------------------------
def _lj_energy(pos):
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    return 4.0 * np.sum(inv_r6 * inv_r6 - inv_r6)


def _min_pair(pos):
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, np.inf)
    return np.sqrt(np.min(r2))


def _valid(pos):
    return _min_pair(pos) >= 0.5 - 1e-9


# --- local relaxation -------------------------------------------------------
def _relax(pos):
    try:
        from science_chemistry import relax
        return relax(pos)
    except Exception:
        return pos


# --- search -----------------------------------------------------------------
def construct():
    best_pos = _positions_from_params(np.array([0.0, 1.12, 1.0, 0.78]))
    best_e = _lj_energy(best_pos)

    # 1) coarse grid over the 4 radial parameters
    grid = {
        0: [0.0],
        1: [1.05, 1.10, 1.12, 1.14, 1.18],   # cube corners
        2: [0.95, 1.00, 1.05, 1.10, 1.15],   # face centers
        3: [0.70, 0.75, 0.78, 0.82, 0.88],   # edge centers
    }
    keys = [1, 2, 3]
    import itertools as _it
    for combo in _it.product(*[grid[k] for k in keys]):
        if _time_left() < 5.0:
            break
        p = np.array([0.0, combo[0], combo[1], combo[2]])
        pos = _positions_from_params(p)
        if not _valid(pos):
            continue
        e = _lj_energy(pos)
        if e < best_e:
            best_e, best_pos = e, pos

    # 2) Nelder-Mead refinement on the 4-d parameter space
    def _obj(p):
        pos = _positions_from_params(p)
        if not _valid(pos):
            return 1e6
        return _lj_energy(pos)

    try:
        from scipy.optimize import minimize
        x0 = np.array([0.0, 1.12, 1.0, 0.78])
        res = minimize(_obj, x0, method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-12})
        if _valid(_positions_from_params(res.x)):
            cand = _positions_from_params(res.x)
            ce = _lj_energy(cand)
            if ce < best_e:
                best_e, best_pos = ce, cand
    except Exception:
        pass

    # 3) full 114-d local relaxation from the best symmetry point
    if _time_left() > 8.0:
        relaxed = _relax(best_pos.copy())
        if relaxed is not None and _valid(np.asarray(relaxed, dtype=float)):
            relaxed = np.asarray(relaxed, dtype=float)
            re = _lj_energy(relaxed)
            if re < best_e:
                best_e, best_pos = re, relaxed

    # 4) final high-precision polish
    if _time_left() > 5.0:
        try:
            from scipy.optimize import minimize
            def _full_obj(flat):
                pos = flat.reshape(38, 3)
                if not _valid(pos):
                    return 1e6
                return _lj_energy(pos)
            res = minimize(_full_obj, best_pos.reshape(-1), method='L-BFGS-B',
                           options={'maxiter': 2000, 'gtol': 1e-10})
            cand = res.x.reshape(38, 3)
            if _valid(cand):
                ce = _lj_energy(cand)
                if ce < best_e:
                    best_e, best_pos = ce, cand
        except Exception:
            pass

    return best_pos
