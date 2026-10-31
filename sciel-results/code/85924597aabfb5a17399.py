"""LJ-38 global minimum: exploit the known FCC truncated-octahedron topology,
build the exact Oh-symmetric candidate, polish with L-BFGS, and guard time."""
import numpy as np
import time

N = 38
_DEADLINE = 150.0
_START = time.time()
_TIME_MARGIN = 6.0          # return this many seconds before the hard limit


def _remaining():
    return _DEADLINE - _TIME_MARGIN - (time.time() - _START)


def _lj_energy(pos):
    pos = np.asarray(pos, dtype=float)
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** -3
    return 4.0 * (r6 * r6 - r6).sum()


def _lj_energy_grad(pos):
    pos = np.asarray(pos, dtype=float)
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** -3
    r12 = r6 * r6
    # per-pair energy e = 4*(r12 - r6); de/dpos_i = 4*(-12 r12 + 6 r6)/r2 * d_i
    c = 4.0 * (-12.0 * r12 + 6.0 * r6) / r2
    c = np.where(np.isfinite(c), c, 0.0)
    grad = np.einsum('ij,ijk->ik', c, d)
    return 4.0 * (r12 - r6).sum(), grad


def _generate_trunc_octahedron(a):
    """38-atom FCC truncated octahedron (Oh symmetry) at lattice constant a."""
    pts = []
    pts.append((0.0, 0.0, 0.0))                                   # 1
    for s1 in (1.0, -1.0):
        for s2 in (1.0, -1.0):
            for s3 in (1.0, -1.0):
                pts.append((0.5 * a * s1, 0.5 * a * s2, 0.5 * a * s3))   # 8
    for s1 in (1.0, -1.0):
        for s2 in (1.0, -1.0):
            pts.append((a * s1, 0.0, 0.0))
            pts.append((0.0, a * s1, 0.0))
            pts.append((0.0, 0.0, a * s2))                          # 12
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                pts.append((0.5 * a * sx, 0.5 * a * sy, a * sz))
                pts.append((0.5 * a * sx, a * sy, 0.5 * a * sz))
                pts.append((a * sx, 0.5 * a * sy, 0.5 * a * sz))     # 24
    return np.array(pts, dtype=float)


def _seeds():
    seeds = []
    seeds.append(_generate_trunc_octahedron(1.09))
    try:
        from scipy.optimize import minimize
        # Refine the single lattice parameter a (exact Oh symmetry) to its optimum.
        e0, _ = _lj_energy_grad(_generate_trunc_octahedron(1.09))

        def _e_of_a(av):
            return _lj_energy(_generate_trunc_octahedron(av[0]))

        res = minimize(_e_of_a, np.array([1.09]), method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000})
        a_opt = float(res.x[0])
        cand = _generate_trunc_octahedron(a_opt)
        if _min_dist(cand) > 0.5 and _lj_energy(cand) < e0:
            seeds.append(cand)
    except Exception:
        pass
    return seeds


def _min_dist(pos):
    d = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    return r.min()


def _relax(pos):
    """L-BFGS polish with analytic gradient; returns (positions, energy)."""
    from scipy.optimize import minimize
    pos = np.asarray(pos, dtype=float)
    if _min_dist(pos) <= 0.5:
        return pos, _lj_energy(pos)
    try:
        res = minimize(_lj_energy_grad, pos, jac=True, method='L-BFGS-B',
                       options={'maxiter': 3000, 'ftol': 1e-16,
                                'gtol': 1e-12, 'maxls': 50, 'maxfun': 200000})
        if res.success or res.x is not None:
            x = np.asarray(res.x, dtype=float)
            if _min_dist(x) > 0.5:
                return x, _lj_energy(x)
    except Exception:
        pass
    return pos, _lj_energy(pos)


def construct():
    best = None
    best_e = np.inf

    # 1) Exact Oh-symmetric truncated-octahedron candidate(s), polished.
    for s in _seeds():
        if _remaining() < 0:
            break
        x, e = _relax(s)
        if e < best_e and _min_dist(x) > 0.5:
            best, best_e = x, e

    # 2) A few relaxed random restarts as a cheap safety net.
    rng = np.random.default_rng(12345)
    for _ in range(12):
        if _remaining() < 3.0:
            break
        x = rng.normal(0.0, 1.1, size=(N, 3))
        x, e = _relax(x)
        if e < best_e and _min_dist(x) > 0.5:
            best, best_e = x, e

    if best is None:
        # Fallback: dense cubic fragment (valid, >=0.5 apart).
        side = int(np.ceil(N ** (1 / 3)))
        pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        pts.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2
                 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best = np.array(pts[:N], dtype=float) * 1.12

    return np.asarray(best, dtype=float)
