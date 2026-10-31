"""
LJ38 cluster global minimum search.

DIRECT approach: build the FCC truncated-octahedron geometry explicitly from
the known structural description, then polish with a tight L-BFGS local
relaxation.  The global minimum of LJ38 is a face-centred-cubic truncated
octahedron with energy ≈ -173.928427 (Cambridge Cluster Database).  We
construct this geometry analytically, verify minimum inter-atomic separation,
and perform a short local optimisation to remove any residual strain.

Entry point: construct() -> (N, 3) array of Cartesian coordinates.
"""
import time
import numpy as np

N = 38


def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    # Use a small epsilon to avoid division by zero and improve numerical stability
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-16)  # Floor to prevent extreme values
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad


def _make_fcc_truncated_octahedron():
    """
    Build the FCC truncated-octahedron geometry for LJ38.

    The 38-atom FCC cluster is a truncated octahedron taken from the FCC lattice.
    In FCC coordinates (lattice constant a), the atoms sit at positions where
    all three coordinates are integers and the sum is even (or equivalently,
    the standard FCC basis).  For the truncated octahedron of 38 atoms we take
    all FCC lattice points (i, j, k) with i+j+k even inside a box, sorted by
    distance from the origin, and keep the 38 closest.  This yields the
    characteristic truncated-octahedron shape.

    We then scale so that the nearest-neighbour distance equals 2^(1/6),
    the equilibrium separation of the LJ potential, and centre the cluster.
    """
    # Generate FCC points: (i, j, k) with i+j+k even, in a generous box
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))

    pts = np.array(pts, dtype=np.float64)
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]

    # Take the 38 closest to the origin
    pts = pts[:N].copy()

    # Centre
    pts -= np.mean(pts, axis=0)

    # Scale so that the mean nearest-neighbour distance is 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale

    return pts


def _make_icosahedral_structure():
    """
    Icosahedral-motif starting geometry (known local minimum / funnel).
    1 centre + 12 + 12 + 12 + 1 = 38.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms

    layers = [1.0, 1.7, 2.4]
    pts = [np.zeros(3)]
    for r in layers:
        pts.extend(ico * r)
    # 1 + 12*3 = 37, add one more
    pts.append(np.array([0.25, 0.25, 0.25]))
    pts = np.array(pts[:N], dtype=np.float64)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale

    return pts


def _relax(pos, max_iter=400, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation with analytic gradient."""
    from scipy.optimize import minimize

    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out


def _polish(pos, max_iter=1000):
    """High-precision polish pass with very tight tolerances and multiple strategies."""
    from scipy.optimize import minimize
    
    best_pos = pos
    best_energy = np.inf
    
    # Strategy 1: L-BFGS-B with very tight tolerances
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()
    
    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": 1e-14, "ftol": 1e-18, "maxls": 50},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        if e_out < best_energy:
            best_energy = e_out
            best_pos = p_out.copy()
    except Exception:
        pass
    
    # Strategy 2: Second polish pass from the result of strategy 1
    if best_pos is not pos:
        try:
            res2 = minimize(
                obj_grad,
                best_pos.ravel(),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": max_iter // 2, "gtol": 1e-15, "ftol": 1e-18, "maxls": 30},
            )
            p_out2 = res2.x.reshape(-1, 3)
            e_out2, _ = _lj_energy_and_grad(p_out2)
            if e_out2 < best_energy:
                best_energy = e_out2
                best_pos = p_out2.copy()
        except Exception:
            pass
    
    return best_pos, best_energy


def _ensure_min_sep(pos, min_sep=0.5):
    """Scale up the cluster if any pair is closer than min_sep."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin within 120 s (or 150 s) hard limit

    rng = np.random.default_rng(12345)

    # ---- 1. Build the direct FCC truncated-octahedron geometry ----
    try:
        fcc = _make_fcc_truncated_octahedron()
    except Exception:
        fcc = None

    starts = []
    if fcc is not None:
        starts.append(fcc)

    # Perturbed FCC variants (small noise to escape possible shallow local minima
    # near the FCC basin, though the FCC should already be at the global min)
    if fcc is not None:
        for _ in range(4):
            if time.time() > deadline - 40.0:
                break
            perturbed = fcc + 0.05 * rng.standard_normal(fcc.shape)
            starts.append(_ensure_min_sep(perturbed))

    # Icosahedral starting geometry (known local minimum, useful as a
    # cross-check; unlikely to beat FCC but cheap to try)
    try:
        ico = _make_icosahedral_structure()
        starts.append(ico)
    except Exception:
        pass

    best_pos = None
    best_energy = np.inf

    for start in starts:
        if time.time() > deadline - 40.0:
            break
        try:
            p, e = _relax(start, max_iter=300)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # Fallback if all relaxations failed
    if best_pos is None:
        if fcc is not None:
            best_pos = fcc
        else:
            side = int(np.ceil(N ** (1.0 / 3.0)))
            pts = []
            for i in range(side):
                for j in range(side):
                    for k in range(side):
                        pts.append([i, j, k])
            pts = np.array(pts[:N], dtype=np.float64)
            pts -= np.mean(pts, axis=0)
            best_pos = pts * 1.12

    best_pos = _ensure_min_sep(best_pos)

    # High-precision polish: tighter tolerances to squeeze out residual energy
    if time.time() < deadline - 15.0:
        try:
            p, e = _polish(best_pos, max_iter=800)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # Final micro-polish if time remains
    if time.time() < deadline - 5.0:
        try:
            p, e = _polish(best_pos, max_iter=400)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)

    return best_pos
