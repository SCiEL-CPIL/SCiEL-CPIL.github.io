import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 38
DEADLINE_S = 120.0
MARGIN = 15.0


def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for positions (N,3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6) / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _relax(pos, maxiter=20000, deadline=None, gtol=1e-10, ftol=1e-16):
    """L-BFGS-B relaxation with optional deadline and precision control."""
    pos = np.array(pos, dtype=float).copy()

    def fun_and_grad(x):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(fun_and_grad, pos.ravel(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol,
                                'maxls': 50})
        return res.x.reshape(-1, 3), res.fun
    except TimeoutError:
        return pos, _lj_energy_and_grad(pos)[0]
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]


def _make_fcc_truncated_octahedron():
    """
    Construct the FCC truncated-octahedron geometry for 38 atoms.
    Uses proper FCC primitive vectors to generate lattice points.
    """
    # FCC primitive vectors
    a1 = np.array([0.0, 1.0, 1.0])
    a2 = np.array([1.0, 0.0, 1.0])
    a3 = np.array([1.0, 1.0, 0.0])

    pts = []
    max_c = 4
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                p = i * a1 + j * a2 + k * a3
                d2 = np.dot(p, p)
                if d2 <= 20.0:
                    pts.append([p[0], p[1], p[2], d2])

    pts = np.array(pts)
    # Deduplicate
    pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    unique_pts = [pts_sorted[0]]
    for p in pts_sorted[1:]:
        if np.linalg.norm(p[:3] - unique_pts[-1][:3]) > 1e-10:
            unique_pts.append(p)
    pts = np.array(unique_pts)

    # Sort by distance from origin and take 38 closest
    pts = pts[np.argsort(pts[:, 3])]
    pos = pts[:N, :3].copy()

    # Center
    pos -= pos.mean(axis=0)

    # Scale so median nearest-neighbour distance = r0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    nn_dists = np.min(r, axis=1)
    median_nn = np.median(nn_dists)

    r0 = 2.0 ** (1.0 / 6.0)
    if median_nn > 1e-12:
        scale = r0 / median_nn
        pos = pos * scale

    # Re-center
    pos -= pos.mean(axis=0)
    return pos


def _perturb(pos, sigma, rng):
    """Add Gaussian noise to positions."""
    return pos + rng.randn(*pos.shape) * sigma


def _make_icosahedral():
    """Construct an icosahedral arrangement for 38 atoms (1+12+25)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    r0 = 2.0 ** (1.0 / 6.0)

    # 12 vertices of an icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * r0

    pos_list = [np.zeros(3)]
    pos_list.extend(list(verts))

    # Add 25 more atoms on a second shell
    n_outer = N - len(pos_list)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n_outer):
        z = 1.0 - 2.0 * (i + 0.5) / n_outer
        radius = np.sqrt(1.0 - z * z)
        theta = 2.0 * np.pi * i / golden
        v = 2.0 * r0 * np.array([radius * np.cos(theta),
                                  radius * np.sin(theta),
                                  z])
        pos_list.append(v)

    pos = np.array(pos_list[:N])
    pos -= pos.mean(axis=0)
    return pos


def _optimize_scale(pos):
    """
    Optimize the global scale of the cluster to minimize energy.
    This is a 1D optimization problem.
    """
    def energy_at_scale(s):
        scaled_pos = pos * s
        e, _ = _lj_energy_and_grad(scaled_pos)
        return e

    # Initial scale is 1.0. We expect the optimal scale to be close to 1.0.
    res = minimize_scalar(energy_at_scale, bounds=(0.8, 1.5), method='bounded', options={'xatol': 1e-10})
    if res.success:
        optimal_scale = res.x
        optimal_pos = pos * optimal_scale
        optimal_energy = res.fun
        return optimal_pos, optimal_energy
    else:
        return pos, energy_at_scale(1.0)


def _basin_hop(pos, rng, max_iters=50, deadline=None, t0=0.15, t1=0.01):
    """
    Basin hopping: perturb, relax, accept/reject by Metropolis criterion.
    """
    current_pos = pos.copy()
    current_energy, _ = _lj_energy_and_grad(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy

    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Perturbation: add Gaussian noise with annealing
        noise_scale = t0 * (t1 / t0) ** (i / max_iters)
        perturbed = current_pos + rng.normal(0, noise_scale, size=current_pos.shape)

        # Relax
        relaxed, e_relaxed = _relax(perturbed, maxiter=3000, deadline=deadline,
                                    gtol=1e-9, ftol=1e-14)
        e_relaxed, _ = _lj_energy_and_grad(relaxed)

        # Metropolis acceptance
        if e_relaxed < current_energy:
            accept = True
        else:
            delta_e = e_relaxed - current_energy
            T_curr = t0 * (t1 / t0) ** (i / max_iters)
            if T_curr > 1e-10:
                accept = rng.random() < np.exp(-delta_e / T_curr)
            else:
                accept = False

        if accept:
            current_pos = relaxed.copy()
            current_energy = e_relaxed
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()

    return best_pos, best_energy


def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure minimum pairwise distance >= min_sep."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_S - MARGIN

    best_pos = None
    best_e = np.inf

    # 1) FCC truncated octahedron: initial relax + scale optimization + high-precision polish
    try:
        pos0 = _make_fcc_truncated_octahedron()
        pos_r, e = _relax(pos0, maxiter=20000, deadline=deadline,
                          gtol=1e-10, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()

        # Scale optimization
        if time.time() < deadline - 5.0:
            pos_s, e_s = _optimize_scale(pos_r)
            if e_s < best_e:
                best_e = e_s
                best_pos = pos_s.copy()

        # High-precision polish pass
        if time.time() < deadline - 3.0:
            pos_r2, e2 = _relax(best_pos, maxiter=10000, deadline=deadline,
                                gtol=1e-12, ftol=1e-16)
            if e2 < best_e:
                best_e = e2
                best_pos = pos_r2.copy()

        # Another scale optimization after polish
        if time.time() < deadline - 2.0:
            pos_s2, e_s2 = _optimize_scale(best_pos)
            if e_s2 < best_e:
                best_e = e_s2
                best_pos = pos_s2.copy()

    except Exception:
        pass

    if best_pos is not None and time.time() > deadline:
        return _ensure_min_sep(best_pos)

    # 2) Icosahedral start
    try:
        pos_ico = _make_icosahedral()
        pos_r, e = _relax(pos_ico, maxiter=15000, deadline=deadline,
                          gtol=1e-10, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
    except Exception:
        pass

    if time.time() > deadline:
        if best_pos is not None:
            return _ensure_min_sep(best_pos)

    # 3) Basin hopping from the best FCC result
    rng = np.random.RandomState(12345)
    if time.time() < deadline - 10.0:
        try:
            pos_bh, e_bh = _basin_hop(best_pos, rng, max_iters=50, deadline=deadline)
            if e_bh < best_e:
                best_e = e_bh
                best_pos = pos_bh.copy()
        except Exception:
            pass

    # 4) Multi-scale perturbations around the best FCC result
    base = _make_fcc_truncated_octahedron()
    for sigma in [0.01, 0.03, 0.05, 0.08, 0.12]:
        if time.time() > deadline - 5.0:
            break
        for _ in range(2):
            if time.time() > deadline - 5.0:
                break
            pos_p = _perturb(base, sigma, rng)
            pos_r, e = _relax(pos_p, maxiter=8000, deadline=deadline,
                              gtol=1e-10, ftol=1e-16)
            if e < best_e:
                best_e = e
                best_pos = pos_r.copy()

    # 5) Random starts (catch-all)
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        pos_rand = rng.randn(N, 3) * 2.5
        pos_r, e = _relax(pos_rand, maxiter=5000, deadline=deadline,
                          gtol=1e-10, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()

    if best_pos is None:
        # Fallback: cubic lattice
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        pts.sort(key=lambda p: sum((c - (side - 1) / 2.0) ** 2 for c in p))
        return np.array(pts[:N], dtype=float) * 1.12

    # Final high-precision polish if time remains
    if time.time() < deadline - 3.0:
        pos_r, e = _relax(best_pos, maxiter=10000, deadline=deadline,
                          gtol=1e-12, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()

    # Final scale optimization
    if time.time() < deadline - 1.0:
        pos_s, e_s = _optimize_scale(best_pos)
        if e_s < best_e:
            best_e = e_s
            best_pos = pos_s.copy()

    # Ensure minimum pairwise distance >= 0.5
    best_pos = _ensure_min_sep(best_pos)

    return best_pos
