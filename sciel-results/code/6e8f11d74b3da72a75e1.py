"""LJ38 global minimum search: symmetry-adapted FCC construction with
multi-scale perturbation, basin hopping, and high-precision L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize

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


def _get_fcc_shell_structure(n_atoms=38):
    """Return base FCC positions and shell assignments for LJ38 truncated octahedron."""
    scale = 1.0 / np.sqrt(2.0)
    max_coord = 8
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x * x + y * y + z * z
                    points.append((x, y, z, r2))
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    # Assign shells based on radial distance
    radii = np.linalg.norm(coords, axis=1)
    # Cluster radii into shells
    unique_r = np.sort(np.unique(np.round(radii, 4)))
    shell_ids = np.zeros(n_atoms, dtype=int)
    for i, r in enumerate(radii):
        for sid, ur in enumerate(unique_r):
            if abs(r - ur) < 0.05:
                shell_ids[i] = sid
                break
    return coords, shell_ids, unique_r


def _symmetry_adapted_energy(params, base_coords, shell_ids):
    """Compute energy for symmetry-adapted parameters."""
    n_atoms = N
    global_scale = params[0]
    shell_params = params[1:]
    
    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        positions[i] = base_coords[i] * breath * global_scale
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return 4.0 * np.sum(inv_r12 - inv_r6) / 2.0


def _symmetry_adapted_energy_and_grad(params, base_coords, shell_ids):
    """
    params: [global_scale, shell_breathing_0, shell_breathing_1, ...]
    Returns energy and gradient w.r.t. params using numerical differentiation.
    """
    energy = _symmetry_adapted_energy(params, base_coords, shell_ids)
    
    # Numerical gradient for robustness
    eps = 1e-7
    grad = np.zeros_like(params)
    for k in range(len(params)):
        params_plus = params.copy()
        params_plus[k] += eps
        e_plus = _symmetry_adapted_energy(params_plus, base_coords, shell_ids)
        params_minus = params.copy()
        params_minus[k] -= eps
        e_minus = _symmetry_adapted_energy(params_minus, base_coords, shell_ids)
        grad[k] = (e_plus - e_minus) / (2 * eps)
    
    return energy, grad


def _make_fcc_truncated_octahedron():
    """
    Construct the LJ38 global-minimum truncated-octahedron geometry using
    symmetry-adapted parameter optimization (global scale + shell breathing).
    """
    r0 = 2.0 ** (1.0 / 6.0)
    
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    n_shells = len(unique_r)
    
    # Initial guess: scale = r0, all breathing = 0
    x0 = np.array([r0] + [0.0] * n_shells)
    
    # Optimize symmetry-adapted parameters
    try:
        res = minimize(
            lambda p: _symmetry_adapted_energy_and_grad(p, base_coords, shell_ids),
            x0, method='L-BFGS-B', jac=True,
            options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 100}
        )
        if res.success or res.fun < _symmetry_adapted_energy(x0, base_coords, shell_ids):
            opt_params = res.x
        else:
            opt_params = x0
    except Exception:
        opt_params = x0
    
    # Build final positions from optimized parameters
    global_scale = opt_params[0]
    shell_params = opt_params[1:]
    
    positions = np.zeros((N, 3))
    for i in range(N):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        positions[i] = base_coords[i] * breath * global_scale
    
    # Re-center
    positions -= positions.mean(axis=0)
    
    return positions


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


def _perturb(pos, sigma, rng):
    """Add Gaussian noise to positions."""
    return pos + rng.randn(*pos.shape) * sigma


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_S - MARGIN

    best_pos = None
    best_e = np.inf

    # 1) FCC truncated octahedron with symmetry-adapted optimization: relax + polish
    try:
        pos0 = _make_fcc_truncated_octahedron()
        pos_r, e = _relax(pos0, maxiter=20000, deadline=deadline,
                          gtol=1e-10, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
        # High-precision polish pass
        if time.time() < deadline - 5.0:
            pos_r2, e2 = _relax(pos_r, maxiter=10000, deadline=deadline,
                                gtol=1e-12, ftol=1e-16)
            if e2 < best_e:
                best_e = e2
                best_pos = pos_r2.copy()
    except Exception:
        pass

    if best_pos is not None and time.time() > deadline:
        return best_pos

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
            return best_pos

    # 3) Multi-scale perturbations around the best FCC result
    rng = np.random.RandomState(12345)
    base = _make_fcc_truncated_octahedron()
    
    # Define a list of sigmas and counts
    perturb_schedule = [
        (0.01, 5),
        (0.03, 5),
        (0.05, 5),
        (0.08, 5),
        (0.12, 5),
        (0.18, 5),
        (0.25, 5),
        (0.35, 5)
    ]
    
    for sigma, count in perturb_schedule:
        if time.time() > deadline:
            break
        for _ in range(count):
            if time.time() > deadline:
                break
            pos_p = _perturb(base, sigma, rng)
            pos_r, e = _relax(pos_p, maxiter=10000, deadline=deadline,
                              gtol=1e-10, ftol=1e-16)
            if e < best_e:
                best_e = e
                best_pos = pos_r.copy()

    # 4) Random starts (catch-all)
    for _ in range(5):
        if time.time() > deadline:
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
        pos_r, e = _relax(best_pos, maxiter=20000, deadline=deadline,
                          gtol=1e-14, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
            
    # One more ultra-high precision pass if time allows
    if time.time() < deadline - 1.0:
        pos_r, e = _relax(best_pos, maxiter=5000, deadline=deadline,
                          gtol=1e-15, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()

    # Ensure minimum pairwise distance >= 0.5
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        scale = 0.5 / min_r * 1.001
        best_pos = best_pos * scale

    return best_pos
