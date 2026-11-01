"""Improved Lennard-Jones cluster global minimum, 55 atoms."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist

# Deadline guard: 150s wall-clock limit, use 145s for safety
_DEADLINE = 145.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _lj_energy(pos):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    d = pdist(pos)
    d = np.maximum(d, 1e-10)
    r6 = d ** -6
    return 4.0 * np.sum(r6 ** 2 - r6)


def _lj_grad(pos):
    """Compute gradient of LJ energy."""
    n = len(pos)
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                continue
            r6 = r2 ** -3
            r12 = r6 ** 2
            coeff = 4.0 * (-12.0 * r12 + 6.0 * r6) / r2
            grad[i] += coeff * rij
            grad[j] -= coeff * rij
    return grad


def _lj_energy_and_grad(pos):
    """Combined energy and gradient for L-BFGS-B."""
    return _lj_energy(pos), _lj_grad(pos)


def _get_mackay_directions():
    """Get the 55 unit direction vectors for a Mackay icosahedron and shell assignments.
    
    Returns:
        directions: (55, 3) array of unit vectors (center is [0,0,0])
        shell_ids: (55,) array of shell indices (0=center, 1=inner, 2=mid, 3=outer)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges of icosahedron (pairs at minimum distance)
    D = cdist(verts, verts)
    np.fill_diagonal(D, np.inf)
    min_dist = D.min()
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_dist) < 1e-10:
                edges.append((i, j))
    
    # 30 edge midpoints (normalized)
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # Assemble directions and shell IDs
    directions = np.zeros((55, 3))
    shell_ids = np.zeros(55, dtype=int)
    
    # Atom 0: center
    directions[0] = [0.0, 0.0, 0.0]
    shell_ids[0] = 0
    
    # Atoms 1-12: Shell 1 (icosahedron vertices)
    for i in range(12):
        directions[1 + i] = verts[i]
        shell_ids[1 + i] = 1
    
    # Atoms 13-42: Shell 2 (edge midpoints)
    for i in range(30):
        directions[13 + i] = midpoints[i]
        shell_ids[13 + i] = 2
    
    # Atoms 43-54: Shell 3 (icosahedron vertices, outer)
    for i in range(12):
        directions[43 + i] = verts[i]
        shell_ids[43 + i] = 3
    
    return directions, shell_ids


def _make_mackay_from_radii(radii):
    """Generate Mackay icosahedron positions given shell radii [r1, r2, r3]."""
    directions, shell_ids = _get_mackay_directions()
    pos = np.zeros((55, 3))
    for i in range(1, 55):
        shell = shell_ids[i]
        if shell == 1:
            pos[i] = directions[i] * radii[0]
        elif shell == 2:
            pos[i] = directions[i] * radii[1]
        elif shell == 3:
            pos[i] = directions[i] * radii[2]
    return pos


def _relax(positions, maxiter=500, gtol=1e-10):
    """L-BFGS-B local relaxation."""
    if _remaining() < 5.0:
        return positions
    
    def fun(x):
        x = x.reshape(-1, 3)
        e = _lj_energy(x)
        g = _lj_grad(x)
        return e, g.ravel()
    
    x0 = positions.reshape(-1)
    try:
        result = minimize(fun, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': gtol})
        return result.x.reshape(-1, 3)
    except Exception:
        return positions


def _polish(positions, deadline=None):
    """High-precision polish pass with progressively tighter tolerances."""
    if deadline is None:
        deadline = _DEADLINE
    
    best_energy = _lj_energy(positions)
    best_pos = positions.copy()
    
    def fun(x):
        x = x.reshape(-1, 3)
        e = _lj_energy(x)
        g = _lj_grad(x)
        return e, g.ravel()
    
    x0 = positions.reshape(-1)
    
    # Multi-stage polish: progressively tighter
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-14},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if time.time() > deadline - 5.0:
            break
        try:
            result = minimize(fun, x0, jac=True, method='L-BFGS-B', options=stage_opts)
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos


def _optimize_radii(radii_init, maxiter_nm=50, maxiter_lbfgs=150, deadline=None):
    """Optimize shell radii using Nelder-Mead, with full relaxation at each step."""
    if deadline is None:
        deadline = _DEADLINE
    
    def objective(radii):
        if time.time() > deadline - 10:
            return 1e10
        radii = np.maximum(radii, 0.5)
        pos = _make_mackay_from_radii(radii)
        pos_relaxed = _relax(pos, maxiter=maxiter_lbfgs)
        return _lj_energy(pos_relaxed)
    
    try:
        res = minimize(objective, radii_init, method='Nelder-Mead',
                       options={'maxiter': maxiter_nm, 'xatol': 1e-4, 'fatol': 1e-4})
        best_radii = res.x
        best_pos = _make_mackay_from_radii(best_radii)
        best_pos = _relax(best_pos, maxiter=maxiter_lbfgs)
        best_e = _lj_energy(best_pos)
        return best_pos, best_e, best_radii
    except Exception:
        return None, np.inf, radii_init


def _check_min_dist(pos, min_d=0.5):
    """Check all pair distances >= min_d."""
    d = pdist(pos)
    return d.min() >= min_d - 1e-9


def construct():
    global _start_time
    _start_time = time.time()
    
    best_pos = None
    best_energy = np.inf
    
    # Strategy 1: Optimize Mackay radii from multiple initial guesses
    # This is the core improvement: instead of fixed radii, we optimize them
    radius_candidates = [
        np.array([1.0, 1.8, 2.6]),
        np.array([1.1, 1.9, 2.7]),
        np.array([0.9, 1.7, 2.5]),
        np.array([1.0, 2.0, 2.8]),
        np.array([1.05, 1.85, 2.65]),
        np.array([1.12, 2.14, 3.16]),  # Based on LJ minima spacing
    ]
    
    for radii_init in radius_candidates:
        if _remaining() < 20.0:
            break
        try:
            pos, e, radii = _optimize_radii(radii_init, maxiter_nm=30, maxiter_lbfgs=100)
            if pos is not None and _check_min_dist(pos) and e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        except Exception:
            continue
    
    # Strategy 2: If we have a good candidate, do a more thorough radius optimization
    if best_pos is not None and _remaining() > 30.0:
        # Extract approximate radii from the best position
        directions, shell_ids = _get_mackay_directions()
        # Estimate current radii
        r_est = np.zeros(3)
        for shell in [1, 2, 3]:
            idx = np.where(shell_ids == shell)[0]
            if len(idx) > 0:
                r_est[shell - 1] = np.mean(np.linalg.norm(best_pos[idx], axis=1))
        
        # Refine with more iterations
        try:
            pos, e, radii = _optimize_radii(r_est, maxiter_nm=50, maxiter_lbfgs=200)
            if pos is not None and _check_min_dist(pos) and e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        except Exception:
            pass
    
    # Strategy 3: Perturbation-based search around the best Mackay configuration
    if best_pos is not None and _remaining() > 20.0:
        rng = np.random.default_rng(42)
        noise_levels = [0.01, 0.02, 0.05, 0.10]
        
        for noise in noise_levels:
            if _remaining() < 15.0:
                break
            try:
                perturbed = best_pos + rng.normal(0, noise, best_pos.shape)
                if _check_min_dist(perturbed):
                    relaxed = _relax(perturbed, maxiter=200)
                    e = _lj_energy(relaxed)
                    if _check_min_dist(relaxed) and e < best_energy:
                        best_energy = e
                        best_pos = relaxed.copy()
            except Exception:
                continue
    
    # Final high-precision polish on best result
    if best_pos is not None and _remaining() > 10.0:
        try:
            polished = _polish(best_pos, deadline=_DEADLINE)
            e = _lj_energy(polished)
            if _check_min_dist(polished) and e < best_energy:
                best_energy = e
                best_pos = polished.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: construct from default radii and relax
        default_radii = np.array([1.12, 2.14, 3.16])
        best_pos = _make_mackay_from_radii(default_radii)
        best_pos = _relax(best_pos, maxiter=300)
    
    return best_pos
# EVOLVE-BLOCK-END
