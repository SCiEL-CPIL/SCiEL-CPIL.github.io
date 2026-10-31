"""
LJ38 cluster optimisation: construct a near-optimal truncated-octahedron
geometry and refine with L-BFGS relaxation.
Uses symmetry-adapted strain parameterization to reduce search space.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 38

# Deadline guard
_START = time.time()
_TIME_LIMIT = 110.0  # seconds; leaves margin before 120 s kill


def _elapsed():
    return time.time() - _START


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _lj_energy(pos):
    """Total Lennard-Jones energy 4*sum(r^-12 - r^-6)."""
    n = pos.shape[0]
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)  # avoid self-interaction
    r6 = r2 ** 3
    r12 = r6 * r6
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6)
    return energy


def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r6_inv = 1.0 / r6
    r12_inv = r6_inv * r6_inv
    energy = 4.0 * np.sum(r12_inv - r6_inv)
    # Gradient: dE/dx_i = sum_j 8*(-12/r12_ij + 6/r6_ij) * (x_i - x_j)
    coeff = 8.0 * (-12.0 * r12_inv + 6.0 * r6_inv)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _lj_grad(pos):
    """Gradient of LJ energy."""
    _, grad = _lj_energy_and_grad(pos)
    return grad


def _enforce_min_separation(pos, min_dist=0.501):
    """Push atoms apart if any pair is closer than min_dist."""
    for _ in range(200):
        n = pos.shape[0]
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2, np.inf)
        r = np.sqrt(r2)
        # Find violations
        mask = (r < min_dist) & (r > 1e-12)
        if not np.any(mask):
            break
        # Push apart
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    d = pos[i] - pos[j]
                    dist = np.linalg.norm(d)
                    if dist < 1e-12:
                        # random small push
                        d = np.random.randn(3) * 0.01
                        dist = np.linalg.norm(d)
                    correction = (min_dist - dist) * d / dist * 0.5
                    pos[i] += correction
                    pos[j] -= correction
    return pos


def _check_validity(pos):
    """Check if positions are valid (finite, min separation)."""
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    return min_r >= 0.5


def _build_fcc_truncated_octahedron(scale=1.1225):
    """
    Build 38-atom FCC truncated octahedron.
    Uses the FCC lattice and picks the 38 closest points to origin.
    """
    d = scale  # desired NN distance
    a = d * np.sqrt(2.0)  # cubic lattice constant
    
    # Generate on the FCC lattice directly
    # FCC lattice: all vectors (n1, n2, n3) * a/2 where n1+n2+n3 is even
    pts = []
    R = int(np.ceil(3.0 * a / (a/2)))  # radius in units of a/2
    for n1 in range(-R, R + 1):
        for n2 in range(-R, R + 1):
            for n3 in range(-R, R + 1):
                if (n1 + n2 + n3) % 2 == 0:
                    pt = np.array([n1, n2, n3]) * (a / 2.0)
                    pts.append(pt)
    
    pts = np.array(pts)
    # Sort by distance from origin
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    # Take 38 closest
    cluster = pts[:38].copy()
    
    # Center of mass to origin
    cluster -= np.mean(cluster, axis=0)
    
    return cluster


def _build_fcc_with_strain(params):
    """
    Build FCC truncated octahedron with symmetry-adapted strain.
    params: [scale, exx, exy, exz, eyy, eyz, ezz] - 7 parameters
    This reduces the 114-dim search to 7 dims while preserving FCC topology.
    """
    scale, exx, exy, exz, eyy, eyz, ezz = params
    
    d = scale
    a = d * np.sqrt(2.0)
    
    pts = []
    R = int(np.ceil(3.0 * a / (a/2)))
    for n1 in range(-R, R + 1):
        for n2 in range(-R, R + 1):
            for n3 in range(-R, R + 1):
                if (n1 + n2 + n3) % 2 == 0:
                    pt = np.array([n1, n2, n3]) * (a / 2.0)
                    pts.append(pt)
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    cluster = pts[:38].copy()
    cluster -= np.mean(cluster, axis=0)
    
    # Apply strain tensor
    strain = np.array([
        [1.0 + exx, exy, exz],
        [exy, 1.0 + eyy, eyz],
        [exz, eyz, 1.0 + ezz]
    ])
    cluster = cluster @ strain.T
    
    return cluster


def _relax(pos, max_time=30.0, tight_tol=False):
    """L-BFGS relaxation with time limit."""
    t0 = time.time()
    
    def objective(x):
        if time.time() - t0 > max_time:
            raise TimeoutError
        return _lj_energy(x)
    
    def gradient(x):
        if time.time() - t0 > max_time:
            raise TimeoutError
        return _lj_grad(x)
    
    gtol = 1e-10 if tight_tol else 1e-8
    ftol = 1e-12 if tight_tol else 1e-10
    maxiter = 5000 if tight_tol else 2000
    
    try:
        result = minimize(
            objective, pos, jac=gradient, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 10000, 'maxls': 50,
                     'gtol': gtol, 'ftol': ftol}
        )
        return result.x
    except (TimeoutError, Exception):
        return pos


def _optimize_strain(base_scale=2.0**(1/6)):
    """
    Optimize the 7 strain parameters using differential evolution.
    This explores the low-dimensional manifold of FCC-preserving deformations.
    """
    t0 = time.time()
    time_budget = min(_remaining() - 20.0, 60.0)
    if time_budget < 10.0:
        return None
    
    def strain_energy(params):
        if time.time() - t0 > time_budget:
            raise TimeoutError
        pos = _build_fcc_with_strain(params)
        pos = _enforce_min_separation(pos)
        if not _check_validity(pos):
            return 1e10
        e, _ = _lj_energy_and_grad(pos)
        return e
    
    # Parameter bounds: scale in [1.1, 1.2], strains in [-0.1, 0.1]
    bounds = [
        (1.10, 1.20),  # scale
        (-0.10, 0.10),  # exx
        (-0.10, 0.10),  # exy
        (-0.10, 0.10),  # exz
        (-0.10, 0.10),  # eyy
        (-0.10, 0.10),  # eyz
        (-0.10, 0.10),  # ezz
    ]
    
    try:
        result = differential_evolution(
            strain_energy, bounds,
            maxiter=50, popsize=15, seed=42,
            tol=1e-8, polish=False
        )
        if result.success and result.fun < 1e9:
            pos = _build_fcc_with_strain(result.x)
            pos = _enforce_min_separation(pos)
            return pos
    except (TimeoutError, Exception):
        pass
    
    return None


def construct():
    """
    Construct LJ38 cluster near global minimum.
    Returns array of shape (38, 3).
    """
    # The optimal NN distance for LJ is 2^(1/6) ≈ 1.12246
    scale = 2.0 ** (1.0 / 6.0)
    
    # Build initial FCC truncated octahedron
    pos = _build_fcc_truncated_octahedron(scale)
    pos = _enforce_min_separation(pos)
    
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    # Stage 1: Quick relaxation to get a good starting point
    if _remaining() > 15.0:
        pos = _relax(pos, max_time=min(_remaining() - 10.0, 30.0))
        pos = _enforce_min_separation(pos)
        if _check_validity(pos):
            e = _lj_energy(pos)
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
    
    # Stage 2: Optimize strain parameters (low-dimensional manifold search)
    if _remaining() > 25.0:
        strain_pos = _optimize_strain(scale)
        if strain_pos is not None and _check_validity(strain_pos):
            strain_pos = _relax(strain_pos, max_time=min(_remaining() - 15.0, 30.0))
            strain_pos = _enforce_min_separation(strain_pos)
            if _check_validity(strain_pos):
                e = _lj_energy(strain_pos)
                if e < best_energy:
                    best_energy = e
                    best_pos = strain_pos.copy()
    
    # Stage 3: Multi-start with different scales
    if _remaining() > 20.0:
        for s in [scale * 0.995, scale * 1.005, scale * 0.99, scale * 1.01]:
            if _remaining() < 12.0:
                break
            pos2 = _build_fcc_truncated_octahedron(s)
            pos2 = _enforce_min_separation(pos2)
            pos2 = _relax(pos2, max_time=min(_remaining() - 10.0, 25.0))
            pos2 = _enforce_min_separation(pos2)
            if _check_validity(pos2):
                e = _lj_energy(pos2)
                if e < best_energy:
                    best_energy = e
                    best_pos = pos2.copy()
    
    # Stage 4: High-precision polish on the best result
    if _remaining() > 10.0:
        best_pos = _relax(best_pos, max_time=min(_remaining() - 5.0, 30.0), tight_tol=True)
        best_pos = _enforce_min_separation(best_pos)
    
    # Final separation check and validity
    best_pos = _enforce_min_separation(best_pos)
    if not _check_validity(best_pos):
        # Fallback to a valid configuration
        fallback = _build_fcc_truncated_octahedron(scale)
        fallback = _enforce_min_separation(fallback)
        best_pos = fallback
    
    return best_pos
