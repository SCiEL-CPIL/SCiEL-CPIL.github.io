import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 38
DEADLINE = 142.0  # Safety margin for 150s limit
START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = float(np.dot(rij, rij))
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            if np.dot(rij, rij) < min_dist * min_dist:
                return False
    return True

def _make_fcc_38_base():
    """
    Generate the 38 closest FCC lattice points (unit-scaled, nearest-neighbor = sqrt(2)).
    FCC lattice: (i, j, k) where i+j+k is even.
    Returns coordinates in lattice units (before LJ scaling).
    """
    max_coord = 5
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = float(i), float(j), float(k)
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2, (i, j, k)))
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    selected = np.array([p[:3] for p in points[:N]], dtype=float)
    return selected

def make_fcc_truncated_octahedron():
    """
    Generate the FCC truncated octahedron cluster for 38 atoms at LJ equilibrium.
    Scale factor: s * sqrt(2) = 2^(1/6) => s = 2^(-1/3).
    """
    scale = 2.0 ** (-1.0 / 3.0)
    base = _make_fcc_38_base()
    coords = base * scale
    return coords

def make_fcc_with_noise(noise_scale=0.04, seed=42):
    """FCC base with small Gaussian noise to escape symmetry traps."""
    base = make_fcc_truncated_octahedron()
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, noise_scale, base.shape)

def _make_icosahedral(n_atoms=38):
    """Icosahedral-inspired initial configuration: two concentric shells + center."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append((sx, sy * phi, sz * phi))
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    outer_r = 2.8
    inner_r = 1.3
    outer = verts * outer_r
    ang = math.pi / 6.0
    R = np.array([[math.cos(ang), -math.sin(ang), 0],
                  [math.sin(ang), math.cos(ang), 0],
                  [0, 0, 1]])
    inner = verts @ R.T
    inner = inner * inner_r
    center = np.array([[0.0, 0.0, 0.0]])
    pts = np.vstack([center, inner, outer])
    if len(pts) >= n_atoms:
        pts = pts[:n_atoms]
    else:
        rng = np.random.default_rng(0)
        extra = rng.normal(0, 0.5, (n_atoms - len(pts), 3))
        pts = np.vstack([pts, extra])
    pts *= 2.0 ** (1.0 / 6.0) / 1.4
    return pts

def _make_initial_clusters():
    """Build a diverse set of initial candidate configurations."""
    clusters = []
    # Primary: perfect FCC truncated octahedron
    try:
        fcc = make_fcc_truncated_octahedron()
        if check_validity(fcc):
            clusters.append(fcc)
    except Exception:
        pass
    # Small-noise FCC variants (escape symmetry, stay near FCC basin)
    for s in (1, 2, 3, 4, 5, 6):
        try:
            c = make_fcc_with_noise(noise_scale=0.04, seed=s)
            if check_validity(c):
                clusters.append(c)
        except Exception:
            pass
    # Larger-noise FCC variants
    for s in (7, 8, 9, 10):
        try:
            c = make_fcc_with_noise(noise_scale=0.10, seed=s)
            if check_validity(c):
                clusters.append(c)
        except Exception:
            pass
    # Icosahedral-inspired (different basin)
    try:
        ic = _make_icosahedral(N)
        if check_validity(ic):
            clusters.append(ic)
    except Exception:
        pass
    return clusters

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient vectorized."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    return energy, grad_flat

def relax(positions, maxiter=500, gtol=1e-8):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-12})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def optimize_scale(positions, max_iter=200):
    """Optimize the global scale factor s to minimize energy E(s * positions)."""
    def energy_at_scale(s):
        scaled = positions * s
        return lj_energy_vectorized(scaled)
    res = minimize_scalar(energy_at_scale, bounds=(0.7, 1.4), method='bounded',
                          options={'maxiter': max_iter, 'xatol': 1e-12})
    if res.success:
        return positions * res.x
    return positions

def optimize_fcc_distortion(positions, max_iter=300):
    """
    Optimize anisotropic distortion of the FCC lattice to better match the
    true global-minimum geometry (surface effects break perfect cubic symmetry).
    We scale x, y, z independently around the FCC base and relax.
    """
    base = _make_fcc_38_base()
    scale = 2.0 ** (-1.0 / 3.0)

    def distorted_energy(params):
        sx, sy, sz = params
        # Apply distortion to the FCC base lattice, then LJ scale
        distorted = base.copy()
        distorted[:, 0] *= sx
        distorted[:, 1] *= sy
        distorted[:, 2] *= sz
        distorted = distorted * scale
        # Quick relaxation to let it settle
        try:
            rel = relax(distorted, maxiter=100, gtol=1e-6)
            return lj_energy_vectorized(rel)
        except Exception:
            return 1e20

    x0 = np.array([1.0, 1.0, 1.0])
    try:
        res = minimize(distorted_energy, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6,
                                'adaptive': True})
        sx, sy, sz = res.x
        distorted = base.copy()
        distorted[:, 0] *= sx
        distorted[:, 1] *= sy
        distorted[:, 2] *= sz
        distorted = distorted * scale
        return distorted
    except Exception:
        return positions

def polish(positions, maxiter=5000, gtol=1e-12):
    """Final high-precision polish pass with very tight tolerances."""
    x0 = positions.flatten()
    try:
        # First, do a scale optimization
        scaled = optimize_scale(positions)
        if not check_validity(scaled):
            scaled = positions
        x0 = scaled.flatten()
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200,
                                   'ftol': 1e-15, 'iprint': 0})
        polished = result.x.reshape(N, 3)
        if check_validity(polished):
            return polished
        return positions
    except Exception:
        return positions

def basin_hopping(positions, n_iter=20, step_size=0.25, maxiter=150, temperature=0.03):
    """Perform basin hopping optimization with analytic gradient."""
    x0 = positions.flatten()
    current_energy, _ = _lj_energy_and_grad(x0)
    best_energy = current_energy
    best_x = x0.copy()
    rng = np.random.default_rng(42)
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 15:
            break
        noise = rng.normal(0, step_size, x0.shape)
        perturbed = x0 + noise
        try:
            result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                              jac=True,
                              options={'maxiter': maxiter, 'gtol': 1e-6, 'maxls': 50})
            new_x = result.x
            new_energy, _ = _lj_energy_and_grad(new_x)
        except Exception:
            continue
        if new_energy < current_energy:
            x0 = new_x
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()
        else:
            delta = new_energy - current_energy
            if delta < 0:
                accept = 1.0
            else:
                accept = math.exp(-delta / temperature)
            if rng.uniform() < accept:
                x0 = new_x
                current_energy = new_energy
    return best_x.reshape(N, 3), best_energy

def construct():
    global START_TIME
    START_TIME = time.time()

    best_energy = float('inf')
    best_positions = None

    def try_candidate(cand):
        nonlocal best_energy, best_positions
        if not check_validity(cand):
            return
        try:
            relaxed = relax(cand, maxiter=400, gtol=1e-8)
            if not check_validity(relaxed):
                return
            scaled = optimize_scale(relaxed)
            if check_validity(scaled):
                relaxed2 = relax(scaled, maxiter=400, gtol=1e-9)
                if check_validity(relaxed2):
                    energy = lj_energy_vectorized(relaxed2)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed2
        except Exception:
            pass

    # 1. Multi-candidate initial search (FCC + noise variants + icosahedral)
    candidates = _make_initial_clusters()
    for cand in candidates:
        if time.time() - START_TIME > DEADLINE - 30:
            break
        try_candidate(cand)

    # 2. FCC distortion optimization from the best candidate
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 25:
        try:
            distorted = optimize_fcc_distortion(best_positions, max_iter=200)
            if check_validity(distorted):
                d_relax = relax(distorted, maxiter=500, gtol=1e-9)
                if check_validity(d_relax):
                    d_scaled = optimize_scale(d_relax)
                    if check_validity(d_scaled):
                        d_final = relax(d_scaled, maxiter=500, gtol=1e-10)
                        if check_validity(d_final):
                            energy = lj_energy_vectorized(d_final)
                            if energy < best_energy:
                                best_energy = energy
                                best_positions = d_final
        except Exception:
            pass

    # 3. Basin hopping from the best result
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 30:
        try:
            bh_result, bh_energy = basin_hopping(best_positions, n_iter=25, step_size=0.25,
                                                 maxiter=150, temperature=0.03)
            if check_validity(bh_result):
                scaled_bh = optimize_scale(bh_result)
                if check_validity(scaled_bh):
                    final_bh = relax(scaled_bh, maxiter=500, gtol=1e-9)
                    if check_validity(final_bh):
                        energy = lj_energy_vectorized(final_bh)
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = final_bh
        except Exception:
            pass

    # 4. Final high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 5:
        try:
            polished = polish(best_positions, maxiter=10000, gtol=1e-13)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass

    # 5. Ultra-tight final polish if time permits
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 1:
        try:
            scaled_final = optimize_scale(best_positions)
            if check_validity(scaled_final):
                x0 = scaled_final.flatten()
                result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                                  jac=True,
                                  options={'maxiter': 20000, 'gtol': 1e-15, 'maxls': 500,
                                           'ftol': 1e-18})
                ultra_polished = result.x.reshape(N, 3)
                if check_validity(ultra_polished):
                    energy = lj_energy_vectorized(ultra_polished)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = ultra_polished
        except Exception:
            pass

    # Fallback
    if best_positions is None:
        try:
            fcc = make_fcc_truncated_octahedron()
            if check_validity(fcc):
                best_positions = fcc
            else:
                best_positions = np.zeros((N, 3))
                for i in range(N):
                    best_positions[i] = [i * 0.6, (i % 3) * 0.6, (i // 3) % 3 * 0.6]
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i * 0.6, (i % 3) * 0.6, (i // 3) % 3 * 0.6]

    return best_positions
