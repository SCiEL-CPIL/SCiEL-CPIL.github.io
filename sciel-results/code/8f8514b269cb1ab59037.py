import numpy as np
import time
import math
from scipy.optimize import minimize as _minimize

N = 38
DEADLINE = 140.0
START_TIME = time.time()

def lj_energy_vectorized(positions):
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return 4.0 * np.sum(inv_r12 - inv_r6) / 2.0

def check_validity(positions, min_dist=0.5):
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            if np.dot(rij, rij) < min_dist * min_dist:
                return False
    return True

def _lj_energy_and_grad(pos_flat):
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

def _generate_fcc_lattice():
    """Generate FCC lattice points and sort by radius."""
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 8
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    return points

def _get_fcc_structure(n_atoms=38):
    """
    Return the 38 closest FCC points.
    For LJ38: 1 center + 12 inner shell + 25 outer shell sites (from 24 available + center counted)
    Actually: FCC has 1 center, 12 at r=sqrt(2)*scale, 6 at r=2*scale, 24 at r=sqrt(6)*scale, ...
    Let me recount: for FCC with a=1/sqrt(2):
    - r=0: 1 atom (center)
    - r=sqrt(2)/sqrt(2)=1: 12 atoms (face centers) -- wait no
    Let me just compute properly.
    FCC lattice points: (i,j,k) with i+j+k even, scaled by 1/sqrt(2)
    - (0,0,0): r=0, 1 point
    - (±1,±1,0) and perms: r=sqrt(2)/sqrt(2)=1, 12 points
    - (±2,0,0) and perms: r=2/sqrt(2)=sqrt(2), 6 points
    - (±2,±2,0) and perms: r=sqrt(8)/sqrt(2)=2, 12 points
    - (±2,±1,±1) and perms: r=sqrt(6)/sqrt(2)=sqrt(3), 24 points
    - (±3,±1,0) and perms: r=sqrt(10)/sqrt(2)=sqrt(5), 24 points
    - (±4,0,0) and perms: r=4/sqrt(2)=2*sqrt(2), 6 points
    - (±3,±3,0) and perms: r=sqrt(18)/sqrt(2)=3, 12 points
    - (±4,±2,0) and perms: r=sqrt(20)/sqrt(2)=2*sqrt(5), 24 points
    - (±4,±2,±2) and perms: r=sqrt(24)/sqrt(2)=2*sqrt(6), 24 points
    
    So cumulative: 1, 13, 19, 31, 55, ...
    First 38: 1 + 12 + 6 + 12 + 7 from the 24 at r=sqrt(3)
    So we have shells: r=0 (1), r=1 (12), r=sqrt(2) (6), r=2 (12), r=sqrt(3) (24 available, need 7)
    
    Wait, that's only 1+12+6+12+7 = 38. But the truncated octahedron for LJ38 is typically described as
    having a specific structure. Let me just use the 38 closest FCC points.
    """
    points = _generate_fcc_lattice()
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    
    # Assign shells based on radial distance
    radii = np.linalg.norm(coords, axis=1)
    unique_r = np.sort(np.unique(np.round(radii, 4)))
    shell_ids = np.zeros(n_atoms, dtype=int)
    for i, r in enumerate(radii):
        for sid, ur in enumerate(unique_r):
            if abs(r - ur) < 0.05:
                shell_ids[i] = sid
                break
    return coords, shell_ids, unique_r

def _symmetry_adapted_energy(params):
    """Compute energy for symmetry-adapted parameters."""
    n_atoms = N
    global_scale = params[0]
    shell_params = params[1:]
    
    base_coords, shell_ids, unique_r = _get_fcc_structure(n_atoms)
    
    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        positions[i] = base_coords[i] * breath * global_scale
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return 4.0 * np.sum(inv_r12 - inv_r6) / 2.0

def _symmetry_adapted_energy_and_grad(params):
    energy = _symmetry_adapted_energy(params)
    eps = 1e-6
    grad = np.zeros_like(params)
    for k in range(len(params)):
        params_plus = params.copy()
        params_plus[k] += eps
        e_plus = _symmetry_adapted_energy(params_plus)
        params_minus = params.copy()
        params_minus[k] -= eps
        e_minus = _symmetry_adapted_energy(params_minus)
        grad[k] = (e_plus - e_minus) / (2 * eps)
    return energy, grad

def optimize_symmetry_adapted(initial_scale=None, maxiter=500):
    base_coords, shell_ids, unique_r = _get_fcc_structure(N)
    n_shells = len(unique_r)
    
    if initial_scale is None:
        initial_scale = 2.0 ** (1.0/6.0)
    
    x0 = np.array([initial_scale] + [0.0] * n_shells)
    
    try:
        result = _minimize(_symmetry_adapted_energy_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': maxiter, 'gtol': 1e-12, 'maxls': 150, 'ftol': 1e-16})
        return result.x
    except Exception:
        return x0

def _make_fcc_start():
    """Make the standard FCC truncated octahedron starting point."""
    base_coords, shell_ids, unique_r = _get_fcc_structure(N)
    coords = base_coords * 2.0 ** (1.0/6.0)
    return coords

def _make_fcc_with_noise(noise_scale=0.04, seed=42):
    base = _make_fcc_start()
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, noise_scale, base.shape)

def _make_icosahedral(n_atoms=38):
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append((sx, sy*phi, sz*phi))
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    outer_r = 2.8
    inner_r = 1.3
    outer = verts * outer_r
    inner = verts * inner_r
    ang = math.pi / 6.0
    R = np.array([[math.cos(ang), -math.sin(ang), 0],
                  [math.sin(ang), math.cos(ang), 0],
                  [0, 0, 1]])
    inner = inner @ R.T
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

def _relax(positions, maxiter=400, gtol=1e-9):
    x0 = positions.flatten()
    try:
        result = _minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 60, 'ftol': 1e-12})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def _polish(positions, maxiter=3000, gtol=1e-14):
    x0 = positions.flatten()
    try:
        result = _minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-16})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def _basin_hopping(positions, n_iter=15, step_size=0.4, maxiter=120, temperature=0.05):
    x0 = positions.flatten()
    current_energy, _ = _lj_energy_and_grad(x0)
    best_energy = current_energy
    best_x = x0.copy()
    rng = np.random.default_rng(99)

    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 12:
            break
        noise = rng.normal(0, step_size, x0.shape)
        perturbed = x0 + noise
        try:
            result = _minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': maxiter, 'gtol': 1e-7, 'maxls': 60})
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

def _optimize_global_scale(positions, maxiter=300):
    def scale_energy(scale):
        p = positions * scale
        return lj_energy_vectorized(p)
    
    x0 = np.array([1.0])
    try:
        result = _minimize(scale_energy, x0, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-14})
        return result.x[0]
    except Exception:
        return 1.0

def construct():
    global START_TIME
    START_TIME = time.time()

    best_energy = float('inf')
    best_positions = None

    def _try_candidate(cand, label=""):
        nonlocal best_energy, best_positions
        if time.time() - START_TIME > DEADLINE - 30:
            return
        try:
            if not check_validity(cand):
                return
            relaxed = _relax(cand, maxiter=300, gtol=1e-9)
            if check_validity(relaxed):
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass

    # Generate diverse initial candidates
    candidates = []

    # 1. Standard FCC
    try:
        fcc = _make_fcc_start()
        candidates.append(fcc)
    except Exception:
        pass

    # 2. Symmetry-adapted optimized FCC
    try:
        params_opt = optimize_symmetry_adapted(initial_scale=2.0 ** (1.0/6.0), maxiter=300)
        base_coords, shell_ids, unique_r = _get_fcc_structure(N)
        global_scale = params_opt[0]
        shell_params = params_opt[1:]
        positions = np.zeros((N, 3))
        for i in range(N):
            s = shell_ids[i]
            if s < len(shell_params):
                breath = 1.0 + shell_params[s]
            else:
                breath = 1.0
            positions[i] = base_coords[i] * breath * global_scale
        candidates.append(positions)
    except Exception:
        pass

    # 3. Symmetry-adapted with perturbed initial scale
    for scale_try in [0.99 * 2.0 ** (1.0/6.0), 1.01 * 2.0 ** (1.0/6.0), 0.98 * 2.0 ** (1.0/6.0), 1.02 * 2.0 ** (1.0/6.0)]:
        try:
            params_opt = optimize_symmetry_adapted(initial_scale=scale_try, maxiter=200)
            base_coords, shell_ids, unique_r = _get_fcc_structure(N)
            global_scale = params_opt[0]
            shell_params = params_opt[1:]
            positions = np.zeros((N, 3))
            for i in range(N):
                s = shell_ids[i]
                if s < len(shell_params):
                    breath = 1.0 + shell_params[s]
                else:
                    breath = 1.0
                positions[i] = base_coords[i] * breath * global_scale
            candidates.append(positions)
        except Exception:
            pass

    # 4. Noisy FCC variants
    for s in range(1, 8):
        try:
            c = _make_fcc_with_noise(noise_scale=0.04, seed=s)
            candidates.append(c)
        except Exception:
            pass

    # 5. Larger noise FCC
    for s in range(8, 12):
        try:
            c = _make_fcc_with_noise(noise_scale=0.10, seed=s)
            candidates.append(c)
        except Exception:
            pass

    # 6. Icosahedral
    try:
        ic = _make_icosahedral(N)
        candidates.append(ic)
    except Exception:
        pass

    # 7. BCC-like
    try:
        side = int(math.ceil(N ** (1.0/3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append((i - side//2, j - side//2, k - side//2))
        pts.sort(key=lambda p: p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
        pts = pts[:N]
        arr = np.array(pts, dtype=float)
        arr *= 1.28 * 2.0 ** (1.0/6.0)
        candidates.append(arr)
    except Exception:
        pass

    # Relax all candidates
    for cand in candidates:
        _try_candidate(cand)

    # Global scale optimization on best
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 35:
        try:
            optimal_scale = _optimize_global_scale(best_positions, maxiter=300)
            scaled = best_positions * optimal_scale
            if check_validity(scaled):
                energy = lj_energy_vectorized(scaled)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = scaled
        except Exception:
            pass

    # Basin hopping passes
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 30:
        try:
            bh_res, bh_e = _basin_hopping(best_positions, n_iter=20, step_size=0.3, maxiter=120, temperature=0.05)
            if check_validity(bh_res) and bh_e < best_energy:
                best_energy = bh_e
                best_positions = bh_res
        except Exception:
            pass

    if best_positions is not None and time.time() - START_TIME < DEADLINE - 25:
        try:
            bh_res2, bh_e2 = _basin_hopping(best_positions, n_iter=14, step_size=0.6, maxiter=100, temperature=0.1)
            if check_validity(bh_res2) and bh_e2 < best_energy:
                best_energy = bh_e2
                best_positions = bh_res2
        except Exception:
            pass

    # Symmetry-adapted re-optimization from best
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            base_coords, shell_ids, unique_r = _get_fcc_structure(N)
            radii_best = np.linalg.norm(best_positions, axis=1)
            radii_base = np.linalg.norm(base_coords, axis=1)
            mask = radii_base > 1e-6
            scale_est = np.sum(radii_best[mask] * radii_base[mask]) / np.sum(radii_base[mask] * radii_base[mask])
            shell_params_est = np.zeros(len(unique_r))
            for s, ur in enumerate(unique_r):
                shell_mask = (shell_ids == s) & (radii_base > 1e-6)
                if np.any(shell_mask):
                    shell_params_est[s] = np.mean(radii_best[shell_mask] / (scale_est * radii_base[shell_mask])) - 1.0
            
            params_opt = optimize_symmetry_adapted(initial_scale=scale_est, maxiter=400)
            global_scale = params_opt[0]
            shell_params = params_opt[1:]
            positions = np.zeros((N, 3))
            for i in range(N):
                s = shell_ids[i]
                if s < len(shell_params):
                    breath = 1.0 + shell_params[s]
                else:
                    breath = 1.0
                positions[i] = base_coords[i] * breath * global_scale
            
            if check_validity(positions):
                energy = lj_energy_vectorized(positions)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Final polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 15:
        try:
            polished = _polish(best_positions, maxiter=5000, gtol=1e-16)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass

    # Short basin hop after polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            bh_final, bh_final_e = _basin_hopping(best_positions, n_iter=10, step_size=0.15, maxiter=80, temperature=0.01)
            if check_validity(bh_final) and bh_final_e < best_energy:
                best_energy = bh_final_e
                best_positions = bh_final
                if time.time() - START_TIME < DEADLINE - 5:
                    polished2 = _polish(best_positions, maxiter=3000, gtol=1e-16)
                    if check_validity(polished2):
                        energy2 = lj_energy_vectorized(polished2)
                        if energy2 < best_energy:
                            best_energy = energy2
                            best_positions = polished2
        except Exception:
            pass

    if best_positions is None:
        try:
            best_positions = _make_fcc_start()
        except Exception:
            side = int(math.ceil(N ** (1/3)))
            pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
            best_positions = np.array(pts[:N], dtype=float) * 1.3

    return best_positions
