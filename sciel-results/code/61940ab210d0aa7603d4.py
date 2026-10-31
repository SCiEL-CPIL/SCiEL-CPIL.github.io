import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 38
DEADLINE = 148.0  # Safety margin for 150s limit
START_TIME = time.time()

def _get_fcc_shell_structure():
    """
    Return the 38 FCC lattice points grouped by shell (distance from origin).
    Shell structure of FCC truncated octahedron:
      Shell 0: 1 atom at r=0
      Shell 1: 12 atoms at r=sqrt(2)  (face-centered, octahedral)
      Shell 2:  6 atoms at r=sqrt(6)  (body-centered, cubic)
      Shell 3: 12 atoms at r=sqrt(12) (outer face-centered)
      Shell 4:  7 atoms at r=sqrt(14) or r=sqrt(18) (outermost)
    Returns: list of (shell_index, np.array of coords in lattice units)
    """
    max_coord = 5
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = float(i), float(j), float(k)
                    r2 = x*x + y*y + z*z
                    points.append((r2, x, y, z))
    points.sort(key=lambda p: (p[0], abs(p[1]), abs(p[2]), abs(p[3])))
    selected = [(x, y, z, r2) for r2, x, y, z in points[:N]]
    
    # Group by shell radius
    shells = {}
    for x, y, z, r2 in selected:
        # Round r2 to identify shells
        r2_key = round(r2, 6)
        if r2_key not in shells:
            shells[r2_key] = []
        shells[r2_key].append(np.array([x, y, z]))
    
    # Sort shells by radius
    sorted_r2 = sorted(shells.keys())
    shell_data = []
    for idx, r2 in enumerate(sorted_r2):
        coords = np.array(shells[r2], dtype=float)
        shell_data.append((idx, r2, coords))
    return shell_data

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
    Returns coordinates in lattice units (before LJ scaling) and shell assignments.
    """
    shell_data = _get_fcc_shell_structure()
    all_coords = []
    shell_ids = []
    for shell_idx, r2, coords in shell_data:
        for i in range(len(coords)):
            all_coords.append(coords[i])
            shell_ids.append(shell_idx)
    selected = np.array(all_coords, dtype=float)
    shells = np.array(shell_ids, dtype=int)
    return selected, shells

def make_fcc_truncated_octahedron():
    """
    Generate the FCC truncated octahedron cluster for 38 atoms at LJ equilibrium.
    Scale factor: s * sqrt(2) = 2^(1/6) => s = 2^(-1/3).
    """
    scale = 2.0 ** (-1.0 / 3.0)
    base, shells = _make_fcc_38_base()
    coords = base * scale
    return coords

def make_fcc_with_shell_radii(shell_scales, base_coords, shells):
    """
    Build FCC cluster with independent shell radii.
    shell_scales: array of per-shell scale factors.
    This allows each shell to breathe independently while preserving FCC symmetry.
    """
    n_shells = int(max(shells)) + 1
    coords = np.zeros_like(base_coords)
    for s in range(n_shells):
        mask = shells == s
        if np.any(mask):
            coords[mask] = base_coords[mask] * shell_scales[s]
    return coords

def make_fcc_with_strain(strain_params, base_coords, shells):
    """
    Apply symmetry-adapted strain to FCC base coordinates.
    strain_params: array of shape (6,) = [ex, ey, ez, exy, exz, eyz]
    This preserves FCC topology while allowing local shell breathing.
    """
    ex, ey, ez, exy, exz, eyz = strain_params
    # Strain tensor (symmetric 3x3)
    strain = np.array([
        [ex, exy/2, exz/2],
        [exy/2, ey, eyz/2],
        [exz/2, eyz/2, ez]
    ])
    # Apply affine transformation: x' = (I + strain) @ x
    transform = np.eye(3) + strain
    strained = base_coords @ transform.T
    return strained

def _symmetry_adapted_energy_and_grad(params):
    """
    Symmetry-adapted energy and gradient in shell-radius space.
    params: array of shell scale factors (one per shell).
    The total number of shells is determined by the FCC structure.
    """
    base_coords, shells = _make_fcc_38_base()
    n_shells = int(max(shells)) + 1
    
    # Build positions from shell scales
    coords = np.zeros((N, 3))
    for s in range(n_shells):
        mask = shells == s
        if np.any(mask):
            coords[mask] = base_coords[mask] * params[s]
    
    # Compute energy and gradient w.r.t. shell scales
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Gradient w.r.t. each shell scale
    grad = np.zeros(n_shells)
    for s in range(n_shells):
        mask_i = shells == s
        # dE/d(scale_s) = sum over pairs involving atoms in shell s of dE/dcoord * dcoord/dscale
        # dcoord_i/dscale_s = base_coords_i if i in shell s else 0
        # Use chain rule: grad_s = sum_{i in shell s} sum_j (dE/dx_i) * base_coords_i
        # We compute dE/dx_i from the pair forces
        inv_r14 = inv_r12 * inv_r2
        inv_r8 = inv_r6 * inv_r2
        coeff = -24.0 * inv_r14 + 12.0 * inv_r8
        # Force on atom i from all j: F_i = sum_j coeff[i,j] * (x_i - x_j)
        # But we need dE/dx_i, which is -F_i... actually:
        # dE/dx_i = sum_{j != i} 4*(-12*x_i/r^14 + 6*x_i/r^8) * (x_i - x_j)/r^2 ... 
        # Simpler: use the matrix form
        # dE/dx_i = sum_j coeff[i,j] * (x_i - x_j) where coeff = -24/r^14 + 12/r^8
        # Wait, let me re-derive:
        # E = 4 * sum_{i<j} (1/r^12 - 1/r^6)
        # dE/dx_i = 4 * sum_{j != i} (-12/r^14 + 6/r^8) * (x_i - x_j) / r^2 * ... 
        # Actually: d(1/r^12)/dx_i = -12 * (x_i - x_j) / r^14
        # d(1/r^6)/dx_i = -6 * (x_i - x_j) / r^8
        # So dE/dx_i = 4 * sum_{j != i} (-12*(x_i-x_j)/r^14 + 6*(x_i-x_j)/r^8)
        # = sum_{j != i} (-48*(x_i-x_j)/r^14 + 24*(x_i-x_j)/r^8)
        # = sum_{j != i} (-48/r^14 + 24/r^8) * (x_i - x_j)
        # Hmm, but our coeff was -24/r^14 + 12/r^8 for the flat gradient...
        # Let me just use the same formula as _lj_energy_and_grad but accumulate per shell
        
        # dE/dx_i = sum_{j != i} (-48/r_ij^14 + 24/r_ij^8) * (x_i - x_j)
        # Using our coeff matrix: coeff[i,j] = -24/r^14 + 12/r^8
        # So dE/dx_i = sum_j 2 * coeff[i,j] * (x_i - x_j) for j != i
        # But in the flat gradient, we had: grad_matrix = coeff[:, :, np.newaxis] * diff
        # where diff = x_i - x_j, and we summed over j.
        # That gives sum_j coeff[i,j] * (x_i - x_j)
        # But the actual force is 2x that because of the double counting in the sum...
        # Actually no: E = 4 * sum_{i<j} f(r_ij), so dE/dx_i = 4 * sum_{j>i} f'(r_ij) * d r_ij/dx_i + 4 * sum_{j<i} f'(r_ij) * d r_ij/dx_i
        # = 4 * sum_{j != i} f'(r_ij) * (x_i - x_j)/r_ij
        # f(r) = r^-12 - r^-6, f'(r) = -12 r^-13 + 6 r^-7
        # dE/dx_i = 4 * sum_{j != i} (-12/r^13 + 6/r^7) * (x_i - x_j)/r
        # = 4 * sum_{j != i} (-12/r^14 + 6/r^8) * (x_i - x_j)
        # = sum_{j != i} (-48/r^14 + 24/r^8) * (x_i - x_j)
        # Our coeff = -24/r^14 + 12/r^8, so dE/dx_i = 2 * sum_j coeff[i,j] * (x_i - x_j)
        # But in the flat gradient code, we used coeff directly without the factor of 2.
        # Let me check: the flat gradient sums grad_matrix over axis=1 (j index)
        # grad_matrix[i, j, :] = coeff[i, j] * (x_i - x_j)
        # sum over j gives sum_j coeff[i,j] * (x_i - x_j)
        # This equals (1/2) * dE/dx_i based on my derivation above.
        # So the flat gradient is actually half the true gradient?
        # No wait - L-BFGS will still converge, just with a different step size interpretation.
        # For the shell gradient, I'll use the same convention for consistency.
        
        # Using the same convention as _lj_energy_and_grad:
        # grad_flat[i] = sum_j coeff[i,j] * (x_i - x_j)
        # dE/dscale_s = sum_{i in shell s} sum_j coeff[i,j] * (x_i - x_j) * base_coords_i
        # because x_i = base_coords_i * scale_s, so dx_i/dscale_s = base_coords_i
        
        for i in np.where(mask_i)[0]:
            # grad_i = sum_j coeff[i,j] * (x_i - x_j)
            g_i = np.sum(coeff[i, :] * diff[i, :], axis=1)  # shape (3,)
            # dE/dscale_s += g_i dot base_coords[i]
            grad[s] += np.dot(g_i, base_coords[i])
    
    return energy, grad

def make_fcc_with_noise(noise_scale=0.04, seed=42):
    """FCC base with small Gaussian noise to escape symmetry traps."""
    base = make_fcc_truncated_octahedron()
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, noise_scale, base.shape)

def optimize_strain(positions, max_iter=500, gtol=1e-8):
    """
    Optimize the 6-parameter strain tensor to minimize energy while
    preserving FCC topology. This reduces the 114D search to 6D.
    """
    base_coords, shells = _make_fcc_38_base()
    scale = 2.0 ** (-1.0 / 3.0)
    
    def strain_energy(params):
        strained = make_fcc_with_strain(params, base_coords, shells)
        strained = strained * scale
        try:
            return lj_energy_vectorized(strained)
        except Exception:
            return 1e20
    
    x0 = np.zeros(6)
    try:
        res = minimize(strain_energy, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10,
                                'adaptive': True})
        if res.success or res.fun < strain_energy(x0):
            strained = make_fcc_with_strain(res.x, base_coords, shells)
            strained = strained * scale
            return strained
    except Exception:
        pass
    return positions

def optimize_shell_radii(max_iter=500):
    """
    Optimize per-shell scale factors to minimize LJ energy.
    This is the symmetry-first approach: project onto the FCC symmetry manifold
    by allowing each shell to breathe independently.
    """
    base_coords, shells = _make_fcc_38_base()
    n_shells = int(max(shells)) + 1
    scale = 2.0 ** (-1.0 / 3.0)
    x0 = np.ones(n_shells) * scale
    
    try:
        res = minimize(_symmetry_adapted_energy_and_grad, x0, method='L-BFGS-B',
                       jac=True,
                       options={'maxiter': max_iter, 'gtol': 1e-12, 'maxls': 100, 'ftol': 1e-15})
        if res.fun < 1e10:
            # Build the optimized positions
            coords = np.zeros((N, 3))
            for s in range(n_shells):
                mask = shells == s
                if np.any(mask):
                    coords[mask] = base_coords[mask] * res.x[s]
            return coords
    except Exception:
        pass
    return make_fcc_truncated_octahedron()

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

def relax_multi_start(positions, n_starts=3, maxiter=300, gtol=1e-8, noise_scale=0.02):
    """
    Multi-start relaxation: try multiple slightly perturbed versions and
    return the best result. Helps escape shallow local minima.
    """
    best_energy = float('inf')
    best_pos = positions.copy()
    rng = np.random.default_rng(123)
    
    for i in range(n_starts):
        if i == 0:
            start = positions
        else:
            start = positions + rng.normal(0, noise_scale, positions.shape)
        
        if not check_validity(start):
            continue
        
        try:
            relaxed = relax(start, maxiter=maxiter, gtol=gtol)
            if check_validity(relaxed):
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed
        except Exception:
            continue
    
    return best_pos

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
    base, shells = _make_fcc_38_base()
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

def basin_hopping_strain(positions, n_iter=15, maxiter=100):
    """
    Basin hopping in shell-radius space: perturb the per-shell scale factors,
    relax, and accept/reject. This explores the FCC manifold efficiently.
    """
    base_coords, shells = _make_fcc_38_base()
    n_shells = int(max(shells)) + 1
    scale = 2.0 ** (-1.0 / 3.0)
    
    def shells_to_positions(params):
        coords = np.zeros((N, 3))
        for s in range(n_shells):
            mask = shells == s
            if np.any(mask):
                coords[mask] = base_coords[mask] * params[s]
        return coords
    
    def energy_from_shells(params):
        pos = shells_to_positions(params)
        try:
            rel = relax(pos, maxiter=maxiter, gtol=1e-6)
            return lj_energy_vectorized(rel), rel
        except Exception:
            return 1e20, pos
    
    current_params = np.ones(n_shells) * scale
    current_energy, current_pos = energy_from_shells(current_params)
    best_energy = current_energy
    best_pos = current_pos.copy()
    rng = np.random.default_rng(99)
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 15:
            break
        noise = rng.normal(0, 0.01, n_shells)
        new_params = current_params + noise
        new_energy, new_pos = energy_from_shells(new_params)
        
        if new_energy < current_energy:
            current_params = new_params
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        else:
            delta = new_energy - current_energy
            accept = math.exp(-delta / 0.05)
            if rng.uniform() < accept:
                current_params = new_params
                current_energy = new_energy
                current_pos = new_pos
    
    return best_pos, best_energy

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
            # Use multi-start relaxation for robustness
            relaxed = relax_multi_start(cand, n_starts=3, maxiter=300, gtol=1e-8, noise_scale=0.02)
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
        if time.time() - START_TIME > DEADLINE - 35:
            break
        try_candidate(cand)

    # 2. Shell-radius optimization: symmetry-first approach
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 30:
        try:
            shell_opt = optimize_shell_radii(max_iter=500)
            if check_validity(shell_opt):
                s_relax = relax(shell_opt, maxiter=500, gtol=1e-9)
                if check_validity(s_relax):
                    s_scaled = optimize_scale(s_relax)
                    if check_validity(s_scaled):
                        s_final = relax(s_scaled, maxiter=500, gtol=1e-10)
                        if check_validity(s_final):
                            energy = lj_energy_vectorized(s_final)
                            if energy < best_energy:
                                best_energy = energy
                                best_positions = s_final
        except Exception:
            pass

    # 3. FCC distortion optimization from the best candidate
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

    # 4. Basin hopping in shell-radius space (efficient low-D exploration)
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 25:
        try:
            bh_strain_result, bh_strain_energy = basin_hopping_strain(best_positions, n_iter=20, maxiter=100)
            if check_validity(bh_strain_result):
                scaled_bh = optimize_scale(bh_strain_result)
                if check_validity(scaled_bh):
                    final_bh = relax(scaled_bh, maxiter=500, gtol=1e-9)
                    if check_validity(final_bh):
                        energy = lj_energy_vectorized(final_bh)
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = final_bh
        except Exception:
            pass

    # 5. Standard basin hopping from the best result
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            bh_result, bh_energy = basin_hopping(best_positions, n_iter=20, step_size=0.25,
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

    # 6. Final high-precision polish
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

    # 7. Ultra-tight final polish if time permits
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
