import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0  # Safety margin for 150s limit
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
            r2 = np.dot(rij, rij)
            if r2 < min_dist * min_dist:
                return False
    return True

_FCC_CACHE = {}

def _get_fcc_shell_structure(n_atoms=38):
    """
    Generate the 38 closest atoms in an FCC lattice.
    Returns base coordinates (unscaled), shell IDs, and unique radii.
    """
    if n_atoms in _FCC_CACHE:
        return _FCC_CACHE[n_atoms]
    
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 10
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    
    radii = np.linalg.norm(coords, axis=1)
    unique_r = np.sort(np.unique(np.round(radii, 4)))
    
    shell_ids = np.zeros(n_atoms, dtype=int)
    for i, r in enumerate(radii):
        for sid, ur in enumerate(unique_r):
            if abs(r - ur) < 0.05:
                shell_ids[i] = sid
                break
    
    _FCC_CACHE[n_atoms] = (coords, shell_ids, unique_r)
    return coords, shell_ids, unique_r

def _make_truncated_octahedron():
    """
    Build an explicit FCC truncated octahedron for 38 atoms.
    FCC truncated octahedron: 1 + 12 + 14 + 11 = 38
    Shell 1 (12): face centers of octahedron - FCC positions at (±1,±1,0) etc., scaled
    Shell 2 (14): edge centers - FCC positions at (±2,0,0) type, plus face centers
    Shell 3 (11): remaining
    
    Actually, the standard FCC truncated octahedron for 38 atoms has:
    - 1 center
    - 12 atoms at distance sqrt(2) (face centers of the octahedron)
    - 14 atoms at distance 2 (edge midpoints and some face centers)
    - 11 atoms at distance sqrt(6) or similar
    
    Let me use the actual FCC lattice points sorted by radius.
    """
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    # Scale to LJ equilibrium: nearest neighbor should be 2^(1/6)
    # In our base coords, NN distance is 1.0 (after dividing by sqrt(2))
    # Wait, let's check: FCC points at (1,1,0) have distance sqrt(2) from origin
    # After scaling by 1/sqrt(2), they're at distance 1.
    # So NN distance in base_coords is 1.0.
    # We want NN distance = 2^(1/6), so global scale = 2^(1/6)
    scale = 2.0 ** (1.0/6.0)
    return base_coords * scale

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient for full Cartesian coordinates."""
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

def _lj_energy_only(pos_flat):
    """Compute LJ energy only (no gradient) - faster for basin hopping."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def relax(positions, maxiter=500, gtol=1e-8):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def _basin_hop(x0, current_energy, rng, step_size, n_iter, deadline_margin=20, temp_start=0.15, temp_end=0.02):
    """
    Basin hopping with adaptive step size and annealing Metropolis acceptance.
    Returns (best_energy, best_x).
    """
    best_bh_energy = current_energy
    best_bh_x = x0.copy()
    
    # Adaptive step size
    base_step = step_size
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - deadline_margin:
            break
        
        # Annealing temperature
        progress = i / max(n_iter - 1, 1)
        temp = temp_start + (temp_end - temp_start) * progress
        
        # Adaptive step size: occasionally increase or decrease
        if i % 8 == 4:
            if rng.uniform() < 0.35:
                step_size = base_step * rng.uniform(1.5, 3.0)
            elif rng.uniform() < 0.3:
                step_size = base_step * rng.uniform(0.3, 0.7)
            else:
                step_size = base_step
        
        noise = rng.normal(0, step_size, x0.shape)
        perturbed = x0 + noise
        
        try:
            result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B', 
                            jac=True,
                            options={'maxiter': 150, 'gtol': 1e-6, 'maxls': 50})
            new_x = result.x
            new_energy = _lj_energy_only(new_x)
            
            if new_energy < current_energy:
                x0 = new_x
                current_energy = new_energy
                if new_energy < best_bh_energy:
                    best_bh_energy = new_energy
                    best_bh_x = new_x.copy()
            else:
                delta = new_energy - current_energy
                if delta > 0:
                    accept = math.exp(-delta / max(temp, 1e-10))
                    if rng.uniform() < accept:
                        x0 = new_x
                        current_energy = new_energy
        except Exception:
            continue
    
    return best_bh_energy, best_bh_x

def _make_icosahedral_structure():
    """
    Generate an icosahedral structure for 38 atoms.
    1 center + 12 inner shell + 25 outer shell
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    min_dist = float('inf')
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if d < min_dist:
                min_dist = d
    
    scale = 1.0 / min_dist
    vertices = vertices * scale
    
    positions = np.zeros((38, 3))
    positions[0] = [0.0, 0.0, 0.0]
    positions[1:13] = vertices
    
    # Outer 25 atoms: place on a larger sphere with good angular distribution
    n_outer = 25
    r_outer = 1.8
    
    for i in range(n_outer):
        z = 1.0 - 2.0 * (i + 0.5) / n_outer
        r = math.sqrt(1.0 - z*z)
        theta = math.pi * (1.0 + math.sqrt(5.0)) * i
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        positions[13 + i] = np.array([x, y, z]) * r_outer
    
    return positions

def _symmetry_adapted_energy_and_grad(params):
    """
    Compute energy and gradient for the symmetry-adapted parameterization.
    params: [global_scale, breath_0, breath_1, ...]
    """
    n_atoms = N
    global_scale = params[0]
    shell_params = params[1:]
    
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(n_atoms)
    n_shells = len(unique_r)
    
    positions = np.zeros((n_atoms, 3))
    atom_scales = np.zeros(n_atoms)
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        atom_scales[i] = breath * global_scale
        positions[i] = base_coords[i] * atom_scales[i]
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
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
    dE_dpos = np.sum(grad_matrix, axis=1)
    
    dE_dscale = np.sum(dE_dpos * base_coords, axis=1)
    
    grad = np.zeros_like(params)
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        grad[0] += dE_dscale[i] * breath
    
    for s in range(min(n_shells, len(shell_params))):
        mask = (shell_ids == s)
        grad[1 + s] = global_scale * np.sum(dE_dscale[mask])
    
    return energy, grad

def _optimize_fcc_with_shell_breathing():
    """
    Optimize FCC structure with per-shell breathing parameters.
    Returns (positions, energy) or (None, inf) on failure.
    """
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    n_shells = len(unique_r)
    
    initial_scale = 2.0 ** (1.0/6.0)
    x0 = np.array([initial_scale] + [0.0] * n_shells)
    
    try:
        res = minimize(_symmetry_adapted_energy_and_grad, x0, method='L-BFGS-B',
                       jac=True,
                       options={'maxiter': 2000, 'gtol': 1e-12, 'maxls': 100, 'ftol': 1e-16})
        
        global_scale = res.x[0]
        shell_params = res.x[1:]
        
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
            return positions, energy
    except Exception:
        pass
    
    return None, float('inf')

def _high_precision_polish(positions):
    """
    Final high-precision local optimization with very tight tolerances.
    """
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': 10000, 'gtol': 1e-14, 'maxls': 200, 'ftol': 1e-18})
        polished = result.x.reshape(N, 3)
        if check_validity(polished):
            energy = lj_energy_vectorized(polished)
            return polished, energy
    except Exception:
        pass
    
    return positions, lj_energy_vectorized(positions)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # Phase 1: Symmetry-adapted FCC optimization with shell breathing
    try:
        fcc_pos, fcc_energy = _optimize_fcc_with_shell_breathing()
        if fcc_pos is not None:
            best_energy = fcc_energy
            best_positions = fcc_pos
    except Exception:
        pass
    
    # Phase 2: High-precision polish of the FCC structure
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 30:
        try:
            polished, pol_energy = _high_precision_polish(best_positions)
            if pol_energy < best_energy:
                best_energy = pol_energy
                best_positions = polished
        except Exception:
            pass
    
    # Phase 3: Basin hopping from FCC - multiple runs with different seeds and step sizes
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 50:
        x0 = best_positions.flatten()
        current_energy = _lj_energy_only(x0)
        
        for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
            if time.time() - START_TIME > DEADLINE - 25:
                break
            
            rng = np.random.default_rng(seed)
            step_size = 0.15 + 0.05 * (seed % 4)
            
            bh_energy, bh_x = _basin_hop(x0, current_energy, rng, step_size, 60, deadline_margin=20, 
                                         temp_start=0.20, temp_end=0.03)
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_x.reshape(N, 3)
                # Update x0 for next iteration
                x0 = bh_x
                current_energy = bh_energy
    
    # Phase 4: Basin hopping from icosahedral structure
    if time.time() - START_TIME < DEADLINE - 40:
        try:
            ico_positions = _make_icosahedral_structure()
            ico_positions = relax(ico_positions, maxiter=300, gtol=1e-7)
            
            if check_validity(ico_positions):
                ico_energy = lj_energy_vectorized(ico_positions)
                
                if ico_energy < best_energy:
                    best_energy = ico_energy
                    best_positions = ico_positions
                
                x0 = ico_positions.flatten()
                current_energy = _lj_energy_only(x0)
                
                for seed in [11, 22, 33, 44, 55]:
                    if time.time() - START_TIME > DEADLINE - 20:
                        break
                    
                    rng = np.random.default_rng(seed)
                    step_size = 0.18 + 0.04 * (seed % 3)
                    
                    bh_energy, bh_x = _basin_hop(x0, current_energy, rng, step_size, 50, deadline_margin=18,
                                                 temp_start=0.25, temp_end=0.04)
                    
                    if bh_energy < best_energy:
                        best_energy = bh_energy
                        best_positions = bh_x.reshape(N, 3)
                        x0 = bh_x
                        current_energy = bh_energy
        except Exception:
            pass
    
    # Phase 5: More basin hopping from the current best with larger perturbations
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            x0 = best_positions.flatten()
            current_energy = _lj_energy_only(x0)
            
            for seed in [999, 888, 777]:
                if time.time() - START_TIME > DEADLINE - 15:
                    break
                
                rng = np.random.default_rng(seed)
                # Larger step size to escape local minima
                bh_energy, bh_x = _basin_hop(x0, current_energy, rng, 0.30, 40, deadline_margin=12,
                                             temp_start=0.30, temp_end=0.05)
                
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_x.reshape(N, 3)
                    x0 = bh_x
                    current_energy = bh_energy
        except Exception:
            pass
    
    # Phase 6: Final ultra-high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 8:
        try:
            polished, pol_energy = _high_precision_polish(best_positions)
            if pol_energy < best_energy:
                best_energy = pol_energy
                best_positions = polished
        except Exception:
            pass
    
    # Fallback
    if best_positions is None:
        try:
            base_coords, _, _ = _get_fcc_shell_structure(N)
            positions = base_coords * (2.0 ** (1.0/6.0))
            if check_validity(positions):
                best_positions = positions
            else:
                best_positions = np.zeros((N, 3))
                for i in range(N):
                    best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]

    return best_positions

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    t1 = time.time()
    e = lj_energy(pos)
    print(f"Energy: {e}, Time: {t1-t0:.2f}s")
    print(f"Valid: {check_validity(pos)}")
