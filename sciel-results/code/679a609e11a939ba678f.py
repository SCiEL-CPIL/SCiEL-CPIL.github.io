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
            r2 = np.dot(rij, rij)
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
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero on diagonal
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def lj_energy_and_grad(positions):
    """Compute LJ energy and gradient simultaneously."""
    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e20, grad
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            pair_energy = 4.0 * (inv_r12 - inv_r6)
            energy += pair_energy
            # dE/dr2 = 4*(-12*inv_r12 + 6*inv_r6) / 2 = 2*(-12*inv_r12 + 6*inv_r6)
            dE_dr2 = 2.0 * (-12.0 * inv_r12 + 6.0 * inv_r6)
            # dE/dpos_i = -dE_dr2 * 2 * rij (since r2 = |rij|^2, dr2/dpos_i = -2*rij)
            # Actually: r2 = sum_k (pos_j_k - pos_i_k)^2
            # dr2/dpos_i_k = -2*(pos_j_k - pos_i_k) = -2*rij_k
            # dE/dpos_i_k = dE_dr2 * dr2/dpos_i_k = -2*dE_dr2*rij_k
            # dE/dpos_j_k = +2*dE_dr2*rij_k
            factor = 2.0 * dE_dr2
            grad[i] -= factor * rij
            grad[j] += factor * rij
    return energy, grad

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r = np.sqrt(np.dot(rij, rij))
            if r < min_dist:
                return False
    return True

def make_fcc_truncated_octahedron():
    """Generate the FCC truncated octahedron cluster for LJ38.
    
    The known global minimum is a truncated octahedron from the FCC lattice.
    For 38 atoms: 1 center + 12 first shell + 25 second shell (truncated).
    
    FCC lattice vectors with nearest-neighbor distance = 1:
    a = 1/sqrt(2) so that |a1-a2| = 1
    """
    a = 1.0 / math.sqrt(2.0)
    
    # Generate FCC lattice points: (i,j,k) where i+j+k is even, scaled by a
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * a, j * a, k * a
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[3])
    
    # Select the first 38 points
    selected = [p[:3] for p in points[:N]]
    return np.array(selected, dtype=float)

def make_fcc_with_strain(base_positions, strain_params):
    """Apply symmetry-adapted strain to FCC base positions.
    
    strain_params: [scale, exx, eyy, ezz, exy, exz, eyz]
    where exx, eyy, ezz are diagonal strain components and
    exy, exz, eyz are shear components.
    """
    scale, exx, eyy, ezz, exy, exz, eyz = strain_params
    # Strain tensor (symmetric)
    S = np.array([
        [1.0 + exx, exy, exz],
        [exy, 1.0 + eyy, eyz],
        [exz, eyz, 1.0 + ezz]
    ])
    # Apply strain and scale
    strained = base_positions @ S.T * scale
    return strained

def optimize_strain(base_positions):
    """Optimize strain parameters for the FCC cluster."""
    def strain_energy(params):
        strained = make_fcc_with_strain(base_positions, params)
        if not check_validity(strained, min_dist=0.4):
            return 1e10
        return lj_energy(strained)
    
    # Initial guess: uniform scale only
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    try:
        result = minimize(strain_energy, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < 1e9:
            return make_fcc_with_strain(base_positions, result.x)
    except Exception:
        pass
    
    return base_positions

def make_fcc_with_noise(base_positions, seed=0, noise_scale=0.05):
    """Add random noise to break symmetry."""
    rng = np.random.default_rng(seed)
    noisy = base_positions + rng.normal(0, noise_scale, base_positions.shape)
    return noisy

def _make_icosahedral():
    """Generate a nested icosahedral cluster."""
    phi = (1 + math.sqrt(5)) / 2.0
    r_ico = 1.0 / math.sqrt(1 + phi * phi)
    
    # 12 vertices of icosahedron
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * r_ico
    
    points = [np.array([0.0, 0.0, 0.0])]  # Center
    points.extend([v * 1.12 for v in ico_verts])  # First shell
    
    # Need 25 more atoms
    rng = np.random.default_rng(42)
    for _ in range(N - len(points)):
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1)
        theta = 2 * math.pi * u
        z = 2 * v - 1
        r = math.sqrt(1 - z*z)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append(np.array([x, y, z]) * 1.6)
        
    return np.array(points, dtype=float)

def _make_initial_clusters():
    """Generate initial candidate clusters."""
    candidates = []
    
    # 1. FCC truncated octahedron (global min candidate)
    try:
        fcc = make_fcc_truncated_octahedron()
        if check_validity(fcc):
            candidates.append(('fcc', fcc))
    except Exception:
        pass
    
    # 2. Strained FCC
    try:
        fcc_base = make_fcc_truncated_octahedron()
        fcc_strained = optimize_strain(fcc_base)
        if check_validity(fcc_strained):
            candidates.append(('fcc_strained', fcc_strained))
    except Exception:
        pass
    
    # 3. Perturbed FCC (multiple seeds)
    for seed in [0, 1, 2, 3, 4]:
        try:
            fcc_base = make_fcc_truncated_octahedron()
            fcc_pert = make_fcc_with_noise(fcc_base, seed=seed, noise_scale=0.03)
            if check_validity(fcc_pert):
                candidates.append((f'fcc_pert_{seed}', fcc_pert))
        except Exception:
            pass
    
    # 4. Icosahedral cluster
    try:
        ico = _make_icosahedral()
        if check_validity(ico):
            candidates.append(('ico', ico))
    except Exception:
        pass
    
    return candidates

def relax(positions, maxiter=300, gtol=1e-8):
    """Relax using L-BFGS-B with gradient."""
    def energy_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy(pos)
    
    def energy_grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        e, g = lj_energy_and_grad(pos)
        return e, g.flatten()
    
    x0 = positions.flatten()
    
    try:
        result = minimize(energy_grad_func, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions):
    """High-precision polish pass with tight tolerances."""
    if time.time() - START_TIME > DEADLINE - 3:
        return positions
    return relax(positions, maxiter=500, gtol=1e-10)

def basin_hop(positions, n_hops=20, hop_scale=0.2, rng=None):
    """Basin hopping: perturb, relax, accept if lower energy."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy = lj_energy(positions)
    current_pos = positions.copy()
    
    for i in range(n_hops):
        if time.time() - START_TIME > DEADLINE - 5:
            break
        
        # Random perturbation
        hop = rng.normal(0, hop_scale, positions.shape)
        perturbed = current_pos + hop
        
        if not check_validity(perturbed, min_dist=0.4):
            continue
        
        # Relax
        relaxed = relax(perturbed, maxiter=100, gtol=1e-6)
        
        if not check_validity(relaxed, min_dist=0.5):
            continue
        
        new_energy = lj_energy(relaxed)
        
        # Accept if better (or with probability for Metropolis-like acceptance)
        if new_energy < current_energy:
            current_pos = relaxed
            current_energy = new_energy
        elif new_energy < current_energy + 0.5:
            # Occasionally accept worse configurations to escape local minima
            if rng.random() < 0.3:
                current_pos = relaxed
                current_energy = new_energy
    
    return current_pos, current_energy

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # Phase 1: Generate and relax initial candidates
    candidates = _make_initial_clusters()
    
    for name, cand in candidates:
        if time.time() - START_TIME > DEADLINE - 10:
            break
            
        try:
            relaxed = relax(cand, maxiter=200, gtol=1e-7)
            if not check_validity(relaxed):
                continue
                
            energy = lj_energy_vectorized(relaxed)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback
        try:
            best_positions = make_fcc_truncated_octahedron()
            if not check_validity(best_positions):
                side = int(np.ceil(N ** (1/3)))
                pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
                pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
                best_positions = np.array(pts[:N], dtype=float) * 1.2
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]
    
    # Phase 2: Basin hopping from best configuration
    if time.time() - START_TIME < DEADLINE - 15:
        try:
            rng = np.random.default_rng(42)
            hopped, hop_energy = basin_hop(best_positions, n_hops=30, hop_scale=0.15, rng=rng)
            if check_validity(hopped) and hop_energy < best_energy:
                best_energy = hop_energy
                best_positions = hopped
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() - START_TIME < DEADLINE - 3:
        try:
            polished = polish(best_positions)
            if check_validity(polished):
                polished_energy = lj_energy_vectorized(polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_positions = polished
        except Exception:
            pass
    
    return best_positions
