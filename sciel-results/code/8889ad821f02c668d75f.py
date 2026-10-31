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
    FCC lattice points: (i, j, k) where i+j+k is even.
    We generate points in a large box, sort by distance from origin, and take the first N.
    Results are cached to avoid redundant computation.
    """
    if n_atoms in _FCC_CACHE:
        return _FCC_CACHE[n_atoms]
    
    # Scale factor to make nearest neighbor distance 1.0 in this intermediate representation
    # Nearest neighbor in FCC is at (1,1,0) type vectors, distance sqrt(2).
    # So we divide by sqrt(2) to make NN distance 1.
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
    
    # Sort by distance from origin
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    
    # Assign shells based on radial distance
    radii = np.linalg.norm(coords, axis=1)
    # Cluster radii into shells. 
    # For FCC 38, there are typically 3 or 4 distinct shells.
    # We use a simple rounding to identify unique shells.
    unique_r = np.sort(np.unique(np.round(radii, 4)))
    
    shell_ids = np.zeros(n_atoms, dtype=int)
    for i, r in enumerate(radii):
        for sid, ur in enumerate(unique_r):
            if abs(r - ur) < 0.05:
                shell_ids[i] = sid
                break
    
    _FCC_CACHE[n_atoms] = (coords, shell_ids, unique_r)
    return coords, shell_ids, unique_r

def _symmetry_adapted_energy_and_grad(params):
    """
    Compute energy and gradient for the symmetry-adapted parameterization.
    params: [global_scale, breath_0, breath_1, ...]
    Uses analytical gradient for speed.
    """
    n_atoms = N
    global_scale = params[0]
    shell_params = params[1:]
    
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(n_atoms)
    n_shells = len(unique_r)
    
    # Construct positions and per-atom scale factors
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
    
    # Compute Energy
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Analytical gradient
    # dE/dpos_i = sum_j -24/r^14 * (pos_i - pos_j) for i != j
    # Since pos_i = base_coords_i * scale_i:
    #   dE/dscale_i = sum_j dE/dpos_i . base_coords_i
    #   dE/dglobal_scale = sum_i dE/dscale_i * breath_i
    #   dE/dbreath_s = sum_{i in shell s} dE/dscale_i * global_scale
    
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    # grad_matrix[i,j] = coeff[i,j] * (pos_i - pos_j)
    # Note: diff[i,j] = pos_i - pos_j
    grad_matrix = coeff[:, :, np.newaxis] * diff
    dE_dpos = np.sum(grad_matrix, axis=1)  # shape (N, 3)
    
    # dE/dscale_i = dot(dE_dpos_i, base_coords_i)
    dE_dscale = np.sum(dE_dpos * base_coords, axis=1)  # shape (N,)
    
    grad = np.zeros_like(params)
    # Global scale gradient
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        grad[0] += dE_dscale[i] * breath
    
    # Shell breathing gradients
    for s in range(min(n_shells, len(shell_params))):
        mask = (shell_ids == s)
        grad[1 + s] = global_scale * np.sum(dE_dscale[mask])
    
    return energy, grad

def _make_icosahedral_structure():
    """
    Generate an icosahedral structure for 38 atoms.
    Icosahedron: 1 center + 12 vertices + 12 edge midpoints + 12 face centers = 37
    We need 38, so we add one more atom.
    
    For LJ38, the icosahedral structure is a strong local minimum.
    """
    # Icosahedron vertices (12 atoms)
    phi = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Scale so nearest neighbor distance is 1.0
    # Nearest neighbors in icosahedron are at edge length
    # Edge length = 2 (in the unnormalized coordinates above, before scaling)
    # After normalization to unit sphere, edge length = 2/sqrt(1+phi^2)
    # We want NN distance = 1, so scale by 1/edge_length
    # But let's just compute it numerically
    min_dist = float('inf')
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if d < min_dist:
                min_dist = d
    
    scale = 1.0 / min_dist
    vertices = vertices * scale
    
    # Center atom
    center = np.array([0.0, 0.0, 0.0])
    
    # Edge midpoints (12 atoms) - these are at the midpoints of the 30 edges
    # Actually, for icosahedron, the 12 edge midpoints form a dodecahedron
    # Let's generate them properly
    # Edges of icosahedron: each vertex connects to 5 others
    # We need to find all 30 edges
    
    # For simplicity, let's place 12 atoms at face centers
    # Face centers of icosahedron form a dodecahedron
    # Dodecahedron vertices (20 atoms), but we only need 12
    
    # Alternative: just use a second icosahedron rotated by 36 degrees
    # This gives a Mackay icosahedron structure
    
    # For now, let's create a simpler structure:
    # 1 center + 12 inner shell (icosahedron) + 25 outer shell
    # The outer 25 can be placed on a truncated icosahedron or similar
    
    # Actually, the standard icosahedral 38-atom cluster is:
    # 1 + 12 + 25 where the 25 are on the faces and edges
    
    # Let me use a different approach: place atoms at all positions in a
    # two-shell icosahedral structure
    
    # Inner shell: 12 atoms at icosahedron vertices
    inner = vertices.copy()
    
    # Outer shell: 25 atoms
    # 12 at face centers (dodecahedron vertices)
    # 12 at edge midpoints  
    # 1 at... wait, that's 25
    
    # Face centers of icosahedron = vertices of dual dodecahedron
    # Dodecahedron has 20 vertices, but icosahedron has 20 faces
    # So 20 face centers, but we only need 12 of them
    
    # This is getting complicated. Let me just use a numerical approach:
    # Generate 38 points on a sphere with icosahedral symmetry as best as possible
    
    # Simple approach: 1 center + 12 at r1 + 25 at r2
    # with proper angular distribution
    
    # For the 25 outer atoms, use a truncated icosahedron (soccer ball) which has 60 vertices
    # But we only need 25, so take a subset
    
    # Actually, let me just return the icosahedron + center and let basin hopping
    # figure out the rest. The key is to provide a good starting point in the
    # icosahedral basin.
    
    positions = np.zeros((38, 3))
    positions[0] = center
    positions[1:13] = inner
    
    # For the remaining 25, place them at larger radius with good angular distribution
    # Use a Fibonacci sphere for the 25 outer atoms
    n_outer = 25
    r_outer = 1.8  # approximate outer shell radius
    
    for i in range(n_outer):
        # Fibonacci sphere
        z = 1.0 - 2.0 * (i + 0.5) / n_outer
        r = math.sqrt(1.0 - z*z)
        theta = math.pi * (1.0 + math.sqrt(5.0)) * i  # golden angle
        
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        # Scale to outer radius
        positions[13 + i] = np.array([x, y, z]) * r_outer
    
    return positions

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

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Symmetry-Adapted Optimization (FCC)
    # This is the primary strategy. It finds the optimal FCC structure with shell breathing.
    try:
        base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
        n_shells = len(unique_r)
        
        # Initial guess: 
        # The LJ equilibrium distance is 2^(1/6). 
        # Our base_coords have NN distance 1.0.
        # So global_scale should be approximately 2^(1/6).
        initial_scale = 2.0 ** (1.0/6.0)
        x0 = np.array([initial_scale] + [0.0] * n_shells)
        
        # Optimize the low-dimensional parameters
        res = minimize(_symmetry_adapted_energy_and_grad, x0, method='L-BFGS-B',
                       jac=True,
                       options={'maxiter': 1000, 'gtol': 1e-10, 'maxls': 100, 'ftol': 1e-14})
        
        if res.success:
            # Map back to Cartesian coordinates
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
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
    except Exception:
        pass

    # 2. Local Relaxation Polish (high-precision)
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 25:
        try:
            polished = relax(best_positions, maxiter=2000, gtol=1e-12)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass

    # 3. Basin Hopping from FCC
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 40:
        try:
            x0 = best_positions.flatten()
            current_energy, _ = _lj_energy_and_grad(x0)
            best_bh_energy = current_energy
            best_bh_x = x0.copy()
            
            rng = np.random.default_rng(42)
            step_size = 0.15
            n_iter = 120
            
            for i in range(n_iter):
                if time.time() - START_TIME > DEADLINE - 20:
                    break
                
                noise = rng.normal(0, step_size, x0.shape)
                perturbed = x0 + noise
                
                try:
                    result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B', 
                                    jac=True,
                                    options={'maxiter': 300, 'gtol': 1e-8, 'maxls': 80})
                    new_x = result.x
                    new_energy, _ = _lj_energy_and_grad(new_x)
                    
                    if new_energy < current_energy:
                        x0 = new_x
                        current_energy = new_energy
                        if new_energy < best_bh_energy:
                            best_bh_energy = new_energy
                            best_bh_x = new_x.copy()
                    else:
                        delta = new_energy - current_energy
                        if delta > 0:
                            accept = math.exp(-delta / 0.05)
                            if rng.uniform() < accept:
                                x0 = new_x
                                current_energy = new_energy
                except Exception:
                    continue
            
            if best_bh_energy < best_energy:
                best_energy = best_bh_energy
                best_positions = best_bh_x.reshape(N, 3)
        except Exception:
            pass

    # 4. Basin Hopping from Icosahedral Structure
    # The icosahedral funnel is the main competitor. Starting from there might
    # find a lower energy if the FCC structure is not the true global minimum.
    if time.time() - START_TIME < DEADLINE - 40:
        try:
            ico_positions = _make_icosahedral_structure()
            ico_positions = relax(ico_positions, maxiter=500, gtol=1e-8)
            
            if check_validity(ico_positions):
                ico_energy = lj_energy_vectorized(ico_positions)
                
                x0 = ico_positions.flatten()
                current_energy, _ = _lj_energy_and_grad(x0)
                best_bh_energy = current_energy
                best_bh_x = x0.copy()
                
                rng = np.random.default_rng(123)
                step_size = 0.15
                n_iter = 100
                
                for i in range(n_iter):
                    if time.time() - START_TIME > DEADLINE - 15:
                        break
                    
                    noise = rng.normal(0, step_size, x0.shape)
                    perturbed = x0 + noise
                    
                    try:
                        result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B', 
                                        jac=True,
                                        options={'maxiter': 300, 'gtol': 1e-8, 'maxls': 80})
                        new_x = result.x
                        new_energy, _ = _lj_energy_and_grad(new_x)
                        
                        if new_energy < current_energy:
                            x0 = new_x
                            current_energy = new_energy
                            if new_energy < best_bh_energy:
                                best_bh_energy = new_energy
                                best_bh_x = new_x.copy()
                        else:
                            delta = new_energy - current_energy
                            if delta > 0:
                                accept = math.exp(-delta / 0.05)
                                if rng.uniform() < accept:
                                    x0 = new_x
                                    current_energy = new_energy
                    except Exception:
                        continue
                
                if best_bh_energy < best_energy:
                    best_energy = best_bh_energy
                    best_positions = best_bh_x.reshape(N, 3)
        except Exception:
            pass

    # 5. Final high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            polished = relax(best_positions, maxiter=5000, gtol=1e-14)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
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
    # Quick self-test
    t0 = time.time()
    pos = construct()
    t1 = time.time()
    e = lj_energy(pos)
    print(f"Energy: {e}, Time: {t1-t0:.2f}s")
    print(f"Valid: {check_validity(pos)}")
