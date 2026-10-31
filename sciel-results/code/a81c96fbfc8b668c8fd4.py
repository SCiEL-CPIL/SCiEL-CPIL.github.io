"""LJ38 cluster optimisation: multi-start with icosahedral/FCC seeds + basin hopping."""
import numpy as np
import time
import sys

N = 38
DEADLINE = 115.0  # safety margin before 120s limit
START_TIME = time.time()


def lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    r6 = r2**3
    r12 = r6**2
    # energy: 4 * sum(r^-12 - r^-6) over pairs (i<j)
    e_matrix = 4.0 * (1.0/r12 - 1.0/r6)
    np.fill_diagonal(e_matrix, 0.0)
    energy = 0.5 * np.sum(e_matrix)
    # gradient: dE/dr_i = 4*(-12*r^-14 + 6*r^-8) * (r_i - r_j)
    # = 4*(-12/r^14 + 6/r^8) * (pos_i - pos_j)
    # In matrix form: grad_i = 4 * sum_j (-12/r_ij^14 + 6/r_ij^8) * (pos_i - pos_j)
    # Note: 1/r^14 = 1/(r^2)^7 = 1/r2^7
    # 1/r^8 = 1/(r^2)^4 = 1/r2^4
    r2_safe = np.where(r2 > 1e-12, r2, 1.0)
    coeff = 4.0 * (-12.0 / (r2_safe**7) + 6.0 / (r2_safe**4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def lj_energy(pos):
    """Compute LJ energy only."""
    e, _ = lj_energy_and_grad(pos)
    return e


def check_valid(pos):
    """Check if all pairwise distances >= 0.5."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, 1.0)
    min_r = np.sqrt(np.min(r2))
    return min_r >= 0.5


def icosahedral_seed():
    """Generate an icosahedral-like configuration for 38 atoms.
    
    Icosahedron has 12 vertices, 30 edges, 20 faces.
    For N=38: 1 center + 12 vertices + 12 mid-edges + 13 additional
    Actually, let's use a known good icosahedral structure:
    - 12 vertices of icosahedron
    - 1 center
    - 12 face centers (or edge midpoints)
    - Remaining atoms placed in second shell
    
    A common icosahedral arrangement for 38:
    Shell 1: 12 vertices (radius ~1.12)
    Shell 2: 12 more (face centers, radius ~1.12*sqrt(3/4) or similar)
    Shell 3: 1 center + 13 more
    
    Let me use a simpler approach: generate icosahedron vertices, then add atoms
    at face centers and adjust.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    # Scale to LJ minimum distance
    scale = 1.12
    verts *= scale
    
    # Face centers of icosahedron (20 faces)
    # Each face is defined by 3 vertices. We need to identify faces.
    # For now, let's just use a simpler multi-shell approach.
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Second shell: 12 atoms at positions corresponding to edge midpoints or face centers
    # Edge midpoints of icosahedron: there are 30 edges, but we want 12 positions
    # Let's use the 12 face centers of the dual dodecahedron, which correspond to
    # the 12 vertices of the icosahedron rotated. Actually, let's just place 12 atoms
    # at a slightly larger radius with different angles.
    
    # Simpler: use two icosahedral shells
    # Shell 1: 12 atoms at radius r1
    # Shell 2: 12 atoms at radius r2, rotated
    # Shell 3: 1 center + 13 atoms
    
    # For 38 atoms: 1 + 12 + 12 + 13 = 38
    # Let's place:
    # - 1 center
    # - 12 at icosahedron vertices (radius ~1.12)
    # - 12 at a rotated icosahedron (radius ~1.12 * 1.5 or so)
    # - 13 more distributed
    
    # Actually, let me just create a reasonable icosahedral-like structure
    # and rely on relaxation to improve it.
    
    # Generate 38 points: 1 center, 12 inner shell, 12 middle shell, 13 outer shell
    rng = np.random.RandomState(42)
    
    points = [center[0]]
    points.extend(list(verts))
    
    # Second icosahedral shell, rotated
    # Rotate by golden angle around z-axis
    theta = np.arccos(1.0 / phi)  # ~63.4 degrees
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    verts2 = verts @ rot.T
    verts2 *= 1.6  # larger radius
    points.extend(list(verts2))
    
    # Third shell: 13 atoms
    # Place 13 atoms on a sphere of radius ~2.5
    # Use Fibonacci sphere for even distribution
    def fibonacci_sphere(n, radius):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            z = 1 - 2 * i / (n - 1) if n > 1 else 0
            r = np.sqrt(max(0, 1 - z**2))
            theta = golden_angle * i
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append(np.array([x, y, z]) * radius)
        return points
    
    shell3 = fibonacci_sphere(13, 2.5)
    points.extend(shell3)
    
    pos = np.array(points, dtype=float)
    return pos[:38]


def fcc_truncated_octahedron_seed():
    """Generate FCC truncated octahedron configuration for 38 atoms.
    
    The FCC truncated octahedron for N=38 is the putative global minimum.
    It consists of:
    - 1 center
    - 12 atoms in first shell (octahedral positions)
    - 14 atoms in second shell
    - 11 atoms in third shell
    Total: 1 + 12 + 14 + 11 = 38
    
    Actually, the standard FCC truncated octahedron for 38 atoms has:
    - 1 center
    - 12 nearest neighbors (octahedral)
    - 14 next-nearest
    - 11 more
    
    Let me construct it properly from FCC lattice.
    """
    # FCC lattice: atoms at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell
    # Truncated octahedron shape
    
    # Generate FCC points within a truncated octahedron
    # FCC basis vectors
    a = 1.0  # lattice constant
    # For LJ, optimal nearest neighbor distance is 2^(1/6) ≈ 1.1225
    # In FCC, nearest neighbor distance = a/sqrt(2)
    # So a = 1.1225 * sqrt(2) ≈ 1.5874
    
    a = 1.1225 * np.sqrt(2)
    
    # Generate FCC lattice points
    points = []
    range_val = 4
    for i in range(-range_val, range_val+1):
        for j in range(-range_val, range_val+1):
            for k in range(-range_val, range_val+1):
                # FCC basis: (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
                for dx, dy, dz in [(0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)]:
                    pt = np.array([i*a + dx, j*a + dy, k*a + dz])
                    points.append(pt)
    
    points = np.array(points)
    
    # Truncated octahedron: |x|+|y|+|z| <= some cutoff, and |x|,|y|,|z| <= some max
    # For FCC truncated octahedron, the shape is defined by:
    # |x| <= a, |y| <= a, |z| <= a, and |x|+|y|+|z| <= 2a (approximately)
    
    # Let's just take the closest 38 points to the origin
    distances = np.linalg.norm(points, axis=1)
    idx = np.argsort(distances)
    points = points[idx[:38]]
    
    # Center the configuration
    points -= np.mean(points, axis=0)
    
    return points


def random_seed(rng):
    """Generate a random valid configuration."""
    # Place atoms randomly in a sphere
    pos = rng.uniform(-3, 3, size=(N, 3))
    # Ensure minimum distance by iterative rejection
    for _ in range(1000):
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(pos[i] - pos[j]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        # Nudge overlapping atoms
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 0.5:
                    direction = pos[i] - pos[j]
                    if np.linalg.norm(direction) < 1e-10:
                        direction = rng.randn(3)
                    direction /= np.linalg.norm(direction)
                    pos[i] += 0.01 * direction
                    pos[j] -= 0.01 * direction
    return pos


def basin_hopping(pos, max_iters=50, rng=None):
    """Simple basin hopping: perturb, relax, accept/reject."""
    if rng is None:
        rng = np.random.RandomState()
    
    current_pos = pos.copy()
    current_energy = lj_energy(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    step_size = 0.3
    for i in range(max_iters):
        if time.time() - START_TIME > DEADLINE:
            break
        
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, size=(N, 3))
        
        # Relax using L-BFGS
        try:
            from scipy.optimize import minimize
            result = minimize(
                lambda x: lj_energy_and_grad(x)[0],
                perturbed,
                jac=lambda x: lj_energy_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            new_pos = result.x
            new_energy = result.fun
        except:
            continue
        
        if not check_valid(new_pos):
            continue
        
        # Metropolis accept/reject
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
        else:
            delta_e = new_energy - current_energy
            if delta_e < 0:
                current_pos = new_pos
                current_energy = new_energy
            else:
                # Accept with probability exp(-delta_e/T), T decreases
                T = 1.0 / (1 + i * 0.05)
                if rng.rand() < np.exp(-delta_e / max(T, 1e-6)):
                    current_pos = new_pos
                    current_energy = new_energy
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_pos, best_energy


def relax(pos):
    """Relax positions using L-BFGS."""
    from scipy.optimize import minimize
    result = minimize(
        lambda x: lj_energy_and_grad(x)[0],
        pos,
        jac=lambda x: lj_energy_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    return result.x, result.fun


def construct():
    global START_TIME
    START_TIME = time.time()
    
    rng = np.random.RandomState(12345)
    
    best_pos = None
    best_energy = np.inf
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Icosahedral seed
    try:
        ico = icosahedral_seed()
        if check_valid(ico):
            seeds.append(ico)
    except:
        pass
    
    # 2. FCC truncated octahedron seed
    try:
        fcc = fcc_truncated_octahedron_seed()
        if check_valid(fcc):
            seeds.append(fcc)
    except:
        pass
    
    # 3. Random seeds
    for _ in range(5):
        try:
            rseed = random_seed(rng)
            if check_valid(rseed):
                seeds.append(rseed)
        except:
            pass
    
    # 4. Perturbed icosahedral seeds
    for _ in range(3):
        try:
            ico = icosahedral_seed()
            perturbed = ico + rng.normal(0, 0.2, size=(N, 3))
            if check_valid(perturbed):
                seeds.append(perturbed)
        except:
            pass
    
    # 5. Perturbed FCC seeds
    for _ in range(3):
        try:
            fcc = fcc_truncated_octahedron_seed()
            perturbed = fcc + rng.normal(0, 0.2, size=(N, 3))
            if check_valid(perturbed):
                seeds.append(perturbed)
        except:
            pass
    
    # Process each seed: relax, then basin hopping
    for seed in seeds:
        if time.time() - START_TIME > DEADLINE:
            break
        
        try:
            # First relax
            relaxed_pos, relaxed_energy = relax(seed)
            if check_valid(relaxed_pos) and relaxed_energy < best_energy:
                best_energy = relaxed_energy
                best_pos = relaxed_pos.copy()
            
            # Then basin hopping from the relaxed position
            bh_pos, bh_energy = basin_hopping(relaxed_pos, max_iters=30, rng=rng)
            if check_valid(bh_pos) and bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except:
            continue
    
    # Final polish with tighter tolerance
    if best_pos is not None:
        try:
            from scipy.optimize import minimize
            result = minimize(
                lambda x: lj_energy_and_grad(x)[0],
                best_pos,
                jac=lambda x: lj_energy_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if check_valid(result.x) and result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x
        except:
            pass
    
    if best_pos is None:
        # Fallback: return the seed program's output
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        return np.array(points[:N], dtype=float) * 1.12
    
    return best_pos
