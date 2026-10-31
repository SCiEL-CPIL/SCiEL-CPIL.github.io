"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Deadline: kill before 145s wall-clock, leave safety margin
DEADLINE = 145.0
START_TIME = time.time()


def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy


def lj_grad_vectorized(positions):
    """Vectorized LJ gradient computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    coeff = 4.0 * (12.0 * inv_r12 - 6.0 * inv_r6) * inv_r2
    forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return forces


def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    return np.min(r2) >= min_dist * min_dist


def make_fcc_truncated_octahedron():
    """
    Create an FCC truncated octahedron cluster of 38 atoms.
    We generate FCC lattice points and select the 38 closest to the origin.
    """
    scale = 1.0 / math.sqrt(2.0)
    
    points = []
    max_coord = 5
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    if r2 <= 50:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    return np.array(selected, dtype=float)


def make_bcc_cluster():
    """Create a BCC cluster of 38 atoms."""
    points = []
    max_coord = 4
    scale = 1.0 / math.sqrt(3.0) * 1.0 # Approximate scale
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # BCC points: (i,j,k) and (i+0.5, j+0.5, k+0.5)
                r2 = i*i + j*j + k*k
                if r2 <= 40:
                    points.append((i * scale, j * scale, k * scale, r2))
                
                r2_b = (i+0.5)**2 + (j+0.5)**2 + (k+0.5)**2
                if r2_b <= 40:
                    points.append(((i+0.5) * scale, (j+0.5) * scale, (k+0.5) * scale, r2_b))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    return np.array(selected, dtype=float)


def make_icosahedral_cluster():
    """
    Create a Mackay icosahedron-like cluster.
    Layer 1: 1 atom (center)
    Layer 2: 12 atoms (vertices of icosahedron)
    Layer 3: 42 atoms (face centers + edge midpoints + vertex extensions)
    We need 38 atoms total.
    """
    # Golden ratio
    phi = (1 + math.sqrt(5)) / 2
    
    # Center
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Icosahedron vertices
    # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = [
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    verts = np.array(verts)
    # Normalize and scale
    norm = np.linalg.norm(verts[0])
    scale_12 = 1.0 / norm * 1.12 # Approximate bond length
    
    layer2 = verts * scale_12
    
    # Layer 3: Face centers of the icosahedron (20 faces) + some others
    # Faces are triangles of adjacent vertices.
    # This is complex to generate programmatically without a mesh library.
    # Instead, let's use a simpler "random shell" approach for the 3rd layer
    # or just take the next closest points from an FCC lattice rotated?
    # For simplicity and robustness, let's generate 25 random points on a sphere
    # of radius ~2.24 (2 * 1.12) and reject if too close to existing.
    
    current = np.vstack([center, layer2])
    target_n = 38
    current_n = len(current)
    needed = target_n - current_n
    
    rng = np.random.default_rng(123)
    new_points = []
    attempts = 0
    while len(new_points) < needed and attempts < 5000:
        # Random direction
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        # Radius between 2.0 and 2.5
        r = rng.uniform(2.0, 2.5)
        p = v * r
        
        # Check distance to all existing points
        dists = np.linalg.norm(current - p, axis=1)
        if np.min(dists) > 0.8: # Loose constraint for seed
            new_points.append(p)
        attempts += 1
        
    if len(new_points) < needed:
        # Fallback: just add random points far away
        while len(new_points) < needed:
            p = rng.uniform(-3, 3, size=3)
            dists = np.linalg.norm(current - p, axis=1)
            if np.min(dists) > 1.0:
                new_points.append(p)
            else:
                p[0] += 0.1
                
    layer3 = np.array(new_points)
    return np.vstack([current, layer3])


def relax_ljfg(positions, high_precision=False):
    """Relax using L-BFGS if available, otherwise return as-is."""
    try:
        from science_chemistry import relax
        return relax(positions)
    except ImportError:
        def energy_func(pos):
            pos = pos.reshape(N, 3)
            return lj_energy_vectorized(pos)
        
        def grad_func(pos):
            pos = pos.reshape(N, 3)
            return lj_grad_vectorized(pos).flatten()
            
        options = {'maxiter': 100, 'gtol': 1e-6}
        if high_precision:
            options = {'maxiter': 500, 'gtol': 1e-8}
            
        result = minimize(energy_func, positions.flatten(), jac=grad_func, method='L-BFGS-B', 
                         options=options)
        return result.x.reshape(N, 3)
    except Exception:
        return positions


def basin_hopping_step(positions, rng, kick_magnitude=0.3):
    """
    Perform a single basin hopping step:
    1. Kick: add random noise to positions.
    2. Relax: minimize energy.
    3. Accept if lower energy (Metropolis criterion or just greedy for global min).
    Here we return the relaxed position regardless, and let the caller decide.
    """
    kicked = positions + rng.normal(0, kick_magnitude, size=positions.shape)
    # Ensure validity after kick? Relaxation usually fixes it, but if atoms overlap too much, LJ blows up.
    # We'll rely on the minimizer to handle it, but check validity first to avoid NaNs.
    if not check_validity(kicked, min_dist=0.4): # Allow slight overlap for kick
        # Push apart
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(kicked[i] - kicked[j])
                if d < 0.4:
                    v = (kicked[i] - kicked[j]) / (d + 1e-10)
                    kicked[i] += v * (0.4 - d)
                    kicked[j] -= v * (0.4 - d)
    
    relaxed = relax_ljfg(kicked)
    return relaxed


def construct():
    """Main entry point: Multi-start Basin Hopping."""
    global START_TIME
    START_TIME = time.time()
    
    best_positions = None
    best_energy = float('inf')
    
    # Generate diverse seeds
    seeds = []
    
    # 1. FCC Truncated Octahedron (High priority)
    try:
        fcc = make_fcc_truncated_octahedron()
        if check_validity(fcc):
            seeds.append(fcc)
    except Exception:
        pass
        
    # 2. BCC Cluster
    try:
        bcc = make_bcc_cluster()
        if check_validity(bcc):
            seeds.append(bcc)
    except Exception:
        pass
        
    # 3. Icosahedral
    try:
        ico = make_icosahedral_cluster()
        if check_validity(ico):
            seeds.append(ico)
    except Exception:
        pass
        
    # 4. Random clusters (a few)
    for seed_val in [1, 2, 3]:
        rng = np.random.default_rng(seed_val)
        # Generate points in a sphere of radius 3
        pts = []
        while len(pts) < N:
            p = rng.normal(size=3)
            p = p / np.linalg.norm(p) * rng.uniform(0.5, 3.0)
            # Check distance
            valid = True
            for q in pts:
                if np.linalg.norm(p - q) < 0.6:
                    valid = False
                    break
            if valid:
                pts.append(p)
        if len(pts) == N:
            seeds.append(np.array(pts))

    # Relax all seeds first
    relaxed_seeds = []
    for seed in seeds:
        if time.time() - START_TIME > DEADLINE - 30:
            break
        try:
            rel = relax_ljfg(seed)
            if check_validity(rel):
                e = lj_energy_vectorized(rel)
                if e < best_energy:
                    best_energy = e
                    best_positions = rel
                relaxed_seeds.append(rel)
        except Exception:
            continue

    if best_positions is None:
        # Fallback if all relaxations failed
        rng = np.random.default_rng(0)
        best_positions = rng.uniform(-2, 2, size=(N, 3))
        best_energy = lj_energy_vectorized(best_positions)
        relaxed_seeds = [best_positions]

    # Basin Hopping Phase
    # We have time left? Let's use it to kick and relax the best candidates.
    # We'll kick the global best and maybe the second best.
    
    # Sort relaxed seeds by energy
    relaxed_seeds.sort(key=lambda x: lj_energy_vectorized(x))
    
    # Take top 3 unique-ish structures
    candidates = relaxed_seeds[:3]
    
    rng = np.random.default_rng(42)
    
    # Perform several BH steps on the best candidate
    # Since we don't have a Metropolis acceptance history, we just greedily keep the best.
    # To escape local minima, we kick with varying magnitudes.
    
    kicks_magnitudes = [0.2, 0.3, 0.5, 0.8, 1.0, 1.5]
    
    for mag in kicks_magnitudes:
        if time.time() - START_TIME > DEADLINE - 10:
            break
            
        # Kick the current best
        kicked = best_positions + rng.normal(0, mag, size=best_positions.shape)
        
        # Ensure basic validity
        if not check_validity(kicked, min_dist=0.4):
             # Simple separation
             for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(kicked[i] - kicked[j])
                    if d < 0.4:
                        v = (kicked[i] - kicked[j]) / (d + 1e-10)
                        kicked[i] += v * (0.4 - d)
                        kicked[j] -= v * (0.4 - d)

        try:
            relaxed = relax_ljfg(kicked)
            if check_validity(relaxed):
                e = lj_energy_vectorized(relaxed)
                if e < best_energy:
                    best_energy = e
                    best_positions = relaxed
        except Exception:
            pass

    # Final high-precision polish
    if time.time() - START_TIME < DEADLINE - 5:
        try:
            polished = relax_ljfg(best_positions, high_precision=True)
            if check_validity(polished):
                e = lj_energy_vectorized(polished)
                if e < best_energy:
                    best_energy = e
                    best_positions = polished
        except Exception:
            pass

    # Final validity check
    if not check_validity(best_positions):
        # Last resort: Grid
        side = int(np.ceil(N ** (1/3)))
        grid = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        grid.sort(key=lambda p: (p[0] - (side-1)/2)**2 + (p[1] - (side-1)/2)**2 + (p[2] - (side-1)/2)**2)
        best_positions = np.array(grid[:N], dtype=float) * 1.12

    return best_positions

# EVOLVE-BLOCK-END
