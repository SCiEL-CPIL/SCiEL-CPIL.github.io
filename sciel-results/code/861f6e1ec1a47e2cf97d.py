"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

# Deadline guard: evaluator kills after 150s wall-clock; keep a safety margin
_DEADLINE = 120.0  # seconds
_start_time = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def lj_energy(pos):
    """Compute total Lennard-Jones energy for positions (N,3)."""
    n = len(pos)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r2 = np.sum((pos[i] - pos[j]) ** 2)
            if r2 < 1e-12:
                return 1e10
            r6 = r2 ** 3
            energy += 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return energy

def lj_energy_grad(pos):
    """Compute energy and gradient for Lennard-Jones potential."""
    n = len(pos)
    energy = 0.0
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pos[i] - pos[j]
            r2 = np.dot(diff, diff)
            if r2 < 1e-12:
                return 1e10, grad
            r6 = r2 ** 3
            r12 = r6 * r6
            # dE/dr = 4*(-12*r^-13 + 6*r^-7) = 24*r^-7 - 48*r^-13
            # But we need dE/dpos_i = -dE/dr * diff/r... let's use the standard formula
            # E_pair = 4*(r^-12 - r^-6)
            # dE_pair/dr = 4*(-12*r^-13 + 6*r^-7)
            # dE_pair/dpos_i = dE_pair/dr * d(pos_i-pos_j)/dpos_i / r = dE_pair/dr * (pos_i-pos_j)/r
            # Actually: dE/dpos_i = -4*(12*r^-14 - 6*r^-8) * (pos_i - pos_j) / r
            # Let me be more careful. Let u = pos_i - pos_j, r = |u|
            # E = 4*(r^-12 - r^-6)
            # dE/dr = 4*(-12*r^-13 + 6*r^-7)
            # dE/dpos_i = dE/dr * u/r = 4*(-12*r^-13 + 6*r^-7) * u / r
            #           = 4*(-12*r^-14 + 6*r^-8) * u
            # dE/dpos_j = -dE/dpos_i
            coeff = 4.0 * (-12.0 / (r2 * r12) + 6.0 / (r2 * r6))
            # Wait, r^-14 = r^-12 * r^-2, r^-8 = r^-6 * r^-2
            coeff = 4.0 * (-12.0 / (r2 * r12) + 6.0 / (r2 * r6))
            dE_di = coeff * diff
            energy += 4.0 * (1.0 / r12 - 1.0 / r6)
            grad[i] += dE_di
            grad[j] -= dE_di
    return energy, grad

def make_fcc_truncated_octahedron():
    """
    Generate a truncated octahedron (FCC) cluster of 38 atoms.
    The truncated octahedron for N=38 in FCC:
    - 1 atom at center
    - 12 atoms at (±1,0,0), (0,±1,0), (0,0,±1) * a (first shell)
    - 24 atoms at (±1,±1,0), (±1,0,±1), (0,±1,±1) * a/√2... 
    
    Actually, let's think about this more carefully. The FCC truncated octahedron 
    with 38 atoms has layers. A common construction:
    Layer 1: 1 atom
    Layer 2: 12 atoms (octahedral)
    Layer 3: 24 atoms (truncated octahedron vertices)
    Layer 4: 1 atom
    
    Total: 1+12+24+1 = 38
    
    The FCC lattice vectors: a/2*(1,1,0), a/2*(1,0,1), a/2*(0,1,1)
    
    For a truncated octahedron, the atoms are at FCC lattice points within 
    a certain radius. Let's use the known structure.
    
    Actually, a simpler approach: the 38-atom FCC truncated octahedron can be 
    constructed as follows. The FCC unit cell has atoms at corners and face centers.
    For a small cluster, we take FCC lattice points within a sphere.
    
    Let me use a known coordinate set for the 38-atom LJ global minimum.
    The structure is a truncated octahedron. In FCC:
    - Center: (0,0,0)
    - 12 nearest neighbors at distance a: (±a,0,0), (0,±a,0), (0,0,±a)
      But wait, in FCC the nearest neighbors are at a/√2 * (1,1,0) type positions.
    
    Let me reconsider. In FCC with conventional cubic cell side a:
    - Atoms at (0,0,0), (a/2,a/2,0), (a/2,0,a/2), (0,a/2,a/2)
    - Nearest neighbor distance = a/√2
    
    For the truncated octahedron cluster of 38 atoms, a standard construction:
    Take FCC lattice points within radius R from center.
    
    Let's just use a perturbed icosahedron or a known good starting point 
    and relax it. Or better, construct the FCC truncated octahedron properly.
    
    The 38-atom truncated octahedron in FCC:
    Shell 0: 1 atom at origin
    Shell 1: 12 atoms at FCC nearest neighbor positions
    Shell 2: 24 atoms at the next shell
    
    In FCC with nearest neighbor distance = 1 (reduced units where sigma=1, 
    and the minimum is at r=2^(1/6)≈1.12246):
    
    FCC lattice vectors: v1=(1,1,0)/√2, v2=(1,0,1)/√2, v3=(0,1,1)/√2
    (These give nearest neighbor distance 1)
    
    Actually, let's just place atoms on an FCC lattice with spacing such that 
    nearest neighbors are at 2^(1/6).
    
    FCC primitive vectors (for nearest neighbor distance d):
    v1 = d/√2 * (1, 1, 0)
    v2 = d/√2 * (1, 0, 1)  
    v3 = d/√2 * (0, 1, 1)
    
    Lattice points: n1*v1 + n2*v2 + n3*v3 for integers n1,n2,n3
    Plus face-center offsets: (v1+v2)/2, (v1+v3)/2, (v2+v3)/2
    
    This is getting complex. Let me use a simpler approach: 
    generate FCC points and select 38 closest to origin.
    """
    d = 2.0 ** (1.0/6.0)  # LJ minimum distance
    
    # FCC lattice points
    # Primitive vectors
    v1 = np.array([1, 1, 0]) * d / np.sqrt(2)
    v2 = np.array([1, 0, 1]) * d / np.sqrt(2)
    v3 = np.array([0, 1, 1]) * d / np.sqrt(2)
    
    # Generate FCC points: n1*v1 + n2*v2 + n3*v3 + offset
    # Offsets for FCC: (0,0,0), (v1+v2)/2, (v1+v3)/2, (v2+v3)/2
    offsets = [
        np.array([0, 0, 0]),
        (v1 + v2) / 2,
        (v1 + v3) / 2,
        (v2 + v3) / 2
    ]
    
    points = []
    max_n = 3
    for n1 in range(-max_n, max_n + 1):
        for n2 in range(-max_n, max_n + 1):
            for n3 in range(-max_n, max_n + 1):
                base = n1 * v1 + n2 * v2 + n3 * v3
                for off in offsets:
                    p = base + off
                    points.append(p)
    
    points = np.array(points)
    # Remove duplicates (within tolerance)
    unique = [points[0]]
    for p in points[1:]:
        if all(np.linalg.norm(p - u) > 1e-6 for u in unique):
            unique.append(p)
    points = np.array(unique)
    
    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx][:N]
    
    return points

def check_validity(pos):
    """Check that all atoms are at least 0.5 apart."""
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) < 0.5:
                return False
    return True

def relax(pos, max_iter=200):
    """Relax using L-BFGS."""
    def obj_and_grad(x):
        x = x.reshape(-1, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    x0 = pos.reshape(-1)
    try:
        res = minimize(obj_and_grad, x0, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'maxls': 50, 'gtol': 1e-8})
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return pos, lj_energy(pos)

def random_icosahedron():
    """Generate a random icosahedral-like starting configuration."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Scale to LJ minimum distance
    d = 2.0 ** (1.0/6.0)
    scale = d / np.min(np.linalg.norm(ico[None,:,:] - ico[:,None,:], axis=2))
    ico = ico * scale
    
    # For 38 atoms: 1 center + 12 first shell + 24 second shell
    # This is approximate; we'll add random perturbations
    np.random.seed(42)
    pos = np.zeros((N, 3))
    pos[0] = 0
    pos[1:13] = ico * 1.0
    # Second shell: 24 atoms, placed at midpoints of edges of icosahedron
    # or at a slightly larger radius with angular positions
    # For simplicity, place them at radius ~1.7 * d with random angles
    # Better: use the known 38-atom structure
    
    # Actually, let's just use the FCC construction and perturb it
    return make_fcc_truncated_octahedron() + np.random.normal(0, 0.1, (N, 3))

def construct():
    global _start_time
    _start_time = time.time()
    
    best_pos = None
    best_energy = 1e10
    energy_threshold = -50.0  # Reject configurations with energy > -50 (likely bad/repulsive)
    
    def try_candidate(pos, name=""):
        nonlocal best_pos, best_energy
        if not check_validity(pos):
            return
        relaxed, e = relax(pos)
        if check_validity(relaxed) and e < best_energy and e < energy_threshold:
            best_energy = e
            best_pos = relaxed
            print(f"Found {name}: E={e}")

    # Strategy 1: Start from FCC truncated octahedron and relax
    fcc_pos = make_fcc_truncated_octahedron()
    try_candidate(fcc_pos, "FCC")
    
    # Strategy 2: Perturbed FCC with different seeds
    np.random.seed(0)
    for trial in range(10):
        if _time_left() < 10:
            break
        pos = make_fcc_truncated_octahedron() + np.random.normal(0, 0.1, (N, 3))
        try_candidate(pos, f"FCC_Pert_{trial}")
    
    # Strategy 3: Random starts with relaxation
    # Use a more structured random start: place atoms on a grid with jitter
    for trial in range(10):
        if _time_left() < 10:
            break
        np.random.seed(200 + trial)
        # Create a rough cluster
        pos = np.zeros((N, 3))
        # Fill a 4x4x4 grid with jitter
        grid_points = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    grid_points.append([i, j, k])
        np.random.shuffle(grid_points)
        for i in range(N):
            p = grid_points[i]
            pos[i] = np.array(p) * 1.12 + np.random.normal(0, 0.1, 3)
        
        # Check validity before relaxing to save time
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(pos[i] - pos[j]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            try_candidate(pos, f"Random_{trial}")

    if best_pos is None:
        # Fallback: Use the unrelaxed FCC guess if it's valid, as it should be close to the minimum
        if check_validity(fcc_pos):
            best_pos = fcc_pos
            best_energy = lj_energy(fcc_pos)
        else:
            # Fallback: simple cubic
            side = int(np.ceil(N ** (1 / 3)))
            points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
            best_pos = np.array(points[:N], dtype=float) * 1.12
            if not check_validity(best_pos):
                 # Force validity
                 for i in range(N):
                     for j in range(i+1, N):
                         if np.linalg.norm(best_pos[i] - best_pos[j]) < 0.5:
                             diff = best_pos[i] - best_pos[j]
                             if np.linalg.norm(diff) < 1e-10:
                                 diff = np.array([0.5, 0, 0])
                             diff = diff / np.linalg.norm(diff) * 0.5
                             best_pos[i] += diff / 2
                             best_pos[j] -= diff / 2

    return best_pos
# EVOLVE-BLOCK-END
