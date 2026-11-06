# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16

def _energy_and_gradient(positions):
    """Compute Dzugutov energy and gradient from scratch."""
    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-8:
                continue
            if r < a_POT:
                V = A_POT * (r ** (-m_POT) - B_POT) * np.exp(c_POT / (r - a_POT))
                dVdr = A_POT * (-m_POT * r ** (-m_POT - 1) * np.exp(c_POT / (r - a_POT))
                          + (r ** (-m_POT) - B_POT) * np.exp(c_POT / (r - a_POT)) * (-c_POT / (r - a_POT) ** 2))
            elif r < b_POT:
                V = B_POT * np.exp(d_POT / (r - b_POT))
                dVdr = B_POT * np.exp(d_POT / (r - b_POT)) * (-d_POT / (r - b_POT) ** 2)
            else:
                continue
            
            energy += V
            f = dVdr / r
            grad[i] += f * rij
            grad[j] -= f * rij
    
    return energy, grad


def _build_icosahedral_shells(n, spacing=1.0):
    """Build a cluster using icosahedral shell structure."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0]) * spacing
    
    # Second shell: octahedron vertices
    oct_verts = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float) * spacing * 1.5
    
    # Third shell: 30 vertices (edges of icosahedron, midpoint direction)
    # Use face centers of the icosahedron (20 faces)
    ico_faces = [
        [0,1,3],[0,3,4],[0,4,6],[0,6,5],[0,5,1],
        [1,5,8],[1,8,9],[1,9,7],[1,7,3],
        [3,7,11],[3,11,10],[3,10,4],
        [4,10,12],[4,12,6],
        [6,12,14],[6,14,5],
        [5,14,15],[5,15,8],
        [8,15,17],[8,17,9],
        [9,17,16],[9,16,7],
        [7,16,13],[7,13,11],
        [11,13,18],[11,18,10],
        [10,18,19],[10,19,12],
        [12,19,20],[12,20,14],
        [14,20,21],[14,21,15],
        [15,21,16]
    ]
    # Wait, icosahedron has 20 faces, not 30. Let me recalculate.
    # Actually, let me use a different approach for the third shell.
    
    # Simpler: use the icosahedron vertices at different radii
    # and add more vertices from the golden ratio structure.
    
    positions = []
    
    # Center
    if n >= 1:
        positions.append([0.0, 0.0, 0.0])
        n -= 1
    
    # First shell: 12 icosahedron vertices
    shell1_size = min(n, 12)
    if shell1_size > 0:
        positions.extend(ico_verts[:shell1_size].tolist())
        n -= shell1_size
    
    # Second shell: 30 vertices (edge midpoints direction, at larger radius)
    # Use 30 points from the icosahedron's edge directions
    edge_dirs = []
    edges = [
        (0,1),(0,3),(0,4),(0,5),(0,6),(0,7),
        (1,3),(1,5),(1,7),(1,8),(1,9),
        (3,4),(3,7),(3,10),(3,11),
        (4,6),(4,10),(4,12),
        (5,6),(5,8),(5,14),(5,15),
        (6,12),(6,14),
        (7,9),(7,11),(7,13),(7,16),
        (8,9),(8,15),(8,17),
        (9,16),(9,17),
        (10,11),(10,12),(10,18),(10,19),
        (11,13),(11,18),
        (12,19),(12,20),
        (13,16),(13,18),
        (14,15),(14,20),(14,21),
        (15,16),(15,21),
        (16,17),(16,21),
        (17,21)
    ]
    # That's not right either. Let me use a simpler approach.
    
    # For the second shell, use 30 vertices from a truncated icosahedron-like structure
    # or simply scale up the icosahedron and add more.
    
    # Simpler approach: use the icosahedron vertices at a larger radius
    # and interpolate
    if n > 0:
        # Add 30 vertices at radius ~1.5*spacing
        r2 = spacing * 1.55
        # Use 30 directions from the icosahedron's symmetry
        # 12 vertices + 20 face centers = 32, take 30
        # Face centers: average of each triangular face
        # Icosahedron faces (20)
        ico_face_idx = [
            (0,1,3),(0,3,4),(0,4,6),(0,6,5),(0,5,1),
            (1,5,8),(1,8,9),(1,9,7),(1,7,3),
            (3,7,11),(3,11,10),(3,10,4),
            (4,10,12),(4,12,6),
            (6,12,14),(6,14,5),
            (5,14,15),(5,15,8),
            (8,15,17),(8,17,9),
            (9,17,16),(9,16,7),
            (7,16,13),(7,13,11),
            (11,13,18),(11,18,10),
            (10,18,19),(10,19,12),
            (12,19,20),(12,20,14),
            (14,20,21),(14,21,15),
            (15,21,16)
        ]
        # Wait, that's 32 faces, which is wrong. Icosahedron has 20 faces.
        # Let me just use a different strategy.
        
        # Use the 12 icosahedron vertices at r2, plus 18 more from octahedron directions
        oct_dirs = np.array([
            [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
            [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
            [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
            [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]
        ], dtype=float)
        oct_dirs = oct_dirs / np.linalg.norm(oct_dirs[0]) * r2
        
        shell2_candidates = np.vstack([ico_verts * (r2 / spacing), oct_dirs])
        shell2_size = min(n, 30)
        positions.extend(shell2_candidates[:shell2_size].tolist())
        n -= shell2_size
    
    # Third shell and beyond: add more at larger radii
    while n > 0:
        r3 = spacing * 2.1
        # Use a grid of directions
        more_dirs = np.array([
            [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
            [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
            [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
            [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1],
            [1,1,1],[-1,1,1],[1,-1,1],[-1,-1,1],
            [1,1,-1],[-1,1,-1],[1,-1,-1],[-1,-1,-1]
        ], dtype=float)
        more_dirs = more_dirs / np.linalg.norm(more_dirs[0]) * r3
        shell3_size = min(n, len(more_dirs))
        positions.extend(more_dirs[:shell3_size].tolist())
        n -= shell3_size
    
    return np.array(positions[:len(positions)])


def _build_bcc_cluster(n, spacing=1.0):
    """Build a BCC-like cluster."""
    positions = []
    # Simple cubic lattice with BCC center
    # For small clusters, just use a compact arrangement
    
    # Start with a cube and fill in
    max_dim = int(np.ceil(n ** (1/3))) + 1
    lattice = []
    for i in range(max_dim):
        for j in range(max_dim):
            for k in range(max_dim):
                lattice.append([i * spacing, j * spacing, k * spacing])
    
    # Add BCC centers
    for i in range(max_dim - 1):
        for j in range(max_dim - 1):
            for k in range(max_dim - 1):
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    # Select the n closest to center
    lattice = np.array(lattice)
    lattice = lattice - np.mean(lattice, axis=0)
    dists = np.linalg.norm(lattice, axis=1)
    idx = np.argsort(dists)[:n]
    return lattice[idx]


def _build_fcc_cluster(n, spacing=1.0):
    """Build an FCC-like cluster."""
    positions = []
    max_dim = int(np.ceil(n ** (1/3))) + 2
    lattice = []
    for i in range(max_dim):
        for j in range(max_dim):
            for k in range(max_dim):
                if (i + j + k) % 2 == 0:
                    lattice.append([i * spacing, j * spacing, k * spacing])
                else:
                    lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    lattice = np.array(lattice)
    lattice = lattice - np.mean(lattice, axis=0)
    dists = np.linalg.norm(lattice, axis=1)
    idx = np.argsort(dists)[:n]
    return lattice[idx]


def construct(instance):
    """Find low-energy Dzugutov clusters using structured initial guesses and local relaxation."""
    size = instance["n"]
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s
    
    # Generate multiple structured initial guesses
    candidates = []
    
    # 1. Icosahedral shell structure (favored by Dzugutov)
    ico = _build_icosahedral_shells(size)
    if len(ico) >= size:
        candidates.append(ico[:size])
    
    # 2. BCC structure
    bcc = _build_bcc_cluster(size)
    if len(bcc) >= size:
        candidates.append(bcc[:size])
    
    # 3. FCC structure
    fcc = _build_fcc_cluster(size)
    if len(fcc) >= size:
        candidates.append(fcc[:size])
    
    # 4. Random compact with minimum distance
    rng = np.random.default_rng(size)
    for trial in range(3):
        radius = 1.1 * size ** (1.0/3.0) * 0.8
        points = []
        attempts = 0
        while len(points) < size and attempts < size * 50:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                min_dist = min([np.linalg.norm(p - q) for q in points], default=np.inf)
                if min_dist >= 0.6:
                    points.append(p)
            attempts += 1
        if len(points) == size:
            candidates.append(np.array(points))
    
    # Relax each candidate and keep the best
    best_energy = np.inf
    best_positions = None
    
    for candidate in candidates:
        if len(candidate) != size:
            continue
        if time.time() > deadline - 30.0:
            break
        try:
            energy, positions = relax(candidate, 'dzugutov')
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # If we have time, do a second round of perturbation and relaxation
    if best_positions is not None and time.time() < deadline - 60.0:
        for _ in range(5):
            if time.time() > deadline - 40.0:
                break
            # Perturb the best solution
            noise = 0.15 * (np.random.default_rng(42 + _).normal(0, 1, (size, 3)))
            perturbed = best_positions + noise
            try:
                energy, positions = relax(perturbed, 'dzugutov')
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                continue
    
    # Final check: ensure minimum distance constraint
    if best_positions is not None:
        # Verify minimum distance
        min_d = np.inf
        for i in range(size):
            for j in range(i + 1, size):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < min_d:
                    min_d = d
        if min_d < 0.5:
            # Fallback to the first candidate
            return candidates[0]
    
    return best_positions if best_positions is not None else candidates[0]
# EVOLVE-BLOCK-END
