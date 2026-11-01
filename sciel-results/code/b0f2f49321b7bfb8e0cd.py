"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy_and_grad(pos):
    """Compute Lennard-Jones energy and gradient for a set of positions."""
    n = pos.shape[0]
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    
    r6 = r2**(-3)
    r12 = r6**2
    
    # Energy: 4 * sum(r^-12 - r^-6) over all pairs (count each pair once)
    # The matrix has each pair twice, so divide by 2
    E = 2.0 * np.sum(r12 - r6)  # 4 * (1/2) * sum = 2 * sum
    
    # Gradient: dE/dpos_i = 4 * sum_j (12*r_i^-13 - 6*r_i^-7) * (pos_i - pos_j)
    # In terms of r2: dE/dpos_i = 24 * sum_j (2*r12 - r6) * (pos_i - pos_j) / r2
    # Using the full matrix (each pair appears twice), we need to be careful
    # Actually: E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # dE/dpos_i = 4 * sum_{j!=i} (-12*r_ij^-14 + 6*r_ij^-8) * (pos_i - pos_j)
    #            = 24 * sum_{j!=i} (2*r_ij^-14 - r_ij^-8) * (pos_i - pos_j)
    # In vector form with r2: r^-14 = r2^(-7), r^-8 = r2^(-4)
    # coeff = 24 * (2*r2^(-7) - r2^(-4)) = 24 * r2^(-7) * (2 - r2^(-3)) = 24*r2^(-7)*(2 - r6)
    
    coeff = 24.0 * (2.0 * r2**(-7) - r2**(-4))
    coeff = coeff * diff  # shape (n, n, 3)
    grad = np.sum(coeff, axis=1)
    
    return E, grad

def _make_mackay_icosahedron(n_atoms=55):
    """
    Generate a Mackay icosahedron cluster.
    For 55 atoms: 1 center + 12 first shell + 42 second shell.
    """
    # icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Standard icosahedron vertices (normalized to unit radius)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit length
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # First shell: 12 atoms at distance ~1.0 from center (in LJ units, optimal ~1.1225)
    r1 = 1.0
    shell1 = verts * r1
    
    # Second shell: 42 atoms. These are at the centers of the icosahedron faces and edges.
    # For a Mackay icosahedron, the second shell consists of:
    # - 30 atoms at edge midpoints (scaled)
    # - 12 atoms at face centers (scaled)
    # Total = 42
    
    # Edge midpoints: 30 edges of icosahedron
    # We need to find all edges. An edge connects two vertices if their dot product is a specific value.
    # For unit icosahedron, adjacent vertices have dot product = 1/phi^2 = 1/(phi*phi)
    dot_adj = 1.0 / (phi * phi)
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.dot(verts[i], verts[j])
            if abs(d - dot_adj) < 1e-6:
                edges.append((i, j))
    
    # Should have 30 edges
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    # Normalize
    edge_mids /= np.linalg.norm(edge_mids, axis=1, keepdims=True)
    
    # Face centers: 20 faces. For a Mackay icosahedron with 42 second-shell atoms,
    # we need 12 face centers? No, 30+12=42. So 12 face centers.
    # Actually, the standard Mackay icosahedron second shell has 42 atoms:
    # 30 at edge centers and 12 at face centers of the *truncated* or specific arrangement.
    # Let me use a simpler approach: place second shell atoms on a sphere at radius r2
    # using a known configuration. Alternatively, use a Fibonacci sphere or known coordinates.
    
    # For now, let's use a simpler second shell: 42 points on a sphere.
    # We can use the vertices of a truncated icosahedron or just distribute evenly.
    # A better approach: the 42 points are the vertices of a 42-point configuration.
    # Let's use a golden spiral distribution for 42 points on a sphere at radius r2.
    
    r2 = 2.0  # approximate second shell radius
    n2 = 42
    # Golden spiral
    inc = np.pi * (3.0 - np.sqrt(5.0))
    golden_angle = inc
    y = np.linspace(-1, 1, n2)
    radius = np.sqrt(1 - y**2)
    theta = golden_angle * np.arange(n2)
    
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    shell2 = np.column_stack([x, y, z]) * r2
    
    # Combine: 1 center + 12 shell1 + 42 shell2 = 55
    center = np.array([[0.0, 0.0, 0.0]])
    all_points = np.vstack([center, shell1, shell2])
    
    # Scale to LJ minimum distance ~1.1225 for first shell
    # Actually, let's just return the structure and let optimization refine it
    return all_points[:n_atoms]

def _cubic_lattice_fragment(n_atoms=55, spacing=1.12):
    """Generate a cubic lattice fragment."""
    side = int(np.ceil(n_atoms ** (1/3)))
    points = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                points.append([i, j, k])
    points.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
    return np.array(points[:n_atoms], dtype=float) * spacing

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_pos = None
    best_energy = np.inf
    
    # Try multiple initial structures
    initial_structures = []
    
    # 1. Mackay icosahedron
    try:
        mackay = _make_mackay_icosahedron(55)
        initial_structures.append(mackay)
    except:
        pass
    
    # 2. Cubic lattice
    cubic = _cubic_lattice_fragment(55, 1.12)
    initial_structures.append(cubic)
    
    # 3. Random with minimum distance constraint
    for trial in range(3):
        np.random.seed(42 + trial)
        pos = np.random.randn(55, 3) * 3.0
        # Ensure minimum distance
        for i in range(55):
            for j in range(i+1, 55):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 0.5:
                    pos[j] = pos[i] + 0.6 * (pos[j] - pos[i]) / max(d, 1e-6)
        initial_structures.append(pos)
    
    for init_pos in initial_structures:
        if time.time() > deadline:
            break
        
        try:
            # Local relaxation with L-BFGS
            def fun(x):
                return _lj_energy_and_grad(x)
            
            result = minimize(fun, init_pos, method='L-BFGS-B', 
                             options={'maxiter': 2000, 'gtol': 1e-8})
            
            if result.success or result.fun < best_energy:
                e, g = _lj_energy_and_grad(result.x)
                if e < best_energy:
                    best_energy = e
                    best_pos = result.x.copy()
        except Exception as e:
            pass
    
    # If no valid result, return cubic lattice
    if best_pos is None:
        return _cubic_lattice_fragment(55, 1.12)
    
    return best_pos
# EVOLVE-BLOCK-END
