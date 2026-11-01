"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _lj_energy(pos):
    """Compute total Lennard-Jones energy for positions (N,3)."""
    n = pos.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e10  # avoid division by zero
            r6 = r2 ** 3
            total += 4.0 * (r6 ** (-2) - r6 ** (-1))
    return total


def _lj_energy_grad(pos):
    """Compute gradient of Lennard-Jones energy for positions (N,3)."""
    n = pos.shape[0]
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                continue
            r6 = r2 ** 3
            # dE/dr = 4*(-12*r^-13 + 6*r^-7) = 4*(-12/r^13 + 6/r^7)
            # dE/dri = dE/dr * (ri - rj)/r
            r = np.sqrt(r2)
            de_dr = 4.0 * (-12.0 / (r2 * r6) + 6.0 / r6) / r
            f = de_dr * rij
            grad[i] += f
            grad[j] -= f
    return grad


def _make_icosahedron(n_atoms):
    """
    Construct a Mackay icosahedral cluster with up to n_atoms atoms.
    Uses the standard Mackay icosahedron shells:
      Shell 0: 1 atom at center
      Shell 1: 12 atoms (icosahedron vertices)
      Shell 2: 42 atoms (30 face centers + 12 edge midpoints... actually 42 in second shell)
    Total for full 2-shell: 55 atoms.
    
    The Mackay icosahedron has:
      - 1 center
      - 12 vertices of a regular icosahedron (radius R1)
      - 30 face centers (radius R2) 
      - 12 edge midpoints? No, let me recall correctly.
    
    Actually, the Mackay icosahedron n=2 has:
      Shell 1 (12 atoms): vertices of icosahedron
      Shell 2 (42 atoms): 30 face centers + 12 "edge" positions
    
    Let me use the standard construction.
    """
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, golden], [0, -1, golden], [0, 1, -golden], [0, -1, -golden],
        [1, golden, 0], [-1, golden, 0], [1, -golden, 0], [-1, -golden, 0],
        [golden, 0, 1], [-golden, 0, 1], [golden, 0, -1], [-golden, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # For LJ, the optimal icosahedron radius for shell 1 is approximately 1.12 * sqrt(1.09) ≈ 1.17
    # But let's use a parameter and optimize. Standard Mackay uses specific ratios.
    # The ratio of shell radii in Mackay icosahedron: R2/R1 = sqrt(1 + 1/phi^2) or similar.
    # Let me just use R1 = 1.12 and R2 = 1.12 * 1.155 (approx for LJ)
    
    # Actually, for the 2-shell Mackay icosahedron with 55 atoms:
    # Shell 1: 12 atoms at radius R1
    # Shell 2: 42 atoms
    #   - 30 face centers: each face is an equilateral triangle. The centroid of a face
    #     of the icosahedron is at a certain radius.
    #   - 12 additional atoms (on edges or other positions)
    
    # Let me construct properly. The icosahedron has 20 faces.
    # Face normals point to the 20 face centers.
    
    # Get all 20 faces of the icosahedron
    # The icosahedron with vertices as above has specific face connectivity.
    # Let me compute face normals by finding all triplets that form faces.
    
    # Alternative: use known face normals for icosahedron
    # The 20 face centers of a regular icosahedron are the vertices of a dodecahedron.
    # Dodecahedron vertices can be constructed as:
    # (±1, ±1, ±1), (0, ±1/phi, ±phi), (±1/phi, ±phi, 0), (±phi, 0, ±1/phi)
    
    phi_inv = 1.0 / golden
    dodeca_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi_inv, golden], [0, phi_inv, -golden],
        [0, -phi_inv, golden], [0, -phi_inv, -golden],
        [phi_inv, golden, 0], [phi_inv, -golden, 0],
        [-phi_inv, golden, 0], [-phi_inv, -golden, 0],
        [golden, 0, phi_inv], [-golden, 0, phi_inv],
        [golden, 0, -phi_inv], [-golden, 0, -phi_inv]
    ], dtype=float)
    
    dodeca_norms = np.linalg.norm(dodeca_verts, axis=1, keepdims=True)
    dodeca_verts = dodeca_verts / dodeca_norms
    
    # Now we have:
    # 1 center atom
    # 12 icosahedron vertices (shell 1)
    # 20 face centers (dodecahedron vertices) - these are part of shell 2
    
    # For the Mackay icosahedron with 55 atoms, shell 2 has 42 atoms.
    # 20 face centers + 12 edge-related + 10 more? 
    # Actually: 1 + 12 + 42 = 55. The 42 in shell 2 are:
    #   - 30 atoms at the centers of the 30 edges of the icosahedron? No, that's not right either.
    
    # Let me look at this differently. The Mackay icosahedron I_2 has:
    #   Center: 1
    #   First shell (12): icosahedron vertices
    #   Second shell (42): These are the 30 face centers of the icosahedron AND 12 more
    #   Actually, I recall that the 42 atoms in the second shell of the Mackay icosahedron
    #   consist of: 30 at the midpoints of edges of the icosahedron? No.
    
    # Let me just use a simpler approach: generate a good initial guess using the icosahedral
    # symmetry and then relax.
    
    # For now, let me construct:
    # 1 center
    # 12 icosahedron vertices at radius R1
    # 20 face centers at radius R2
    # 12 edge midpoints at radius R3
    # 10 more? 
    # 1+12+20+12 = 45, need 10 more.
    
    # Actually, the correct Mackay icosahedron I_2 has:
    #   1 center
    #   12 first-shell (icosahedron vertices)
    #   42 second-shell: 30 face centers + 12 "pentagonal" positions
    #   Wait, 30 + 12 = 42. Yes!
    
    # The 30 face centers are the 20 faces... no, an icosahedron has 20 faces, not 30.
    # 30 is the number of edges.
    
    # I think the 42 second-shell atoms are:
    #   - 20 at face centers (one per face of the icosahedron)
    #   - 12 at edge midpoints (one per edge of the icosahedron)  
    #   - 10 at... ? 20+12=32, need 10 more.
    
    # I'm getting confused. Let me just use a different strategy:
    # Create a reasonable initial guess and let L-BFGS find the minimum.
    
    # Strategy: Place atoms in a shell-like structure with icosahedral symmetry
    # and relax.
    
    # For a 55-atom cluster, a good initial guess is:
    # - 1 center
    # - 12 at icosahedron vertices, radius ~1.15
    # - 20 at face centers, radius ~1.7
    # - 12 at edge midpoints, radius ~1.4
    # - 10 additional atoms to reach 55
    
    # Let me just use a simpler approach: random initialization with icosahedral symmetry hints
    # and then relax multiple times.
    
    pass  # We'll handle initialization in construct()


def construct():
    """
    Construct initial positions for 55-atom LJ cluster.
    Uses a Mackay icosahedron-inspired initial guess followed by L-BFGS relaxation.
    """
    np.random.seed(42)
    
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of a regular icosahedron (unit radius)
    verts = np.array([
        [0, 1, golden], [0, -1, golden], [0, 1, -golden], [0, -1, -golden],
        [1, golden, 0], [-1, golden, 0], [1, -golden, 0], [-1, -golden, 0],
        [golden, 0, 1], [-golden, 0, 1], [golden, 0, -1], [-golden, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # 20 face centers = vertices of dual dodecahedron
    phi_inv = 1.0 / golden
    dodeca = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi_inv, golden], [0, phi_inv, -golden],
        [0, -phi_inv, golden], [0, -phi_inv, -golden],
        [phi_inv, golden, 0], [phi_inv, -golden, 0],
        [-phi_inv, golden, 0], [-phi_inv, -golden, 0],
        [golden, 0, phi_inv], [-golden, 0, phi_inv],
        [golden, 0, -phi_inv], [-golden, 0, -phi_inv]
    ], dtype=float)
    dodeca_norms = np.linalg.norm(dodeca, axis=1, keepdims=True)
    dodeca_unit = dodeca / dodeca_norms
    
    # 12 edge midpoints of the icosahedron
    # Edges connect vertices that are closest to each other
    # For unit icosahedron, edge length is 2/sqrt(phi^2+1) * ... 
    # Let me compute: distance between [0,1,phi] and [1,phi,0]:
    # sqrt((1-0)^2 + (phi-1)^2 + (0-phi)^2) = sqrt(1 + (phi-1)^2 + phi^2)
    # phi^2 = phi+1, (phi-1)^2 = phi^2-2phi+1 = phi+1-2phi+1 = 2-phi
    # So dist^2 = 1 + 2-phi + phi+1 = 4. dist = 2.
    # Wait, that doesn't seem right for unit vertices.
    
    # Let me just find all pairs of vertices with the smallest distance
    n_v = len(verts_unit)
    min_dist = np.inf
    for i in range(n_v):
        for j in range(i+1, n_v):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if d < min_dist:
                min_dist = d
    
    edges = []
    for i in range(n_v):
        for j in range(i+1, n_v):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - min_dist) < 1e-10:
                edges.append((i, j))
    
    edge_midpoints = np.array([(verts_unit[i] + verts_unit[j]) / 2.0 for i, j in edges])
    edge_mid_norms = np.linalg.norm(edge_midpoints, axis=1, keepdims=True)
    edge_mid_unit = edge_midpoints / edge_mid_norms
    
    # Now build the cluster:
    # 1 center
    # 12 icosahedron vertices at radius R1
    # 20 face centers at radius R2  
    # 12 edge midpoints at radius R3
    # Total so far: 1 + 12 + 20 + 12 = 45
    # Need 10 more. Let me add 10 more at a larger radius with random directions
    # that respect some symmetry, or just use the remaining from a larger shell.
    
    # Actually, for the Mackay icosahedron, the second shell has 42 atoms.
    # 1 + 12 + 42 = 55.
    # The 42 are: 30 + 12 or 20 + 12 + 10.
    
    # Let me just use: 1 center + 12 verts + 20 faces + 12 edges + 10 extra
    # and place the 10 extra at radius R4 in random directions.
    
    R1 = 1.15  # icosahedron vertex radius
    R2 = 1.75  # face center radius
    R3 = 1.45  # edge midpoint radius
    R4 = 2.0   # extra atoms radius
    
    positions = [np.zeros(3)]  # center
    positions.extend([R1 * v for v in verts_unit])  # 12
    positions.extend([R2 * v for v in dodeca_unit])  # 20
    positions.extend([R3 * v for v in edge_mid_unit])  # 12
    # Need 10 more
    for _ in range(10):
        # Random direction
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)
        positions.append(R4 * v)
    
    pos = np.array(positions, dtype=float)
    
    # Relax with L-BFGS
    try:
        result = minimize(
            _lj_energy,
            pos,
            jac=_lj_energy_grad,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        pos = result.x
    except Exception:
        pass
    
    # Ensure minimum distance constraint
    # Check and fix any violations
    n = pos.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < 0.5:
                # Push apart
                if d < 1e-10:
                    pos[j] += np.array([0.5, 0.5, 0.5])
                else:
                    direction = (pos[i] - pos[j]) / d
                    pos[i] += direction * 0.25
                    pos[j] -= direction * 0.25
    
    return pos

# EVOLVE-BLOCK-END
