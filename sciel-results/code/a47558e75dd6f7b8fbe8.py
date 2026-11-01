"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
import science_chemistry

def _generate_mackay_icosahedron(R1, R2):
    """
    Generate the coordinates for a two-shell Mackay icosahedron.
    R1: Radius of the inner shell (12 atoms).
    R2: Radius of the outer shell (42 atoms).
    """
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of a regular icosahedron
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize vertices to unit length
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices_unit = vertices / norms
    
    # Find edge midpoints
    # In this coordinate system, the distance between connected vertices is 2.0
    mids = []
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if abs(d - 2.0) < 1e-6:
                mid = (vertices[i] + vertices[j]) / 2.0
                mids.append(mid)
                
    mids = np.array(mids, dtype=float)
    if len(mids) != 30:
        # Fallback if edge detection fails (should not happen)
        # Just return center + inner + outer corners (43 atoms) and pad? 
        # Better to raise an error or use a fallback structure.
        # For robustness, if this fails, we return a cubic lattice.
        return None
        
    # Normalize midpoints to unit length
    mids_norms = np.linalg.norm(mids, axis=1, keepdims=True)
    mids_unit = mids / mids_norms

    # Construct positions
    # 1. Center
    pos = np.array([[0.0, 0.0, 0.0]])
    
    # 2. Inner shell (12 atoms)
    inner = vertices_unit * R1
    pos = np.vstack([pos, inner])
    
    # 3. Outer shell corners (12 atoms)
    outer_corners = vertices_unit * R2
    pos = np.vstack([pos, outer_corners])
    
    # 4. Outer shell edges (30 atoms)
    outer_edges = mids_unit * R2
    pos = np.vstack([pos, outer_edges])
    
    return pos

def construct():
    start_time = time.time()
    
    # Initial guess for Mackay icosahedron radii
    # These values are close to the optimal LJ minimum
    R1 = 1.12
    R2 = 2.24
    
    positions = _generate_mackay_icosahedron(R1, R2)
    
    if positions is None:
        # Fallback to seed program logic
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        return np.array(points[:N], dtype=float) * 1.12

    # Relax the structure using L-BFGS
    try:
        relaxed_positions = science_chemistry.relax(positions)
        if relaxed_positions is not None and relaxed_positions.shape == (N, 3):
            return relaxed_positions
    except Exception:
        pass
        
    # If relaxation fails or returns invalid, return the initial topological guess
    return positions
# EVOLVE-BLOCK-END
