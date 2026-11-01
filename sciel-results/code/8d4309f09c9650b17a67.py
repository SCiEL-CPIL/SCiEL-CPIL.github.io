import numpy as np
import time

N = 55

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # 10s margin

    # 1. Construct Mackay Icosahedron
    phi = (1 + np.sqrt(5)) / 2
    
    # Vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize vertices to unit length
    norms = np.linalg.norm(verts, axis=1)
    verts_unit = verts / norms[:, np.newaxis]
    
    # Initial radii estimates
    R1 = 1.0
    R2 = 1.5
    
    # Inner shell: 12 atoms
    inner_shell = R1 * verts_unit
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Outer shell:
    # 12 atoms at vertices
    outer_vertices = R2 * verts_unit
    
    # 30 atoms at edge centers
    dists = np.sum((verts_unit[:, np.newaxis, :] - verts_unit[np.newaxis, :, :])**2, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    edge_mask = dists < min_dist * 1.01
    
    edge_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-6:
                    edge_centers.append(R2 * (mid / mid_norm))
                    
    edge_centers = np.array(edge_centers)
    
    # Combine all atoms
    positions = np.vstack([center, inner_shell, outer_vertices, edge_centers])
    
    if len(positions) != N:
        # Fallback
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        return np.array(points[:N], dtype=float) * 1.12

    # 2. Relax
    try:
        import science_chemistry
        # Check time before calling relax
        if time.time() - start_time < deadline - 5:
            relaxed = science_chemistry.relax(positions)
            if relaxed is not None and len(relaxed) == N:
                return relaxed
    except Exception:
        pass
    
    # Return the constructed Mackay structure if relaxation fails or times out
    return positions
