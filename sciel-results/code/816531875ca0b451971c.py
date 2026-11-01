"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time

try:
    from science_chemistry import relax
except ImportError:
    def relax(positions):
        return positions

def _generate_mackay_icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    
    # 1. Icosahedron vertices (12)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    norms_ico = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_unit = ico_verts / norms_ico
    
    # 2. Find edge midpoints (30)
    # Calculate all pairwise distances
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico_unit[i] - ico_unit[j])
            dists[i, j] = d
            dists[j, i] = d
    
    # Minimum distance is the edge length
    min_dist = np.min(dists[dists > 0])
    
    # Collect midpoints of edges
    midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.abs(dists[i, j] - min_dist) < 1e-10:
                mid = (ico_unit[i] + ico_unit[j]) / 2.0
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-10:
                    midpoints.append(mid / mid_norm)
    
    edge_mid_unit = np.array(midpoints)
    
    # 3. Construct positions
    # Radii: r1 ~ 1.0, r2 ~ 2.0
    r1 = 1.0
    r2 = 2.0
    
    positions = []
    # Center
    positions.append([0.0, 0.0, 0.0])
    
    # Shell 1: 12 ico vertices
    for v in ico_unit:
        positions.append(v * r1)
        
    # Shell 2: 12 ico vertices + 30 edge midpoints
    for v in ico_unit:
        positions.append(v * r2)
    for v in edge_mid_unit:
        positions.append(v * r2)
        
    return np.array(positions)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    positions = _generate_mackay_icosahedron()
    
    try:
        relaxed_positions = relax(positions)
        if relaxed_positions is not None and len(relaxed_positions) == N:
            positions = relaxed_positions
    except Exception:
        pass
    
    return positions

# EVOLVE-BLOCK-END
