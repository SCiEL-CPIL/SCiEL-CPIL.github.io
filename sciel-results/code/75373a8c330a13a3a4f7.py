import numpy as np
import time
import science_chemistry

N = 55

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    # Safety margin: return best result before 140s (limit is 150s)
    deadline = 140.0

    # 1. Construct Analytic Mackay Icosahedron
    # This is the known global minimum structure for N=55 LJ clusters.
    # Structure: 1 center, 12 inner shell (icosahedron vertices), 
    #            30 middle shell (icosahedron edge midpoints), 
    #            12 outer shell (icosahedron vertices)
    
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of a regular icosahedron
    # Cyclic permutations of (0, 1, phi)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize vertices to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find the 30 edge directions (midpoints of connected vertices)
    # Calculate pairwise distances to identify edges (shortest non-zero distance)
    diff = verts[:, np.newaxis, :] - verts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_edge_len = np.min(dists)
    
    edge_dirs = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_edge_len) < 1e-6:
                # Midpoint direction
                mid_vec = verts[i] + verts[j]
                mid_vec = mid_vec / np.linalg.norm(mid_vec)
                edge_dirs.append(mid_vec)
                
    edge_dirs = np.array(edge_dirs, dtype=float)
    # Should have 30 directions
    if len(edge_dirs) != 30:
        # Fallback: just use random or simple lattice if geometry fails
        # But this should never happen for a standard icosahedron
        return np.random.rand(55, 3) * 2.0 + 0.5

    # Initial radii for the shells
    # These are approximations; relaxation will optimize them.
    # r_min for LJ is ~1.1225
    r_center = 0.0
    r_inner = 1.10
    r_mid = 1.65
    r_outer = 2.20

    positions = np.zeros((55, 3), dtype=float)
    
    # Atom 0: Center
    positions[0] = [0.0, 0.0, 0.0]
    
    # Atoms 1-12: Inner Shell (Vertices)
    positions[1:13] = r_inner * verts
    
    # Atoms 13-42: Middle Shell (Edge Midpoints)
    positions[13:43] = r_mid * edge_dirs
    
    # Atoms 43-54: Outer Shell (Vertices)
    positions[43:55] = r_outer * verts

    # Check time before relaxation
    if time.time() > deadline - 10.0:
        return positions

    # 2. Relax using L-BFGS
    try:
        relaxed_positions = science_chemistry.relax(positions)
        # Ensure the returned array is valid
        if relaxed_positions is not None and relaxed_positions.shape == (55, 3):
            # Check time again
            if time.time() < deadline:
                return relaxed_positions
    except Exception:
        pass

    # Fallback: return analytic solution if relaxation failed or timed out
    return positions

# EVOLVE-BLOCK-END
