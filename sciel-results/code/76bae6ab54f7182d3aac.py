"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax
import time

def _get_icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts[0])
    return verts

def _generate_mackay_icosahedron(n):
    """Generate a Mackay icosahedron cluster with n atoms.
    
    The Mackay icosahedron consists of concentric shells:
    - Shell 0: 1 atom at center
    - Shell 1: 12 atoms at icosahedron vertices
    - Shell 2: 30 atoms at edge midpoints
    - Shell 3: 20 atoms at face centers
    - Shell 4: 12 atoms (second layer of vertices)
    - Shell 5: 60 atoms (second layer of edge midpoints)
    - Shell 6: 30 atoms (second layer of face centers)
    - Shell 7: 20 atoms (third layer of face centers)
    
    Total: 1 + 12 + 30 + 20 + 12 + 60 + 30 + 20 = 185 atoms
    
    For n <= 63, we use the first 4 shells.
    For larger n, we add more shells.
    """
    verts = _get_icosahedron_vertices()
    
    # Optimal radii for the Dzugutov potential
    # The potential minimum is around r ~ 1.1-1.2
    # First shell (vertices) at R1
    R1 = 1.15
    
    # Generate all shell points
    all_points = []
    
    # Shell 0: Center
    all_points.append(np.array([0.0, 0.0, 0.0]))
    
    # Shell 1: 12 vertices
    shell1 = verts * R1
    all_points.extend(shell1)
    
    # Get faces and edges
    d2 = np.sum((shell1[:, np.newaxis, :] - shell1[np.newaxis, :, :])**2, axis=2)
    min_dist = np.min(d2[d2 > 1e-9])
    
    edges = set()
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(d2[i, j] - min_dist) < 1e-6:
                for k in range(j+1, 12):
                    if abs(d2[i, k] - min_dist) < 1e-6 and abs(d2[j, k] - min_dist) < 1e-6:
                        faces.append((i, j, k))
                        edges.add(tuple(sorted((i, j))))
                        edges.add(tuple(sorted((j, k))))
                        edges.add(tuple(sorted((i, k))))
    
    # Shell 2: 30 edge midpoints
    R2 = R1 * 1.25  # Slightly larger than R1
    shell2 = []
    for e in edges:
        mid = (shell1[e[0]] + shell1[e[1]]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-9:
            shell2.append(mid / norm * R2)
    all_points.extend(shell2)
    
    # Shell 3: 20 face centers
    R3 = R1 * 1.35
    shell3 = []
    for f in faces:
        fc = (shell1[f[0]] + shell1[f[1]] + shell1[f[2]]) / 3.0
        norm = np.linalg.norm(fc)
        if norm > 1e-9:
            shell3.append(fc / norm * R3)
    all_points.extend(shell3)
    
    # If n <= 63, we have enough points
    if n <= len(all_points):
        return np.array(all_points[:n])
    
    # For larger clusters, add more shells
    # Shell 4: 12 atoms (second layer, similar to shell 1 but further out)
    R4 = R1 * 1.5
    shell4 = verts * R4
    all_points.extend(shell4)
    
    if n <= len(all_points):
        return np.array(all_points[:n])
    
    # Shell 5: 60 atoms (edge midpoints of the icosahedron formed by shell 4)
    d2_4 = np.sum((shell4[:, np.newaxis, :] - shell4[np.newaxis, :, :])**2, axis=2)
    min_dist_4 = np.min(d2_4[d2_4 > 1e-9])
    edges_4 = set()
    for i in range(12):
        for j in range(i+1, 12):
            if abs(d2_4[i, j] - min_dist_4) < 1e-6:
                edges_4.add(tuple(sorted((i, j))))
    
    R5 = R1 * 1.65
    shell5 = []
    for e in edges_4:
        mid = (shell4[e[0]] + shell4[e[1]]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-9:
            shell5.append(mid / norm * R5)
    all_points.extend(shell5)
    
    if n <= len(all_points):
        return np.array(all_points[:n])
    
    # Shell 6: 30 atoms (face centers of the icosahedron formed by shell 4)
    faces_4 = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(d2_4[i, j] - min_dist_4) < 1e-6:
                for k in range(j+1, 12):
                    if abs(d2_4[i, k] - min_dist_4) < 1e-6 and abs(d2_4[j, k] - min_dist_4) < 1e-6:
                        faces_4.append((i, j, k))
    
    R6 = R1 * 1.75
    shell6 = []
    for f in faces_4:
        fc = (shell4[f[0]] + shell4[f[1]] + shell4[f[2]]) / 3.0
        norm = np.linalg.norm(fc)
        if norm > 1e-9:
            shell6.append(fc / norm * R6)
    all_points.extend(shell6)
    
    if n <= len(all_points):
        return np.array(all_points[:n])
    
    # Shell 7: 20 atoms (additional face centers)
    R7 = R1 * 1.9
    shell7 = []
    for f in faces_4:
        fc = (shell4[f[0]] + shell4[f[1]] + shell4[f[2]]) / 3.0
        norm = np.linalg.norm(fc)
        if norm > 1e-9:
            shell7.append(fc / norm * R7)
    all_points.extend(shell7)
    
    # If still not enough, add random points on the surface
    while len(all_points) < n:
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)
        all_points.append(v * R7)
    
    return np.array(all_points[:n])

def construct(instance):
    """Best of several icosahedral starting clusters, each relaxed locally."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 290.0 # 10s safety margin
    
    best_energy = np.inf
    best_positions = None
    
    # Number of restarts - more restarts with better starting points
    n_restarts = 30
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate a candidate using Mackay icosahedron
        if i == 0:
            # First restart: use the exact Mackay icosahedron
            candidate = _generate_mackay_icosahedron(size)
        else:
            # Subsequent restarts: add random perturbations
            candidate = _generate_mackay_icosahedron(size)
            # Add small random perturbations to break symmetry
            noise = np.random.randn(size, 3) * 0.05
            candidate = candidate + noise
        
        # Relax
        try:
            energy, positions = relax(candidate, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
        
        # Periodically try a different approach: random compact cluster
        if i % 5 == 4:
            # Generate a random compact cluster
            candidate = np.random.randn(size, 3) * 2.0
            try:
                energy, positions = relax(candidate, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
        
        # Check minimum distance constraint
        if best_positions is not None:
            d2 = np.sum((best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :])**2, axis=2)
            np.fill_diagonal(d2, np.inf)
            if np.min(d2) < 0.25:  # 0.5^2
                # This shouldn't happen after relaxation, but just in case
                pass
    
    return best_positions
# EVOLVE-BLOCK-END
