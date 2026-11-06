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
    # Normalize to unit radius
    verts = verts / np.linalg.norm(verts[0])
    return verts

def _get_icosahedron_faces(verts):
    # Find faces by checking triples of vertices
    # An icosahedron face is an equilateral triangle on the surface
    # Edge length squared
    d2 = np.sum((verts[:, np.newaxis, :] - verts[np.newaxis, :, :])**2, axis=2)
    min_dist = np.min(d2[d2 > 1e-9])
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(d2[i, j] - min_dist) < 1e-6:
                for k in range(j+1, 12):
                    if abs(d2[i, k] - min_dist) < 1e-6 and abs(d2[j, k] - min_dist) < 1e-6:
                        faces.append((i, j, k))
    return faces

def _generate_special_points():
    """Generates the 63 'special' points of the 2-shell Mackay icosahedron geometry."""
    verts = _get_icosahedron_vertices()
    # Scale to a reasonable radius for the potential
    # The potential has a minimum around r ~ 1.1-1.2
    # The icosahedron vertices should be at the first shell radius
    R1 = 1.1 
    verts = verts * R1
    
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Edge midpoints
    faces = _get_icosahedron_faces(verts)
    # We need edges, not just faces. 
    # Edges are pairs of vertices that share a face.
    edges = set()
    for f in faces:
        edges.add(tuple(sorted((f[0], f[1]))))
        edges.add(tuple(sorted((f[1], f[2]))))
        edges.add(tuple(sorted((f[0], f[2]))))
    
    edge_midpoints = []
    for e in edges:
        mid = (verts[e[0]] + verts[e[1]]) / 2.0
        # Project to sphere? No, Mackay icosahedron atoms are on the surface of the icosahedron shell?
        # Actually, the 2nd shell atoms are further out.
        # Let's scale them to be on a sphere of radius R2.
        # R2 is typically larger than R1.
        # For Mackay icosahedron, the ratio is fixed by geometry.
        # The distance from center to edge midpoint of unit icosahedron is cos(31.7 deg) ~ 0.85?
        # No, the edge midpoint is closer to the center than the vertices?
        # In a regular icosahedron, the edge midpoint is CLOSER to the center than the vertices.
        # But in the Mackay icosahedron, the 2nd shell atoms are FURTHER out.
        # So we should project the edge midpoints to a larger radius.
        # Let's assume the 2nd shell radius is R2 = 1.4 * R1?
        # Actually, let's just use the geometric position and scale it.
        norm_mid = np.linalg.norm(mid)
        if norm_mid > 1e-9:
            mid_scaled = mid / norm_mid * (R1 * 1.3) # Heuristic radius for 2nd shell
            edge_midpoints.append(mid_scaled)
            
    edge_midpoints = np.array(edge_midpoints)
    
    # Face centers
    face_centers = []
    for f in faces:
        fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        norm_fc = np.linalg.norm(fc)
        if norm_fc > 1e-9:
            fc_scaled = fc / norm_fc * (R1 * 1.3)
            face_centers.append(fc_scaled)
            
    face_centers = np.array(face_centers)
    
    # Combine: Center (1), Vertices (12), Edge Midpoints (30), Face Centers (20)
    # Total 63
    all_points = np.vstack([center, verts, edge_midpoints, face_centers])
    return all_points

def construct(instance):
    """Best of several icosahedral starting clusters, each relaxed locally."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 290.0 # 10s safety margin
    
    special_points = _generate_special_points()
    
    best_energy = np.inf
    best_positions = None
    
    # Number of restarts
    n_restarts = 20
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Generate a candidate
        if size <= 63:
            # Select 'size' points from special_points
            # Prioritize core: Center, Vertices, then Edge Midpoints, then Face Centers
            # To add variety, we can randomly drop some from the outer shells
            # Or just pick a random subset, but ensure connectivity?
            # Let's pick a random subset of indices, but bias towards the core.
            # Weights: Center (10), Vertices (5), Edges (2), Faces (1)
            weights = np.zeros(63)
            weights[0] = 10.0
            weights[1:13] = 5.0
            weights[13:43] = 2.0
            weights[43:63] = 1.0
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Sample without replacement
            indices = np.random.choice(63, size=size, replace=False, p=weights)
            candidate = special_points[indices]
        else:
            # For size > 63, start with all 63 and add random points
            candidate = special_points.copy()
            n_add = size - 63
            # Generate random points on the surface of the icosahedron (radius ~ 1.5)
            new_points = []
            for _ in range(n_add):
                # Random direction
                v = np.random.randn(3)
                v = v / np.linalg.norm(v)
                # Project to sphere of radius 1.5
                p = v * 1.5
                new_points.append(p)
            new_points = np.array(new_points)
            candidate = np.vstack([candidate, new_points])
            
        # Relax
        try:
            energy, positions = relax(candidate, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
            
    return best_positions
# EVOLVE-BLOCK-END
