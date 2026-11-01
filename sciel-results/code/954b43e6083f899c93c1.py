"""Lennard-Jones 55-atom cluster: search approach with Mackay icosahedron + relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

def lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def lj_energy_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    # Energy
    e = 4.0 * (r12 - r6).sum() / 2.0
    # Gradient: de/dr * dr/dpos
    # de/dr = 4 * (-12*r^-13 + 6*r^-7) = 24*r^-7 - 48*r^-13
    # In terms of r2: r^-7 = r2^(-3.5), r^-13 = r2^(-6.5)
    # de/dr = 24*r2^(-3.5) - 48*r2^(-6.5)
    # grad_i = sum_{j != i} de/dr_ij * (pos_i - pos_j) / r_ij
    #        = sum_{j != i} (24*r2^(-3.5) - 48*r2^(-6.5)) * diff[i,j,:] / r_ij
    #        = sum_{j != i} (24*r2^(-4) - 48*r2^(-7)) * diff[i,j,:]
    # Because diff/r = diff / sqrt(r2), and (24*r2^(-3.5) - 48*r2^(-6.5)) / sqrt(r2) = 24*r2^(-4) - 48*r2^(-7)
    coeff = 24.0 * r2 ** (-4) - 48.0 * r2 ** (-7)
    np.fill_diagonal(coeff, 0.0)
    grad = 4.0 * np.einsum('ij,ijk->ik', coeff, diff) / 2.0  # divide by 2 for pair counting
    return e, grad

def make_mackay_icosahedron(n_atoms):
    """
    Generate a Mackay icosahedron cluster.
    Shell 0: 1 atom at origin
    Shell 1: 12 atoms at distance r1
    Shell 2: 42 atoms at distance r2
    Total: 55 atoms
    """
    # Icosahedron vertices (unit length)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    # Normalize to unit length
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    
    # Shell 1: 12 atoms at distance r1
    r1 = 1.0  # Will be optimized, but start with reasonable value
    shell1 = vertices * r1
    
    # Shell 2: 42 atoms
    # The 42 positions of the second shell in a Mackay icosahedron are:
    # 12 face centers (distance ~r2_face)
    # 30 edge midpoints (distance ~r2_edge)
    
    # Actually, for a proper Mackay icosahedron with 55 atoms:
    # Shell 0: 1 atom
    # Shell 1: 12 atoms (icosahedron vertices)
    # Shell 2: 42 atoms
    # The 42 atoms in shell 2 are arranged as:
    # - 12 atoms at face centers of the icosahedron
    # - 30 atoms at edge midpoints of the icosahedron
    # Total = 12 + 30 = 42 ✓
    
    # Icosahedron edges: pairs of vertices that are closest neighbors
    # In a regular icosahedron, each vertex connects to 5 others
    # There are 30 edges total
    
    # Let's find edges by distance
    dist_matrix = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dist_matrix[i, j] = np.linalg.norm(vertices[i] - vertices[j])
            dist_matrix[j, i] = dist_matrix[i, j]
    
    # Find the minimum distance (edge length)
    min_dist = np.min(dist_matrix[dist_matrix > 0])
    tol = min_dist * 1.01
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if dist_matrix[i, j] < tol:
                edges.append((i, j))
    
    # Edge midpoints (30 atoms)
    edge_midpoints = []
    for i, j in edges:
        mid = (vertices[i] + vertices[j]) / 2.0
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)
    
    # Face centers (12 atoms)
    # Icosahedron has 20 triangular faces
    # We need to find the faces
    # A face is a triple of mutually adjacent vertices
    
    faces = []
    # Build adjacency
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    # Find all triangles
    for i in range(12):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and i in adj[k]:
                    triangle = tuple(sorted([i, j, k]))
                    if triangle not in faces:
                        faces.append(triangle)
    
    # Deduplicate
    faces = list(set(faces))
    
    # There should be 20 faces
    face_centers = []
    for face in faces:
        center = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) / 3.0
        face_centers.append(center)
    face_centers = np.array(face_centers)
    
    # Normalize edge midpoints and face centers to unit length, then scale
    edge_norms = np.linalg.norm(edge_midpoints, axis=1, keepdims=True)
    face_norms = np.linalg.norm(face_centers, axis=1, keepdims=True)
    
    # The radial distances for shell 2 in Mackay icosahedron:
    # For a complete two-shell Mackay icosahedron, the ratio r2/r1 is typically around 1.5-1.6
    # Let's use r2 = 1.5 * r1 as starting point
    r2 = 1.5
    
    shell2_face = (face_centers / face_norms) * r2
    shell2_edge = (edge_midpoints / edge_norms) * r2
    
    # Combine all atoms
    atoms = [np.zeros(3)]  # Shell 0
    atoms.extend(list(shell1))  # Shell 1
    atoms.extend(list(shell2_face))  # Shell 2 face centers
    atoms.extend(list(shell2_edge))  # Shell 2 edge midpoints
    
    positions = np.array(atoms[:n_atoms])
    
    # Center the cluster
    positions -= np.mean(positions, axis=0)
    
    return positions

def make_cubic_lattice():
    """Generate cubic lattice fragment."""
    side = int(np.ceil(N ** (1/3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side-1)/2)**2 + (p[1] - (side-1)/2)**2 + (p[2] - (side-1)/2)**2)
    return np.array(points[:N], dtype=float) * 1.12

def try_relax(positions, time_limit=60.0):
    """Try to relax positions using L-BFGS. Return best result."""
    start_time = time.time()
    best_energy = lj_energy(positions)
    best_pos = positions.copy()
    
    try:
        result = minimize(
            lj_energy,
            positions,
            jac=lj_energy_grad,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x
        
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > time_limit:
        return best_pos, best_energy
    
    return best_pos, best_energy

def construct():
    """Construct initial positions for Lennard-Jones 55-atom cluster."""
    global N
    start_time = time.time()
    time_limit = 120.0
    margin = 15.0  # Safety margin
    
    # Strategy: Try multiple starting configurations and relax each
    # 1. Mackay icosahedron (best known structure)
    # 2. Cubic lattice (fallback)
    # 3. Random perturbations of Mackay
    
    best_energy = np.inf
    best_pos = None
    
    # Start with Mackay icosahedron
    try:
        mackay = make_mackay_icosahedron(N)
        # Ensure minimum distance constraint
        # Check pairwise distances
        min_dist_ok = True
        n = len(mackay)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(mackay[i] - mackay[j])
                if d < 0.5:
                    min_dist_ok = False
                    break
            if not min_dist_ok:
                break
        
        if min_dist_ok:
            pos, energy = try_relax(mackay, time_limit=40.0)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        else:
            # If Mackay violates constraint, fall back
            pass
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit - margin:
        if best_pos is not None:
            return best_pos
        return make_cubic_lattice()
    
    # Try cubic lattice
    try:
        cubic = make_cubic_lattice()
        pos, energy = try_relax(cubic, time_limit=30.0)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit - margin:
        if best_pos is not None:
            return best_pos
        return make_cubic_lattice()
    
    # Try random perturbations of Mackay
    rng = np.random.default_rng(42)
    try:
        mackay = make_mackay_icosahedron(N)
        for trial in range(5):
            if time.time() - start_time > time_limit - margin - 10:
                break
            
            # Add small random perturbation
            perturbation = rng.normal(0, 0.05, mackay.shape)
            pos_perturbed = mackay + perturbation
            
            # Check minimum distance
            min_dist_ok = True
            n = len(pos_perturbed)
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(pos_perturbed[i] - pos_perturbed[j])
                    if d < 0.5:
                        min_dist_ok = False
                        break
                if not min_dist_ok:
                    break
            
            if min_dist_ok:
                pos, energy = try_relax(pos_perturbed, time_limit=15.0)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos.copy()
    except Exception:
        pass
    
    # Final time check
    elapsed = time.time() - start_time
    if best_pos is not None:
        return best_pos
    return make_cubic_lattice()

# EVOLVE-BLOCK-END
