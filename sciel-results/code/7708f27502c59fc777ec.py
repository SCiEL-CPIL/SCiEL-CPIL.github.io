"""Lennard-Jones 55-atom cluster: parameterized Mackay icosahedron with aggressive optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

def _lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def _lj_energy_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    coeff = 24.0 * r2 ** (-4) - 48.0 * r2 ** (-7)
    np.fill_diagonal(coeff, 0.0)
    grad = 4.0 * np.einsum('ij,ijk->ik', coeff, diff) / 2.0
    return e, grad

def _get_icosahedron_geometry():
    """
    Return the icosahedron geometry:
    - vertices: 12 unit-length vertex directions
    - edges: list of 30 edge index pairs
    - faces: list of 20 face index triples
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    
    # Find edges
    dist_matrix = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    min_dist = np.min(dist_matrix[dist_matrix > 0])
    tol = min_dist * 1.01
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if dist_matrix[i, j] < tol:
                edges.append((i, j))
    
    # Find faces (triangles)
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    faces = set()
    for i in range(12):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and i in adj[k]:
                    faces.add(tuple(sorted([i, j, k])))
    faces = sorted(faces)
    
    return vertices, edges, faces

def _construct_from_params(params):
    """
    Construct 55-atom Mackay icosahedron from parameters.
    params: [r1, r2_face, r2_edge]
    Returns (N, 3) array of positions.
    """
    r1, r2_face, r2_edge = params
    vertices, edges, faces = _get_icosahedron_geometry()
    
    positions = [np.zeros(3)]  # Shell 0: center
    
    # Shell 1: 12 atoms at icosahedron vertices scaled by r1
    for v in vertices:
        positions.append(v * r1)
    
    # Shell 2: 42 atoms
    # Face centers (20 faces)
    for face in faces:
        center = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) / 3.0
        norm = np.linalg.norm(center)
        if norm > 1e-10:
            positions.append((center / norm) * r2_face)
    
    # Edge midpoints (30 edges)
    for i, j in edges:
        mid = (vertices[i] + vertices[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-10:
            positions.append((mid / norm) * r2_edge)
    
    pos = np.array(positions[:N])
    return pos

def _objective(params):
    """Objective function for parameter optimization."""
    pos = _construct_from_params(params)
    return _lj_energy(pos)

def _relax_lbfgs(positions, time_limit=40.0):
    """Relax positions using L-BFGS with time limit."""
    start = time.time()
    best_energy = _lj_energy(positions)
    best_pos = positions.copy()
    
    try:
        result = minimize(
            _lj_energy,
            positions,
            jac=_lj_energy_grad,
            method='L-BFGS-B',
            options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.copy()
    except Exception:
        pass
    
    return best_pos, best_energy

def _polish(positions, time_limit=20.0):
    """Multi-stage polish with progressively tighter tolerances."""
    start = time.time()
    best_pos = positions.copy()
    best_energy = _lj_energy(positions)
    
    stages = [
        {'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8},
        {'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10},
        {'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12},
    ]
    
    for opts in stages:
        if time.time() - start > time_limit:
            break
        try:
            result = minimize(
                _lj_energy,
                best_pos,
                jac=_lj_energy_grad,
                method='L-BFGS-B',
                options=opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.copy()
        except Exception:
            break
    
    return best_pos, best_energy

def construct():
    """Construct initial positions for Lennard-Jones 55-atom cluster."""
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    best_energy = np.inf
    best_pos = None
    
    # Strategy 1: Optimize Mackay icosahedron parameters
    try:
        # Initial guess: r1 ~ 1.0, r2_face ~ 1.5, r2_edge ~ 1.5
        x0 = np.array([1.0, 1.5, 1.5])
        
        # Optimize parameters
        param_result = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if param_result.fun < best_energy:
            best_energy = param_result.fun
            best_params = param_result.x
            best_pos = _construct_from_params(best_params).copy()
        
        # Now relax the full 55-atom structure
        if time.time() - start_time < deadline - 30:
            pos, energy = _relax_lbfgs(best_pos, time_limit=40.0)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
                
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > deadline - 20:
        if best_pos is not None:
            return best_pos
        # Fallback: basic Mackay
        vertices, edges, faces = _get_icosahedron_geometry()
        positions = [np.zeros(3)]
        for v in vertices:
            positions.append(v * 1.0)
        for face in faces:
            center = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) / 3.0
            norm = np.linalg.norm(center)
            if norm > 1e-10:
                positions.append((center / norm) * 1.5)
        for i, j in edges:
            mid = (vertices[i] + vertices[j]) / 2.0
            norm = np.linalg.norm(mid)
            if norm > 1e-10:
                positions.append((mid / norm) * 1.5)
        return np.array(positions[:N])
    
    # Strategy 2: Polish the result
    if best_pos is not None:
        pos, energy = _polish(best_pos, time_limit=20.0)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    # Strategy 3: Try different initial parameter values
    if time.time() - start_time < deadline - 20:
        alt_params_list = [
            [0.95, 1.45, 1.45],
            [1.05, 1.55, 1.55],
            [1.0, 1.4, 1.6],
            [1.0, 1.6, 1.4],
        ]
        
        for alt_params in alt_params_list:
            if time.time() - start_time > deadline - 15:
                break
            try:
                pos = _construct_from_params(alt_params)
                # Quick relax
                pos_relaxed, energy = _relax_lbfgs(pos, time_limit=15.0)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos_relaxed.copy()
            except Exception:
                continue
    
    # Final polish if time permits
    if best_pos is not None and time.time() - start_time < deadline - 5:
        pos, energy = _polish(best_pos, time_limit=10.0)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    if best_pos is not None:
        return best_pos
    
    # Final fallback
    vertices, edges, faces = _get_icosahedron_geometry()
    positions = [np.zeros(3)]
    for v in vertices:
        positions.append(v * 1.0)
    for face in faces:
        center = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) / 3.0
        norm = np.linalg.norm(center)
        if norm > 1e-10:
            positions.append((center / norm) * 1.5)
    for i, j in edges:
        mid = (vertices[i] + vertices[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-10:
            positions.append((mid / norm) * 1.5)
    return np.array(positions[:N])

# EVOLVE-BLOCK-END
