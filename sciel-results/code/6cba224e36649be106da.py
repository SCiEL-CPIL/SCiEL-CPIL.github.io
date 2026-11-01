"""Improved Lennard-Jones cluster global minimum, 55 atoms."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist

# Deadline guard: 150s wall-clock limit, use 145s for safety
_DEADLINE = 145.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _lj_energy(pos):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    d = pdist(pos)
    d = np.maximum(d, 1e-10)
    r6 = d ** -6
    return 4.0 * np.sum(r6 ** 2 - r6)


def _lj_grad(pos):
    """Compute gradient of LJ energy using vectorized operations."""
    n = len(pos)
    # Compute all pair differences
    # diff[i, j] = pos[i] - pos[j]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # Compute squared distances
    r2 = np.sum(diff ** 2, axis=2)
    # Avoid division by zero
    np.fill_diagonal(r2, 1e10)
    
    # Compute coefficients
    # coeff = 4 * (-12 * r^-12 + 6 * r^-6) / r^2
    # r^-6 = r2^-3
    # r^-12 = r2^-6
    r6 = r2 ** -3
    r12 = r6 ** 2
    coeff = 4.0 * (-12.0 * r12 + 6.0 * r6) / r2
    
    # Gradient is sum over j of coeff[i,j] * diff[i,j]
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    # Using einsum: grad[i, k] = sum_j coeff[i, j] * diff[i, j, k]
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return grad


def _lj_energy_and_grad(pos):
    """Combined energy and gradient for L-BFGS-B."""
    e = _lj_energy(pos)
    g = _lj_grad(pos)
    return e, g.ravel()


def _mackay_icosahedron_55(r1=1.12, r2=2.24):
    """Construct the two-shell Mackay icosahedron for 55 atoms.
    
    Shell 0: 1 center atom
    Shell 1: 12 atoms at icosahedron vertices (radius r1)
    Shell 2: 42 atoms (12 face centers + 30 edge midpoints) at radius r2
    """
    phi = (1 + np.sqrt(5)) / 2.0
    
    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Shell 1: 12 icosahedron vertices at radius r1
    shell1 = verts * r1
    
    # Find edges of icosahedron
    D = cdist(verts, verts)
    np.fill_diagonal(D, np.inf)
    min_dist = D.min()
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_dist) < 1e-10:
                edges.append((i, j))
    
    # Edge midpoint directions (30 edges)
    edge_mids = []
    for i, j in edges:
        mid_dir = (verts[i] + verts[j]) / np.linalg.norm(verts[i] + verts[j])
        edge_mids.append(mid_dir)
    edge_mids = np.array(edge_mids)  # shape (30, 3)
    
    # Face centers of icosahedron (20 faces)
    # We need to find the 20 faces.
    # A face is a triangle of vertices.
    # We can find them by checking triples of vertices that form a triangle.
    # Alternatively, we know the icosahedron has 20 faces.
    # Let's find them by checking all triples.
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_dist) > 1e-10:
                continue
            for k in range(j + 1, 12):
                if abs(D[i, k] - min_dist) > 1e-10:
                    continue
                if abs(D[j, k] - min_dist) > 1e-10:
                    continue
                # Check if i,j,k form a face
                # The normal vector is (v_j - v_i) x (v_k - v_i)
                # If the distance from center to face is consistent, it's a face
                # For a regular icosahedron, all faces are equivalent.
                # We can just collect all such triples.
                faces.append((i, j, k))
    
    # There should be 20 faces.
    face_centers = []
    for i, j, k in faces:
        center_dir = (verts[i] + verts[j] + verts[k]) / np.linalg.norm(verts[i] + verts[j] + verts[k])
        face_centers.append(center_dir)
    face_centers = np.array(face_centers)  # shape (20, 3)
    
    # Mackay outer shell: 12 (vertex directions) + 30 (edge midpoints) = 42
    # Wait, the standard Mackay icosahedron N=55 has:
    # 1 (center)
    # 12 (first shell, vertices)
    # 42 (second shell)
    # The 42 atoms in the second shell are located at:
    # - 12 positions corresponding to the vertices of the inner icosahedron (but further out)
    # - 30 positions corresponding to the edge midpoints of the inner icosahedron (but further out)
    # This matches the 42 points we generated (12 vertex_outer + 30 edge_mids).
    # Note: The face centers are NOT part of the standard Mackay N=55 shell.
    # The Mackay N=55 is a "complete icosahedron" with shells 1, 2, 3? No, N=55 is 1+12+42.
    # The 42 points are indeed the 12 vertex directions and 30 edge midpoint directions.
    
    vertex_outer = verts * r2
    shell2 = np.vstack([vertex_outer, edge_mids * r2])
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    positions = np.vstack([center, shell1, shell2])
    return positions


def _relax(positions, maxiter=500, gtol=1e-10, deadline=None):
    """L-BFGS-B local relaxation."""
    if deadline is None:
        deadline = _DEADLINE
    if time.time() > deadline - 5.0:
        return positions
    
    def fun(x):
        x = x.reshape(-1, 3)
        e = _lj_energy(x)
        g = _lj_grad(x)
        return e, g.ravel()
    
    x0 = positions.reshape(-1)
    try:
        result = minimize(fun, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': gtol})
        return result.x.reshape(-1, 3)
    except Exception:
        return positions


def _polish(positions, deadline=None):
    """High-precision polish pass with progressively tighter tolerances."""
    if deadline is None:
        deadline = _DEADLINE
    
    best_energy = _lj_energy(positions)
    best_pos = positions.copy()
    
    def fun(x):
        x = x.reshape(-1, 3)
        e = _lj_energy(x)
        g = _lj_grad(x)
        return e, g.ravel()
    
    x0 = positions.reshape(-1)
    
    # Multi-stage polish: progressively tighter
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-14},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if time.time() > deadline - 5.0:
            break
        try:
            result = minimize(fun, x0, jac=True, method='L-BFGS-B', options=stage_opts)
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos


def _check_min_dist(pos, min_d=0.5):
    """Check all pair distances >= min_d."""
    d = pdist(pos)
    return d.min() >= min_d - 1e-9


def construct():
    global _start_time
    _start_time = time.time()
    
    best_pos = None
    best_energy = np.inf
    
    # Strategy 1: Start from Mackay icosahedron and relax
    if _remaining() > 10.0:
        try:
            mackay = _mackay_icosahedron_55()
            relaxed = _relax(mackay, maxiter=500, gtol=1e-8)
            e = _lj_energy(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
    
    # Strategy 2: Multi-start with Gaussian perturbations of Mackay
    rng = np.random.default_rng(42)
    mackay_base = _mackay_icosahedron_55()
    
    # Try several noise levels for perturbation
    noise_levels = [0.01, 0.02, 0.05, 0.10]
    
    for noise in noise_levels:
        if _remaining() < 15.0:
            break
        try:
            # Perturb the Mackay structure
            perturbed = mackay_base + rng.normal(0, noise, mackay_base.shape)
            
            # Ensure minimum distance constraint
            if not _check_min_dist(perturbed):
                # Try to fix by scaling
                scale = 1.0
                for _ in range(50):
                    if _check_min_dist(perturbed * scale):
                        break
                    scale *= 1.01
                perturbed = perturbed * scale
            
            relaxed = _relax(perturbed, maxiter=300, gtol=1e-8)
            e = _lj_energy(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            continue
    
    # Final high-precision polish on best result
    if best_pos is not None and _remaining() > 10.0:
        try:
            polished = _polish(best_pos, deadline=_DEADLINE)
            e = _lj_energy(polished)
            if _check_min_dist(polished) and e < best_energy:
                best_energy = e
                best_pos = polished.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: original cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos
# EVOLVE-BLOCK-END
