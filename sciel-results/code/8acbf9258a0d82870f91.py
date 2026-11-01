"""Lennard-Jones cluster global minimum, 55 atoms: Mackay icosahedron search."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 55

# Time budget: evaluator kills at 150s wall-clock. Use 140s as deadline.
DEADLINE = 140.0

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for positions (N,3)."""
    n = len(pos)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r = np.sqrt(r2)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    r2_u = r2[iu]
    r6_u = r2_u ** (-3)
    r12_u = r6_u ** 2
    energy = 4.0 * np.sum(r12_u - r6_u)
    return energy

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    # Energy
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    # Zero out diagonal
    r6[np.diag_indices(n)] = 0
    r12[np.diag_indices(n)] = 0
    energy = 4.0 * np.sum(r12 - r6)
    
    # Gradient: dE/dxi = 4 * sum_j ( -12/r^14 + 6/r^8 ) * (xi - xj)
    # = 24 * sum_j ( -2/r^14 + 1/r^8 ) * (xi - xj)
    # Note: (xi - xj) = diff[i,j]
    r8 = r6 * r2 ** (-1)  # r^(-8)
    r14 = r12 * r2 ** (-1)  # r^(-14)
    # factor[i,j] = 24 * (-2*r^(-14) + r^(-8))
    factor = 24.0 * (-2.0 * r14 + r8)
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _relax(pos, maxiter=2000):
    """L-BFGS-B relaxation."""
    def func_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.reshape(-1)
    
    result = minimize(func_and_grad, pos.flatten(), method='L-BFGS-B',
                      jac=True, options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10})
    return result.x.reshape(-1, 3), result.fun

def _generate_mackay_icosahedron():
    """Generate a two-shell Mackay icosahedron for 55 atoms.
    
    Shell 1: 12 vertices of an icosahedron (radius r1)
    Shell 2: 42 atoms on the second shell
    
    For Mackay icosahedra, the shell radii follow specific ratios.
    For n=2 (two shells):
    - Inner shell: 12 atoms at icosahedral vertices, radius ~1.0
    - Outer shell: 42 atoms, radius ~2.0
    
    Actually, for the Mackay icosahedron with 55 atoms:
    - Center: 1 atom
    - First shell: 12 atoms (icosahedron vertices)
    - Second shell: 42 atoms
    
    Let me use the known structure. The icosahedron has 12 vertices.
    The second shell of a Mackay icosahedron (n=2) has 42 atoms.
    
    Standard coordinates for icosahedron vertices (golden ratio φ = (1+√5)/2):
    (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    
    For the two-shell Mackay icosahedron, I'll generate approximate positions
    and then relax them.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Icosahedron vertices (12 atoms) - first shell
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    # For Mackay icosahedron n=2, the shell radii are approximately:
    # r1 ≈ 1.0, r2 ≈ 2.0 (in units where sigma=1, but we need to scale)
    # The LJ minimum is at r=2^(1/6) ≈ 1.1225
    # Let's use r1 = 1.0 * 2^(1/6) and r2 = 2.0 * 2^(1/6) as starting points
    
    scale = 2.0 ** (1.0/6.0)
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # First shell: 12 atoms at icosahedron vertices
    shell1 = ico_vertices * scale
    
    # Second shell: 42 atoms
    # For Mackay icosahedron, the second shell consists of:
    # - 30 atoms at edge midpoints of the icosahedron (scaled to r2)
    # - 12 atoms at face centers of the icosahedron (scaled to r2)
    
    # Icosahedron edges: each vertex connects to 5 others
    # Find edges by checking which vertex pairs are closest
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dists[i,j] = np.linalg.norm(ico_vertices[i] - ico_vertices[j])
            dists[j,i] = dists[i,j]
    
    # Edge length for icosahedron with circumradius 1 is 2/sqrt(phi+1/sqrt(5))... 
    # Actually, let's find the minimum distance (edge length)
    edge_len = np.min(dists[~np.eye(12, dtype=bool)])
    
    # Find all edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i,j] - edge_len) < 1e-10:
                edges.append((i, j))
    
    # 30 edge midpoints
    edge_midpoints = []
    for i, j in edges:
        mid = (ico_vertices[i] + ico_vertices[j]) / 2.0
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)
    
    # 12 face centers: each face is a triangle of 3 mutually connected vertices
    # Find triangles
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                if (abs(dists[i,j] - edge_len) < 1e-10 and 
                    abs(dists[j,k] - edge_len) < 1e-10 and 
                    abs(dists[i,k] - edge_len) < 1e-10):
                    faces.append((i, j, k))
    
    # There should be 20 faces
    face_centers = []
    for i, j, k in faces:
        fc = (ico_vertices[i] + ico_vertices[j] + ico_vertices[k]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # Scale second shell to radius r2
    # The Mackay icosahedron n=2 has outer shell at radius ~2*r1
    # But we need to be careful about the exact ratio
    r2 = 2.0 * scale
    
    # Normalize and scale edge midpoints and face centers to r2
    edge_midpoints_norm = edge_midpoints / np.linalg.norm(edge_midpoints, axis=1, keepdims=True)
    face_centers_norm = face_centers / np.linalg.norm(face_centers, axis=1, keepdims=True)
    
    shell2 = np.vstack([edge_midpoints_norm * r2, face_centers_norm * r2])
    
    # Total: 1 + 12 + 30 + 12 = 55
    positions = np.vstack([center, shell1, shell2])
    
    return positions

def _generate_random_positions(seed=None):
    """Generate random positions with minimum distance constraint."""
    rng = np.random.RandomState(seed)
    positions = np.zeros((N, 3))
    positions[0] = [0, 0, 0]
    
    min_dist = 0.5
    max_radius = 8.0
    
    for i in range(1, N):
        attempts = 0
        while attempts < 10000:
            pos = rng.uniform(-max_radius, max_radius, size=3)
            dists = np.linalg.norm(positions[:i] - pos, axis=1)
            if np.min(dists) >= min_dist:
                positions[i] = pos
                break
            attempts += 1
        else:
            # If can't find, place at random position (may violate constraint)
            positions[i] = pos
    
    return positions

def _generate_cubic_lattice():
    """Generate cubic lattice fragment."""
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(points[:N], dtype=float) * 1.12

def construct():
    """Main entry point. Returns array of shape (55, 3)."""
    start_time = time.time()
    
    best_pos = None
    best_energy = np.inf
    
    # Strategy 1: Start from Mackay icosahedron template and relax
    try:
        mackay = _generate_mackay_icosahedron()
        # Try several perturbations of the Mackay structure
        for trial in range(5):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            if trial == 0:
                pos = mackay.copy()
            else:
                rng = np.random.RandomState(42 + trial)
                pos = mackay + rng.normal(0, 0.05, size=mackay.shape)
            
            # Relax
            relaxed, energy = _relax(pos, maxiter=3000)
            
            # Check minimum distance constraint
            if _check_min_dist(relaxed, 0.5):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed.copy()
    except Exception:
        pass
    
    # Strategy 2: Random starts with L-BFGS relaxation
    if time.time() - start_time < DEADLINE - 20:
        n_random = 10
        for i in range(n_random):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            seed = 100 + i
            pos = _generate_random_positions(seed=seed)
            relaxed, energy = _relax(pos, maxiter=2000)
            
            if _check_min_dist(relaxed, 0.5):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed.copy()
    
    # Strategy 3: Cubic lattice (fallback from seed)
    if time.time() - start_time < DEADLINE - 10:
        cubic = _generate_cubic_lattice()
        relaxed, energy = _relax(cubic, maxiter=2000)
        
        if _check_min_dist(relaxed, 0.5):
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
    
    # If we have a result, do a final high-precision polish
    if best_pos is not None and time.time() - start_time < DEADLINE - 5:
        try:
            polished, polished_energy = _relax(best_pos, maxiter=5000)
            if _check_min_dist(polished, 0.5) and polished_energy < best_energy:
                best_pos = polished
                best_energy = polished_energy
        except Exception:
            pass
    
    # Fallback: return cubic lattice if nothing else worked
    if best_pos is None:
        best_pos = _generate_cubic_lattice()
    
    # Final validation: ensure minimum distance
    if not _check_min_dist(best_pos, 0.5):
        # Try to fix by spreading out
        best_pos = _fix_min_dist(best_pos)
    
    return best_pos

def _check_min_dist(pos, min_d):
    """Check if all pairwise distances are >= min_d."""
    n = len(pos)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pos[i] - pos[j]) < min_d:
                return False
    return True

def _fix_min_dist(pos, min_d=0.5):
    """Attempt to fix minimum distance violations by spreading atoms."""
    pos = pos.copy()
    for iteration in range(100):
        violated = False
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    violated = True
                    # Push apart
                    if d > 1e-10:
                        direction = (pos[i] - pos[j]) / d
                        push = (min_d - d) / 2.0 + 0.01
                        pos[i] += direction * push
                        pos[j] -= direction * push
                    else:
                        pos[i] += np.array([0.01, 0, 0])
                        pos[j] -= np.array([0.01, 0, 0])
        if not violated:
            break
    return pos
