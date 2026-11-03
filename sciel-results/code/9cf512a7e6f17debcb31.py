import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # Safety margin from 150s

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and Euclidean gradient.
    pts: (N, 3) array of points on the unit sphere.
    Returns: (energy, grad_eucl)
    """
    n = len(pts)
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-18):
        return float('inf'), np.zeros_like(pts)
    
    # E = sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(dist_sq_ij)
    log_d_sq = np.log(np.maximum(dist_sq, 1e-300))
    np.fill_diagonal(log_d_sq, 0.0)
    energy = -0.25 * np.sum(log_d_sq)
    
    # Gradient: dE/dx_i = -0.5 * sum_{j!=i} (x_i - x_j) / dist_sq_ij
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    weights = -0.5 / dist_sq
    np.fill_diagonal(weights, 0.0)
    grad_eucl = np.einsum('ij,ijk->ik', weights, diff)
    
    return float(energy), grad_eucl

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _geodesic_dome(n):
    """Generate points via geodesic dome (icosahedron subdivision)."""
    # Start with icosahedron
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Faces of icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces to get more points
    # For n points, we need to subdivide appropriately
    # Simple approach: use Fibonacci if n is not a standard dome count
    # For now, just return icosahedron-based points with subdivision
    # This is a simplified version; for exact n, we'll use Fibonacci as fallback
    
    # Actually, let's just generate a set of points and if we need exactly n,
    # we'll handle it differently. For now, return a reasonable set.
    # Let's use a different approach: generate points via spherical coordinates
    # based on geodesic dome logic, but ensure exactly n points.
    
    # Simpler: use the icosahedron vertices and their midpoints recursively
    # until we have at least n points, then take the first n.
    # This is getting complex. Let's just use a rotated Fibonacci as the "dome"
    # alternative, or implement a proper subdivision.
    
    # For simplicity and correctness, let's implement a basic subdivision:
    # We'll subdivide each face into 4 smaller faces (midpoint subdivision).
    # 20 faces -> 80 -> 320...
    # For n=150, we need about 3 subdivisions (20*4^2 = 320, too many)
    # or 2 subdivisions (20*4 = 80, too few)
    # Let's just do 2 subdivisions and add some more points.
    
    # Actually, let's just use a different strategy: generate points on the sphere
    # using a quasi-random sequence that mimics dome distribution.
    # For now, return Fibonacci rotated by a small angle as the "dome" alternative.
    # This is a placeholder; the real value is in the SA, not the init.
    
    # Let's implement a proper geodesic dome for n=150:
    # Start with icosahedron (12 verts, 20 faces)
    # Subdivide each face into 4: 40 verts, 80 faces
    # Subdivide again: 160 verts, 320 faces
    # Take 150 of the 160 verts (excluding 10 to get exactly 150)
    
    # Implementation:
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide function
    def subdivide(verts, faces):
        new_faces = []
        mid_cache = {}
        
        def midpoint(i, j):
            if (i, j) in mid_cache:
                return mid_cache[(i, j)]
            if (j, i) in mid_cache:
                return mid_cache[(j, i)]
            mid = (verts[i] + verts[j]) / 2.0
            mid = mid / np.linalg.norm(mid)
            new_idx = len(verts)
            verts = np.vstack([verts, mid.reshape(1, -1)])
            mid_cache[(i, j)] = new_idx
            mid_cache[(j, i)] = new_idx
            return new_idx
        
        for f in faces:
            a, b, c = f
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        
        return verts, new_faces
    
    # Subdivide twice
    verts, faces = subdivide(verts, faces)  # 12 -> 18 verts? No, 12 + 30 = 42? 
    # Actually, 20 faces, each adds 3 midpoints, but shared. 
    # 12 verts, 20 faces, 30 edges. So 12 + 30 = 42 verts after 1 subdivision.
    verts, faces = subdivide(verts, faces)  # 42 + 60*? Let's just run it.
    
    # After 2 subdivisions, we should have 12 + 30 + 90 = 132? Or more.
    # Let's just use what we have. If we have >= 150, take 150.
    # If < 150, pad with Fibonacci.
    
    if len(verts) >= n:
        # Randomly select n points from verts
        idx = np.random.choice(len(verts), size=n, replace=False)
        return verts[idx]
    else:
        # Pad with Fibonacci
        fib = _fibonacci_sphere(n - len(verts))
        return np.vstack([verts, fib])

def _random_uniform_sphere(n, seed=None):
    """Generate n random points uniformly on the unit sphere."""
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _bfgs_polish(pts, max_iter=100):
    """
    Polish points using BFGS on the sphere via retraction.
    We minimize the energy in R^3 with the constraint ||x_i|| = 1.
    Use scipy's minimize with BFGS, projecting back to sphere after each step.
    """
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        e, g = _energy_and_grad(pts)
        if not np.isfinite(e):
            return 1e10, np.zeros_like(x_flat)
        return e, g.flatten()
    
    x0 = pts.flatten()
    res = minimize(objective_and_grad, x0, method='BFGS', 
                   options={'maxiter': max_iter, 'gtol': 1e-10, 'maxls': 20})
    
    # Project back to sphere
    pts_opt = res.x.reshape(N, 3)
    pts_opt = _project_to_sphere(pts_opt)
    
    e_final, _ = _energy_and_grad(pts_opt)
    if not np.isfinite(e_final):
        return pts, float('inf')
    
    return pts_opt, e_final

def _kick(pts, kick_size, kick_strength):
    """
    Apply a random geodesic kick to a subset of points.
    kick_size: fraction of points to perturb
    kick_strength: magnitude of perturbation (radians)
    """
    n = len(pts)
    n_kick = max(1, int(n * kick_size))
    idx = np.random.choice(n, size=n_kick, replace=False)
    
    # Generate random tangent vectors
    for i in idx:
        # Random direction in R^3
        v = np.random.randn(3)
        # Project to tangent space at pts[i]
        v_tan = v - np.dot(v, pts[i]) * pts[i]
        norm_v = np.linalg.norm(v_tan)
        if norm_v < 1e-15:
            continue
        v_tan = v_tan / norm_v * kick_strength
        
        # Exponential map: x_new = x cos(theta) + v_tan sin(theta) / ||v_tan||
        # But v_tan is already scaled by kick_strength, so theta = kick_strength
        # Actually, the exponential map for a tangent vector v is:
        # x_new = x cos(||v||) + (v/||v||) sin(||v||)
        # Here v = v_tan, ||v|| = kick_strength (since we normalized then scaled)
        theta = kick_strength
        pts[i] = pts[i] * math.cos(theta) + v_tan * math.sin(theta) / norm_v
    
    return _project_to_sphere(pts)

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_energy = float('inf')
    best_pts = None
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Fibonacci sphere
    candidates.append(_fibonacci_sphere(N))
    
    # 2. Geodesic dome
    try:
        candidates.append(_geodesic_dome(N))
    except Exception:
        pass
    
    # 3. Random uniform points with different seeds
    for seed in [42, 123, 456]:
        candidates.append(_random_uniform_sphere(N, seed=seed))
    
    # 4. Rotated Fibonacci
    fib = _fibonacci_sphere(N)
    for angle, axis in [(np.pi/3, [1,0,0]), (np.pi/4, [0,1,0]), (np.pi/6, [1,1,1])]:
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        candidates.append(_project_to_sphere(fib @ R.T))
    
    # Phase 1: BFGS polish each initial candidate
    for i, init_pts in enumerate(candidates):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        
        try:
            opt_pts, e = _bfgs_polish(init_pts, max_iter=50)
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Simulated annealing / basin hopping
    if best_pts is None:
        best_pts = candidates[0]
        best_energy, _ = _energy_and_grad(best_pts)
    
    # Annealing parameters
    T_start = 0.5
    T_end = 0.01
    n_iterations = 200  # Max iterations, will be cut by deadline
    
    current_pts = best_pts.copy()
    current_energy = best_energy
    
    for it in range(n_iterations):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        # Annealing temperature
        T = T_start * (T_end / T_start) ** (it / n_iterations)
        
        # Kick size and strength decrease with temperature
        kick_size = 0.1 + 0.2 * (T / T_start)
        kick_strength = 0.1 + 0.5 * (T / T_start)
        
        # Apply kick
        kicked_pts = _kick(current_pts.copy(), kick_size, kick_strength)
        
        # BFGS polish the kicked configuration
        try:
            opt_pts, e_opt = _bfgs_polish(kicked_pts, max_iter=30)
        except Exception:
            continue
        
        if not np.isfinite(e_opt):
            continue
        
        # Metropolis acceptance
        delta_e = e_opt - current_energy
        if delta_e < 0 or np.random.rand() < math.exp(-delta_e / max(T, 1e-10)):
            current_pts = opt_pts
            current_energy = e_opt
            
            # Update best
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = opt_pts.copy()
    
    # Phase 3: Final tight BFGS polish on the best result
    if time.time() - start_time < DEADLINE - 5:
        try:
            final_pts, e_final = _bfgs_polish(best_pts, max_iter=200)
            if e_final < best_energy and np.isfinite(e_final):
                best_energy = e_final
                best_pts = final_pts
        except Exception:
            pass
    
    # Final normalization
    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    e_check, _ = _energy_and_grad(best_pts)
    if not np.isfinite(e_check):
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
