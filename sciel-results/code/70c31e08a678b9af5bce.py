import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 150
DEADLINE = 118.0  # Safety margin from 120s

def _log_energy(pts):
    """Compute the logarithmic energy of points on the sphere using math.fsum for accuracy."""
    n = len(pts)
    # Compute pairwise distances
    # dist_sq[i,j] = ||x_i - x_j||^2
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 0.0)
    
    # Avoid log(0) - though points should be distinct
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    # d_ij = sqrt(dist_sq_ij)
    # Energy = sum_{i<j} log(1/d_ij) = sum_{i<j} -log(d_ij) = -0.5 * sum_{i<j} log(dist_sq_ij)
    
    # Get upper triangle indices
    i_upper = np.triu_indices(n, 1)
    log_d_sq = np.log(dist_sq[i_upper])
    
    # Use math.fsum for higher precision as specified in task
    energy = -0.5 * math.fsum(log_d_sq.tolist())
    
    return energy

def _log_energy_and_grad(pts):
    """Compute logarithmic energy and its gradient."""
    n = len(pts)
    # Compute pairwise distances
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 0.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    # Energy
    i_upper = np.triu_indices(n, 1)
    log_d_sq = np.log(dist_sq[i_upper])
    energy = -0.5 * math.fsum(log_d_sq.tolist())
    
    # Gradient: grad_i = -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # Let's compute efficiently
    # For each pair (i,j), the contribution to grad_i is -(x_i - x_j)/dist_sq_ij
    # And to grad_j is +(x_i - x_j)/dist_sq_ij
    
    grad = np.zeros_like(pts)
    
    # Vectorized computation
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    
    # dist_sq is (n, n)
    # We need to divide diff by dist_sq[..., np.newaxis]
    # But we need to handle diagonal (where dist_sq = 0)
    
    # Create mask for off-diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Safe division
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # grad_i = -sum_j (x_i - x_j) / dist_sq_ij
    # = -sum_j diff[i,j] * inv_dist_sq[i,j]
    # Using matrix multiplication:
    # grad = -diff @ inv_dist_sq[..., np.newaxis]? No.
    # sum_j diff[i,j] * inv_dist_sq[i,j]
    # This is element-wise multiplication then sum over j
    
    # Let's do it differently
    # term[i,j] = diff[i,j] * inv_dist_sq[i,j, None]
    # grad[i] = -sum_j term[i,j]
    
    # To avoid large memory, let's loop or use batched operations
    # Actually n=150, so (150, 150, 3) is fine
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    # Project gradient to tangent space of sphere
    # grad_tangent_i = grad_i - (grad_i . x_i) x_i
    proj = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - proj * pts
    
    return energy, grad

def _subdivided_icosahedron(subdivisions=2):
    """Create a subdivided icosahedron on the unit sphere."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    
    # Normalize
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 20 faces of icosahedron
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    # Subdivide
    for _ in range(subdivisions):
        new_faces = []
        edge_midpoint_cache = {}
        
        for face in faces:
            a, b, c = face
            # Find midpoints of edges
            def get_midpoint(p, q):
                key = (min(p, q), max(p, q))
                if key in edge_midpoint_cache:
                    return edge_midpoint_cache[key]
                mid = (verts[p] + verts[q]) / 2
                mid = mid / np.linalg.norm(mid)
                new_idx = len(verts)
                verts = np.vstack([verts, mid])
                edge_midpoint_cache[key] = new_idx
                return new_idx
            
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            
            new_faces.append((a, ab, ca))
            new_faces.append((b, bc, ab))
            new_faces.append((c, ca, bc))
            new_faces.append((ab, bc, ca))
        
        faces = new_faces
    
    return verts

def _fibonacci_sphere(n):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed=42):
    """Generate n random points uniformly distributed on the sphere."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = np.arccos(2 * rng.uniform(0, 1, n) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _riemannian_l_bfgs(pts, max_iter=200):
    """
    Optimize points on the sphere using Riemannian L-BFGS.
    Uses retraction to stay on the sphere.
    """
    current_pts = pts.copy()
    
    # L-BFGS memory
    m = 10
    history = []
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        # Normalize to sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, 1e-15)
        e, g = _log_energy_and_grad(x)
        return e, g
    
    # Use scipy's minimize with L-BFGS-B, but we need to handle the sphere constraint
    # We'll use a retraction-based approach manually
    
    current_energy, current_grad = objective_and_grad(current_pts.flatten())
    
    for k in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
            
        # L-BFGS two-loop recursion to compute search direction
        if len(history) == 0:
            direction = -current_grad
        else:
            # Two-loop recursion
            r = -current_grad
            alphas = []
            betas = []
            
            # First loop
            for y, g in reversed(history):
                sigma = np.dot(y, g)
                alpha = np.dot(y, r) / sigma
                alphas.append(alpha)
                r = r - alpha * y
            betas.append(1.0)
            
            # Second loop
            for i, (y, g) in enumerate(history):
                sigma = np.dot(y, g)
                beta = np.dot(g, r) / sigma
                alphas.reverse()
                r = r + (alphas[i] - beta) * y
                betas.append(beta)
            
            direction = r
        
        # Line search (simple backtracking)
        step_size = 0.01
        accepted = False
        
        for ls_iter in range(20):
            # Retraction: move in direction and project back to sphere
            new_pts = current_pts + step_size * direction
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / np.maximum(norms, 1e-15)
            
            new_energy, new_grad = objective_and_grad(new_pts.flatten())
            
            if new_energy < current_energy:
                # Update history
                if len(history) > 0:
                    y = new_grad - current_grad
                    g = direction
                    history.append((y, g))
                    if len(history) > m:
                        history.pop(0)
                
                current_pts = new_pts
                current_energy = new_energy
                current_grad = new_grad
                accepted = True
                break
            else:
                step_size *= 0.5
        
        if not accepted:
            break
            
        # Check for convergence
        if np.linalg.norm(current_grad) < 1e-10:
            break
    
    return current_pts, current_energy

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_energy = float('inf')
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Fibonacci sphere
    fib = _fibonacci_sphere(N)
    inits.append(fib)
    
    # 2. Subdivided icosahedron (closest to 150 points)
    # icosahedron has 12 vertices, each subdivision multiplies faces by 4
    # Level 0: 12 verts, 20 faces
    # Level 1: 42 verts, 80 faces  
    # Level 2: 162 verts, 320 faces - close to 150!
    ico = _subdivided_icosahedron(2)
    # We have 162 points, take first 150
    ico_150 = ico[:N]
    inits.append(ico_150)
    
    # 3. Random with different seeds
    for seed in [42, 123, 456]:
        rand = _random_uniform_sphere(N, seed)
        inits.append(rand)
    
    # Optimize each initialization
    for idx, init in enumerate(inits):
        if time.time() > DEADLINE - 20:
            break
            
        try:
            # Run Riemannian L-BFGS
            opt_pts, e = _riemannian_l_bfgs(init, max_iter=300)
            
            # Verify energy
            e_check, _ = _log_energy_and_grad(opt_pts)
            
            if e_check < best_energy:
                best_energy = e_check
                best_pts = opt_pts.copy()
                
        except Exception as ex:
            continue
    
    # If we have time, try more optimizations from the best result
    if best_pts is not None and time.time() < DEADLINE - 10:
        try:
            # Try additional L-BFGS runs with different initial step sizes
            for trial in range(3):
                if time.time() > DEADLINE - 5:
                    break
                
                # Add small random perturbation
                noise = np.random.normal(0, 0.005, (N, 3))
                perturbed = best_pts + noise
                perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
                
                opt_pts, e = _riemannian_l_bfgs(perturbed, max_iter=200)
                e_check, _ = _log_energy_and_grad(opt_pts)
                
                if e_check < best_energy:
                    best_energy = e_check
                    best_pts = opt_pts.copy()
                    
        except Exception:
            pass
    
    # Final polish with very small steps if time permits
    if best_pts is not None and time.time() < DEADLINE - 3:
        try:
            opt_pts, e = _riemannian_l_bfgs(best_pts, max_iter=100)
            e_check, _ = _log_energy_and_grad(opt_pts)
            
            if e_check < best_energy:
                best_energy = e_check
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = fib
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    # Verify final result
    final_energy = _log_energy(best_pts)
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    e, g = _log_energy_and_grad(pts)
    print(f"Energy: {e}")
    print(f"Gradient norm: {np.linalg.norm(g)}")
    print(f"Number of points: {len(pts)}")
