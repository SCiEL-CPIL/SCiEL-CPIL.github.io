"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, L_BFGS_B
from scipy.special import lpmv, sph_harm  # may not be needed, using manual
from scipy.integrate import quad

def _energy(pts):
    """Compute logarithmic energy E = sum_{i<j} log(1/|xi-xj|)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    dist2 = dist * dist
    np.fill_diagonal(dist2, 1.0)
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _tangent_grad(pts, grad):
    """Project gradient onto tangent space of the sphere."""
    # Tangent projection: g_tan = g - (g . x) x
    return grad - np.sum(grad * pts, axis=1, keepdims=True) * pts

def _retract(pts, grad_tangent, step):
    """Retract tangent step onto the sphere using exponential map approximation."""
    # Simple: pts + step * grad_tangent, then normalize
    new_pts = pts + step * grad_tangent
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return new_pts / norms

def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate n random points on sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_subdivide(n_target):
    """Start from icosahedron and subdivide to get ~n_target points."""
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Faces of icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide each face into 4 smaller triangles
    def midpoint_normalize(a, b):
        m = (a + b) / 2.0
        return m / np.linalg.norm(m)
    
    new_faces = []
    for f in faces:
        a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
        ab = midpoint_normalize(a, b)
        bc = midpoint_normalize(b, c)
        ca = midpoint_normalize(c, a)
        new_faces.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
    
    # Extract unique vertices
    all_verts = []
    for f in new_faces:
        for v in f:
            all_verts.append(v)
    all_verts = np.array(all_verts)
    # Remove duplicates
    unique_verts = []
    for v in all_verts:
        if not any(np.linalg.norm(v - u) < 1e-10 for u in unique_verts):
            unique_verts.append(v)
    unique_verts = np.array(unique_verts)
    
    # If we have more than n_target, we can just return the first n_target
    # For now, return as is (80 points from icosahedron subdivision)
    if len(unique_verts) >= n_target:
        return unique_verts[:n_target]
    else:
        # Pad with Fibonacci if needed
        remaining = n_target - len(unique_verts)
        fib = _fibonacci_sphere(remaining)
        return np.vstack([unique_verts, fib])

def _riemannian_descent(pts, max_iter=2000, tol=1e-12, deadline=None):
    """Riemannian gradient descent with backtracking line search."""
    start_time = time.time()
    best_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    
    step = 0.1
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        grad_tan = _tangent_grad(pts, grad)
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        alpha = step
        for _ in range(30):
            new_pts = _retract(pts, grad_tan, alpha)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy:
                pts = new_pts
                break
            alpha *= 0.5
        else:
            break
        
        # Adaptive step
        step = min(step * 1.05, 0.5)
    
    return best_pts, best_energy

def _polish_lbfgs(pts, deadline=None):
    """Polish using L-BFGS-B with penalty method for sphere constraint."""
    start_time = time.time()
    
    def objective(x):
        if deadline is not None and time.time() > deadline:
            return np.inf
        pts = x.reshape(N, 3)
        # Penalty for not being on sphere
        norms = np.linalg.norm(pts, axis=1)
        penalty = 1e6 * np.sum((norms - 1.0)**2)
        energy, _ = _energy_and_grad(pts)
        return energy + penalty
    
    def jac(x):
        if deadline is not None and time.time() > deadline:
            return np.zeros(N*3)
        pts = x.reshape(N, 3)
        energy, grad = _energy_and_grad(pts)
        # Penalty gradient
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        penalty_grad = 2e6 * (norms - 1.0) * pts / norms
        return (grad + penalty_grad).flatten()
    
    result = minimize(objective, pts.flatten(), jac=jac, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
    
    if result.success or result.fun < _energy(pts):
        new_pts = result.x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_pts = new_pts / norms
        return new_pts
    return pts

def _newton_polish(pts, max_iter=50, deadline=None):
    """Newton-like polish using approximate Hessian."""
    start_time = time.time()
    best_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        grad_tan = _tangent_grad(pts, grad)
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-13:
            break
        
        # Approximate Newton step using diagonal Hessian approximation
        # H_ii ≈ sum_{j != i} 1/dist(i,j)^2 + (sum_j 1/dist(i,j)^2) * I_tangent
        # Simplified: use step = grad_tan / (grad_norm * something)
        step_size = 0.01 * (1.0 + 0.5 * math.exp(-it / 10.0))
        new_pts = _retract(pts, grad_tan, -step_size * grad_tan / (grad_norm + 1e-15))
        
        new_energy, _ = _energy_and_grad(new_pts)
        if new_energy < energy:
            pts = new_pts
        else:
            # Try smaller step
            new_pts = _retract(pts, grad_tan, -step_size * 0.5 * grad_tan / (grad_norm + 1e-15))
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy:
                pts = new_pts
            else:
                break
    
    return best_pts, best_energy

def _perturb_escape(pts, n_perturb=5, deadline=None):
    """Try to escape local minimum by perturbing and re-optimizing."""
    start_time = time.time()
    rng = np.random.default_rng(42)
    best_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    
    for _ in range(n_perturb):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb a random subset of points
        n_pert = np.random.randint(3, 10)
        idx = rng.choice(N, size=n_pert, replace=False)
        perturbed = pts.copy()
        for i in idx:
            # Small random rotation
            theta = rng.uniform(0.1, 0.5)
            axis = rng.standard_normal(3)
            axis = axis / np.linalg.norm(axis)
            # Rodrigues rotation
            c, s = np.cos(theta), np.sin(theta)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + s * K + (1 - c) * (K @ K)
            perturbed[i] = R @ perturbed[i]
        
        # Re-optimize
        new_pts, new_energy = _riemannian_descent(perturbed, max_iter=500, deadline=deadline)
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = np.random.default_rng(12345)
    
    # Multiple initial configurations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # 2. Random
    initializations.append(_random_sphere(N, rng))
    
    # 3. Icosahedron subdivision
    initializations.append(_icosahedron_subdivide(N))
    
    # 4. Fibonacci with offset
    i = np.arange(N) + 0.3
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 0.7
    r = np.sqrt(1.0 - z * z)
    initializations.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    best_pts = None
    best_energy = np.inf
    
    for init_pts in initializations:
        if time.time() > deadline - 20:
            break
        
        # Riemannian descent
        pts, energy = _riemannian_descent(init_pts, max_iter=1500, deadline=deadline - 30)
        
        # L-BFGS polish
        if time.time() < deadline - 15:
            pts = _polish_lbfgs(pts, deadline=deadline - 10)
        
        # Newton polish
        if time.time() < deadline - 5:
            pts, energy = _newton_polish(pts, max_iter=30, deadline=deadline - 2)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Perturbation escape from best
    if best_pts is not None and time.time() < deadline - 10:
        best_pts, best_energy = _perturb_escape(best_pts, n_perturb=8, deadline=deadline - 5)
        
        # Final polish
        if time.time() < deadline - 3:
            best_pts = _polish_lbfgs(best_pts, deadline=deadline - 1)
            best_pts, best_energy = _newton_polish(best_pts, max_iter=20, deadline=deadline)
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
