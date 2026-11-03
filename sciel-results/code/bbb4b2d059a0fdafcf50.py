import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150
DEADLINE = 117.0  # Safety margin for 150s limit

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def _random_sphere(n, seed=0):
    """Generate n random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)

def _geodesic_dome_init(n, subdivisions=2):
    """Generate points from a subdivided icosahedron (geodesic dome)."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    for _ in range(subdivisions):
        new_faces = []
        mid_cache = {}
        for face in faces:
            a, b, c = face
            mids = []
            for i, j in [(a, b), (b, c), (c, a)]:
                key = (min(i, j), max(i, j))
                if key not in mid_cache:
                    mid = (verts[i] + verts[j])
                    mid = _project_to_sphere(mid.reshape(1, 3))[0]
                    mid_cache[key] = len(verts)
                    verts = np.vstack([verts, mid.reshape(1, 3)])
                mids.append(mid_cache[key])
            new_faces.append([a, mids[0], mids[2]])
            new_faces.append([b, mids[1], mids[0]])
            new_faces.append([c, mids[2], mids[1]])
            new_faces.append([mids[0], mids[1], mids[2]])
        faces = new_faces
    
    # We now have 20 * 4^subdivisions vertices
    # Sample n points from these vertices
    n_verts = len(verts)
    if n_verts >= n:
        rng = np.random.RandomState(42)
        indices = rng.choice(n_verts, size=n, replace=False)
        return verts[indices]
    else:
        # If not enough vertices, just return them all plus random
        pts = list(verts)
        while len(pts) < n:
            pts.append(_project_to_sphere(np.array([np.random.randn(3)]))[0])
        return np.array(pts[:n])

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere."""
    pts = _project_to_sphere(pts)
    n = len(pts)
    
    # Compute pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, 1.0)
    
    # Clip distances to avoid log(0)
    dist_safe = np.clip(dist, 1e-10, 2.0)
    
    # Energy: sum over i<j of log(1/d_ij)
    # = -0.5 * sum_{i!=j} log(d_ij)  ... wait, sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # Using full matrix: sum_{i!=j} log(1/d_ij) = 2 * sum_{i<j} log(1/d_ij)
    # So energy = 0.5 * sum_{i!=j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    
    # Actually let's compute directly
    triu_idx = np.triu_indices(n, k=1)
    energy = np.sum(np.log(1.0 / dist_safe[triu_idx]))
    
    # Gradient: dE/dx_i = sum_{j!=i} d/dx_i log(1/|x_i - x_j|)
    # = sum_{j!=i} -(x_i - x_j) / |x_i - x_j|^2
    # Project to tangent space: grad_tangent = grad - (grad . x_i) x_i
    
    grad = np.zeros_like(pts)
    for i in range(n):
        # Force from all other points on point i
        diffs = pts[i] - pts  # shape (n, 3)
        diffs[i] = 0  # self
        dists = np.linalg.norm(diffs, axis=1, keepdims=True)
        dists[i] = 1.0  # avoid div by zero
        # Force: -(x_i - x_j) / d^2
        forces = -diffs / (dists ** 2)
        forces[i] = 0
        grad[i] = np.sum(forces, axis=0)
        # Project to tangent
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return energy, grad

def _energy_only(pts):
    """Compute just the energy."""
    energy, _ = _energy_and_grad(pts)
    return energy

def _adam_optimize_sphere(pts, n_steps=200, lr=0.01):
    """Adam optimization on the sphere."""
    pts = _project_to_sphere(pts.copy())
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    for t in range(1, n_steps + 1):
        energy, grad = _energy_and_grad(pts)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        pts -= update
        pts = _project_to_sphere(pts)
    
    return pts

def _lbfgs_polish(pts, max_iter=50):
    """Polish with L-BFGS-B on the tangent space."""
    def objective(flat_pts):
        pts = _project_to_sphere(flat_pts.reshape(-1, 3))
        energy, grad = _energy_and_grad(pts)
        # For L-BFGS, we need gradient in the flat representation
        # Since we project after each step, we use the projected gradient
        return energy, grad.reshape(-1)
    
    result = minimize(objective, pts.reshape(-1), method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'maxfun': 200})
    return _project_to_sphere(result.x.reshape(-1, 3))

def _basin_hopping(pts, n_restarts=5, max_iter=20):
    """Simple basin hopping: perturb and re-optimize."""
    best = pts.copy()
    best_e = _energy_only(best)
    
    for _ in range(n_restarts):
        # Perturb
        perturbed = pts + 0.05 * np.random.randn(*pts.shape)
        perturbed = _project_to_sphere(perturbed)
        
        # Quick optimization
        optimized = _adam_optimize_sphere(perturbed, n_steps=50, lr=0.005)
        e = _energy_only(optimized)
        
        if e < best_e:
            best_e = e
            best = optimized
    
    return best

def construct():
    start_time = time.time()
    
    best_points = None
    best_energy = np.inf
    
    # Generate diverse initializations
    inits = []
    inits.append(("fib", _fibonacci_sphere(N)))
    inits.append(("dome2", _geodesic_dome_init(N, subdivisions=2)))
    inits.append(("dome3", _geodesic_dome_init(N, subdivisions=3)))
    for i in range(8):
        inits.append((f"rand{i}", _random_sphere(N, seed=i * 7 + 13)))
    
    # Phase 1: Adam optimization from each init
    for name, init in inits:
        if time.time() > DEADLINE - 10:
            break
        
        # Quick check
        if time.time() > DEADLINE - 3:
            break
        
        optimized = _adam_optimize_sphere(init, n_steps=100, lr=0.01)
        energy = _energy_only(optimized)
        
        if energy < best_energy:
            best_energy = energy
            best_points = optimized.copy()
    
    # Phase 2: L-BFGS polish on best
    if best_points is not None and time.time() < DEADLINE - 8:
        polished = _lbfgs_polish(best_points, max_iter=30)
        energy = _energy_only(polished)
        if energy < best_energy:
            best_energy = energy
            best_points = polished.copy()
    
    # Phase 3: Basin hopping if time allows
    if best_points is not None and time.time() < DEADLINE - 15:
        bh_result = _basin_hopping(best_points, n_restarts=3, max_iter=15)
        energy = _energy_only(bh_result)
        if energy < best_energy:
            best_energy = energy
            best_points = bh_result.copy()
    
    # Final polish
    if best_points is not None and time.time() < DEADLINE - 5:
        final = _lbfgs_polish(best_points, max_iter=20)
        energy = _energy_only(final)
        if energy < best_energy:
            best_energy = energy
            best_points = final.copy()
    
    # Ensure validity
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    # Final normalization and validity check
    best_points = _project_to_sphere(best_points)
    
    # Check for minimum distance
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            min_dist = min(min_dist, d)
    
    if min_dist < 1e-9:
        # Points too close, fallback to Fibonacci
        best_points = _fibonacci_sphere(N)
    
    # Final check: all finite
    if not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_sphere(N)
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    e = _energy_only(pts)
    print(f"Energy: {e}")
