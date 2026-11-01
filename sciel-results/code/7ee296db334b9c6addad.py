"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(X):
    """Compute logarithmic energy and its gradient efficiently."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    
    # Compute pairwise squared distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
    # For points on sphere: = 2 - 2*x_i·x_j
    dot = X @ X.T  # (n, n)
    dists_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dists_sq, 1.0)  # avoid issues
    dists_sq = np.maximum(dists_sq, 1e-30)  # numerical safety
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: grad_i = -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # = -sum_{j!=i} (x_i - x_j) / dists_sq[i,j]
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    # diff[i,j] = x_i - x_j
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    grad = -np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(X):
    """Project points onto unit sphere."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=0):
    """Generate n random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _icosahedron_subdivide(n_target, subdivisions=3):
    """Generate points by subdividing an icosahedron, then subsample to n_target."""
    # Icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Faces (triangles)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces
    all_points = set()
    for v in verts:
        all_points.add(tuple(np.round(v, 10)))
    
    for _ in range(subdivisions):
        new_points = set(all_points)
        for face in faces:
            a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
            ab = (a + b) / 2.0
            bc = (b + c) / 2.0
            ca = (c + a) / 2.0
            ab /= np.linalg.norm(ab)
            bc /= np.linalg.norm(bc)
            ca /= np.linalg.norm(ca)
            new_points.add(tuple(np.round(ab, 10)))
            new_points.add(tuple(np.round(bc, 10)))
            new_points.add(tuple(np.round(ca, 10)))
        all_points = new_points
    
    pts = np.array(list(all_points))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    if len(pts) >= n_target:
        # Subsample to get n_target points
        rng = np.random.RandomState(42)
        idx = rng.choice(len(pts), n_target, replace=False)
        return pts[idx]
    else:
        # Pad with random points if needed
        extra = _random_sphere(n_target - len(pts), seed=123)
        return np.vstack([pts, extra])

def _optimize_single(X0, deadline, maxiter=800):
    """Run L-BFGS-B optimization from a single starting point."""
    if time.time() > deadline - 15.0:
        return X0, _energy_and_grad(X0)[0]
    
    try:
        result = minimize(
            fun=lambda x: _energy_and_grad(x.reshape(N, 3))[0],
            x0=X0.flatten(),
            jac=lambda x: _energy_and_grad(x.reshape(N, 3))[1].flatten(),
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'ftol': 1e-16,
                'gtol': 1e-13,
                'maxfun': maxiter * 2,
                'maxls': 20
            }
        )
        
        X_opt = _project_to_sphere(result.x.reshape(N, 3))
        e_opt, _ = _energy_and_grad(X_opt)
        
        return X_opt, e_opt
    except Exception:
        return X0, _energy_and_grad(X0)[0]

def _polish(X, deadline, maxiter=2000):
    """Final high-precision polish pass."""
    if time.time() > deadline - 10.0:
        return X, _energy_and_grad(X)[0]
    
    try:
        result = minimize(
            fun=lambda x: _energy_and_grad(x.reshape(N, 3))[0],
            x0=X.flatten(),
            jac=lambda x: _energy_and_grad(x.reshape(N, 3))[1].flatten(),
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'ftol': 1e-18,
                'gtol': 1e-15,
                'maxfun': maxiter * 3,
                'maxls': 30
            }
        )
        
        X_polished = _project_to_sphere(result.x.reshape(N, 3))
        e_polished, _ = _energy_and_grad(X_polished)
        
        return X_polished, e_polished
    except Exception:
        return X, _energy_and_grad(X)[0]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_X = None
    best_energy = np.inf
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # 2. Icosahedron subdivision
    initializations.append(_icosahedron_subdivide(N, subdivisions=3))
    
    # 3. Random points with various seeds
    for seed in range(8):
        initializations.append(_random_sphere(N, seed=seed))
    
    # 4. Fibonacci with perturbations
    rng = np.random.RandomState(42)
    for sigma in [0.02, 0.05, 0.1, 0.15, 0.2]:
        base = _fibonacci_sphere(N)
        noise = rng.randn(N, 3) * sigma
        initializations.append(_project_to_sphere(base + noise))
    
    # 5. Icosahedron with perturbations
    for sigma in [0.05, 0.1]:
        base = _icosahedron_subdivide(N, subdivisions=3)
        noise = rng.randn(N, 3) * sigma
        initializations.append(_project_to_sphere(base + noise))
    
    # Optimize from each initialization
    for init_idx, X0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        X_opt, e_opt = _optimize_single(X0, deadline)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_X = X_opt.copy()
    
    # Final polish on the best configuration
    if best_X is not None:
        best_X, best_energy = _polish(best_X, deadline)
        
        # Additional polish if time permits
        if time.time() < deadline - 15.0:
            best_X, e2 = _polish(best_X, deadline, maxiter=3000)
            if e2 < best_energy:
                best_energy = e2
                best_X = best_X.copy()
    
    # Fallback
    if best_X is None:
        best_X = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_X = _project_to_sphere(best_X)
    
    # Check for near-duplicate points and fix
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < 1e-9:
                # Nudge point j away
                direction = best_X[j] - best_X[i]
                if np.linalg.norm(direction) < 1e-12:
                    direction = np.array([1.0, 0.0, 0.0])
                direction /= np.linalg.norm(direction)
                best_X[j] = best_X[j] + 1e-7 * direction
                best_X[j] = best_X[j] / np.linalg.norm(best_X[j])
    
    return best_X
# EVOLVE-BLOCK-END
