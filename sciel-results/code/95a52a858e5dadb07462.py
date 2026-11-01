"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 14
D = 3

# EVOLVE-BLOCK-START

def _pairwise_dists(P):
    """Compute all pairwise squared distances."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    return np.sum(diff * diff, axis=2)

def _score(P):
    """Compute the score: (max pairwise dist / min pairwise dist)^2."""
    D2 = _pairwise_dists(P)
    np.fill_diagonal(D2, 0)
    mask = ~np.eye(N, dtype=bool)
    D2 = D2[mask]
    if np.min(D2) < 1e-15:
        return 1e15
    return (np.max(D2) / np.min(D2)) ** 2

def _objective(x):
    """Objective: minimize (max d^2 / min d^2)^2 = (max d / min d)^2."""
    P = x.reshape(N, D)
    D2 = _pairwise_dists(P)
    np.fill_diagonal(D2, 0)
    mask = ~np.eye(N, dtype=bool)
    D2 = D2[mask]
    d_min = np.min(D2)
    d_max = np.max(D2)
    if d_min < 1e-15:
        return 1e15
    return (d_max / d_min) ** 2

def _grad_objective(x):
    """Gradient of (max d^2 / min d^2)^2 with respect to x."""
    P = x.reshape(N, D)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    D2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(D2, 0)
    
    # Find min and max pairs
    mask = ~np.eye(N, dtype=bool)
    D2_masked = D2.copy()
    D2_masked[~mask] = np.inf
    min_idx = np.unravel_index(np.argmin(D2_masked), D2.shape)
    D2_masked[~mask] = 0
    max_idx = np.unravel_index(np.argmax(D2_masked), D2.shape)
    
    d_min = D2[min_idx]
    d_max = D2[max_idx]
    
    if d_min < 1e-15:
        return np.zeros_like(x)
    
    # f = (d_max / d_min)^2
    # df/dd_max = 2 * d_max / d_min^2
    # df/dd_min = -2 * d_max^2 / d_min^3
    # dd^2/dP_i = 2*(P_i - P_j) for the pair (i,j)
    
    grad = np.zeros_like(x)
    
    # Gradient contribution from max pair
    i, j = max_idx
    coeff_max = 2.0 * d_max / (d_min ** 2)
    grad[i] += coeff_max * 2 * (P[i] - P[j])
    grad[j] += coeff_max * 2 * (P[j] - P[i])
    
    # Gradient contribution from min pair
    i, j = min_idx
    coeff_min = -2.0 * d_max ** 2 / (d_min ** 3)
    grad[i] += coeff_min * 2 * (P[i] - P[j])
    grad[j] += coeff_min * 2 * (P[j] - P[i])
    
    return grad

def _refine(P0, max_iter=200, tol=1e-12):
    """Refine a point configuration using L-BFGS-B."""
    x0 = P0.flatten()
    result = minimize(
        _objective,
        x0,
        jac=_grad_objective,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxls': 50}
    )
    return result.x.reshape(N, D)

def _center(P):
    """Center the point cloud at origin."""
    return P - np.mean(P, axis=0)

def _scale(P, target_min=1.0):
    """Scale so that the minimum pairwise distance is approximately target_min."""
    D2 = _pairwise_dists(P)
    np.fill_diagonal(D2, 0)
    mask = ~np.eye(N, dtype=bool)
    d_min = np.sqrt(np.min(D2[mask]))
    if d_min > 1e-15:
        return P * (target_min / d_min)
    return P

def _icosahedron():
    """Generate 12 vertices of an icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1)
    vertices = vertices / norms[:, np.newaxis]
    return vertices

def _tetrahedron():
    """Generate 4 vertices of a regular tetrahedron on unit sphere."""
    v = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1)
    return v / norms[:, np.newaxis]

def _octahedron():
    """Generate 6 vertices of a regular octahedron on unit sphere."""
    v = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    return v

def _cube():
    """Generate 8 vertices of a cube on unit sphere."""
    v = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]], dtype=float)
    norms = np.linalg.norm(v, axis=1)
    return v / norms[:, np.newaxis]

def _random_on_sphere(n, seed):
    """Generate n random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1)
    return pts / norms[:, np.newaxis]

def _combine_configs():
    """Create diverse initial configurations by combining known symmetric shapes."""
    configs = []
    
    # Start with icosahedron (12 points) + 2 more
    ico = _icosahedron()
    for extra_seed in range(5):
        extra = _random_on_sphere(2, seed=1000 + extra_seed)
        P = np.vstack([ico, extra])
        P = _center(P)
        P = _scale(P)
        configs.append(P)
    
    # Cube (8) + tetrahedron (4) + 2 random
    cube = _cube()
    tetra = _tetahedron()
    for extra_seed in range(5):
        extra = _random_on_sphere(2, seed=2000 + extra_seed)
        P = np.vstack([cube, tetra, extra])
        P = _center(P)
        P = _scale(P)
        configs.append(P)
    
    # Octahedron (6) + 8 random
    octa = _octahedron()
    for extra_seed in range(5):
        extra = _random_on_sphere(8, seed=3000 + extra_seed)
        P = np.vstack([octa, extra])
        P = _center(P)
        P = _scale(P)
        configs.append(P)
    
    # Pure random configurations
    for seed in range(10):
        P = _random_on_sphere(N, seed=5000 + seed)
        P = _center(P)
        P = _scale(P)
        configs.append(P)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_P = None
    best_score = 1e15
    
    # Generate diverse initial configurations
    configs = _combine_configs()
    
    for i, P0 in enumerate(configs):
        if time.time() > deadline - 5.0:
            break
        
        try:
            P_refined = _refine(P0, max_iter=300)
            P_centered = _center(P_refined)
            
            # Check for distinct points
            D2 = _pairwise_dists(P_centered)
            np.fill_diagonal(D2, 0)
            mask = ~np.eye(N, dtype=bool)
            if np.min(D2[mask]) < 1e-10:
                continue
            
            s = _score(P_centered)
            if s < best_score:
                best_score = s
                best_P = P_centered.copy()
        except Exception:
            continue
    
    # Final polish with higher precision if time permits
    if best_P is not None and time.time() < deadline - 10.0:
        try:
            P_polished = _refine(best_P, max_iter=500, tol=1e-14)
            P_polished = _center(P_polished)
            s = _score(P_polished)
            if s < best_score:
                best_score = s
                best_P = P_polished
        except Exception:
            pass
    
    if best_P is None:
        # Fallback: return icosahedron + 2 random points
        ico = _icosahedron()
        extra = _random_on_sphere(2, seed=42)
        best_P = np.vstack([ico, extra])
        best_P = _center(best_P)
        best_P = _scale(best_P)
    
    return best_P

# EVOLVE-BLOCK-END
