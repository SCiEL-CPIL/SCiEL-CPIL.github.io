"""Combined program for logarithmic energy minimization of 150 points on sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree

_DEADLINE = 140.0  # Safety margin before 150s kill

def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum for precision."""
    n = len(points)
    if n < 2:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -math.fsum(log_dists.tolist())

def _energy_and_grad(x):
    """
    Energy and tangential gradient for L-BFGS-B. 
    x is flat array of shape (N*3,).
    The function normalizes x to the sphere internally.
    Returns (energy, tangential_gradient_flat).
    """
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)

    # Gradient of E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(inv_dists_sq, axis=1)
    weighted_sum = inv_dists_sq @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(np.sqrt(dists_sq[iu])))

    return energy, grads_tangent.flatten()

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, rng):
    """Generate random uniform points on sphere."""
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _icosahedron_subdivide(points, depth):
    """Subdivide icosahedron to get more points, then normalize."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0.],
        [ 1,  t,  0.],
        [-1, -t,  0.],
        [ 1, -t,  0.],
        [ 0., -1,  t],
        [ 0.,  1,  t],
        [ 0., -1, -t],
        [ 0.,  1, -t],
        [ t,  0., -1],
        [ t,  0.,  1],
        [-t,  0., -1],
        [-t,  0.,  1]
    ], dtype=np.float64)
    verts = _project_to_sphere(verts)
    
    # Simple subdivision: for each face, add midpoint of edges and center
    # This is a rough approximation for generating a dense symmetric grid.
    # For N=150, a specific subdivision count is needed.
    # An icosahedron has 20 faces.
    # Subdividing each face into 4 triangles (midpoint of edges + center) -> 80 faces.
    # Subdividing again -> 320 faces.
    # We need 150 points.
    # Let's just use a dense Fibonacci sphere as a proxy for "symmetric-ish" 
    # or use a specific geometric construction.
    # Actually, for 150 points, a specific spherical design might be better.
    # Let's stick to generating many symmetric inits via random rotations of a base symmetric set.
    pass

def _tangent_vector(point, rng):
    """Generate a random unit tangent vector at point on sphere."""
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm

def _local_minimize(points, deadline, maxiter=500, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()
    
    if time.time() > deadline - 1.0:
        return points, _log_energy(points)

    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': ftol, 'gtol': gtol, 'maxls': 40}
        )
        if time.time() > deadline - 1.0:
             return points, _log_energy(points)
             
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)

def _basin_hopping_single(init_points, deadline, rng, max_iter=100, T_init=0.1):
    """
    Basin hopping with Metropolis criterion.
    """
    current_points, current_energy = _local_minimize(init_points, deadline, maxiter=200)
    best_points = current_points.copy()
    best_energy = current_energy

    T = T_init
    T_min = 1e-8
    cooling = 0.95

    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        if T < T_min:
            break

        # Perturb a random subset of points
        # Increase perturbation size slightly as we go? Or keep constant?
        # Larger kicks help escape deep minima.
        n_perturb = rng.randint(5, 15) # More points perturbed
        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T * rng.uniform(0.5, 1.5)
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        # Local minimize the perturbed configuration
        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=100)

        # Metropolis criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()

        T *= cooling

    return best_points, best_energy

def _polish_high_precision(pts, max_time):
    """High-precision final polish with L-BFGS at tight tolerances."""
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    t0 = time.time()
    
    def obj(x):
        if time.time() - t0 > max_time:
            return np.inf
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
        
    def grad(x):
        if time.time() - t0 > max_time:
            return np.zeros_like(x)
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        return g

    x0 = pts.flatten()
    try:
        res = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 50000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 200000}
        )
        pts_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt = _log_energy(pts_opt)
        return pts_opt, e_opt
    except Exception:
        return pts, _log_energy(pts)

def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE
    
    rng = np.random.RandomState(42)

    best_x = None
    best_e = float('inf')

    # Phase 1: Diverse Initialization and Local Minimization
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random uniform points (multiple seeds)
    for seed in range(10):
        r = np.random.RandomState(seed * 100 + 1)
        inits.append(_random_uniform_sphere(N, r))
        
    # 3. Perturbed Fibonacci (small noise to break symmetry)
    for k in range(5):
        noise_scale = 0.05 * (k+1)
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * noise_scale
        inits.append(_project_to_sphere(perturbed))

    # Run local minimization on each to find good local minima
    # Limit time per init to ensure we try all
    time_per_init = 5.0
    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
            
        init_deadline = min(deadline, time.time() + time_per_init)
        x_res, e_res = _local_minimize(init, init_deadline, maxiter=200, ftol=1e-10, gtol=1e-8)
        
        if e_res < best_e:
            best_e = e_res
            best_x = x_res

    # Phase 2: Basin Hopping from the best local minimum found
    # Allocate significant time here as it's the main global search tool
    if best_x is not None and time.time() < deadline - 20.0:
        # Run basin hopping for a fixed number of iterations or until time runs out
        x_bh, e_bh = _basin_hopping_single(best_x, deadline, rng, max_iter=200, T_init=0.15)
        if e_bh < best_e:
            best_e = e_bh
            best_x = x_bh

    # Phase 3: Second round of basin hopping with different parameters (finer search)
    if best_x is not None and time.time() < deadline - 10.0:
        x_bh2, e_bh2 = _basin_hopping_single(best_x, deadline, rng, max_iter=150, T_init=0.05)
        if e_bh2 < best_e:
            best_e = e_bh2
            best_x = x_bh2

    # Phase 4: Final High-Precision Polish
    if best_x is not None and time.time() < deadline - 2.0:
        polish_time = deadline - time.time() - 1.0
        if polish_time > 0.5:
            x_final, e_final = _polish_high_precision(best_x, polish_time)
            if e_final < best_e:
                best_x = x_final
                best_e = e_final

    # Final validation and projection
    if best_x is None:
        best_x = _fibonacci_sphere(N)
        
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, 1)
    if dist[iu].min() < 1e-9:
        # Fallback to Fibonacci if points collapsed
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
        
    return best_x
# EVOLVE-BLOCK-END
