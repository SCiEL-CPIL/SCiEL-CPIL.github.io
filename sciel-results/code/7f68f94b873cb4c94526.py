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
    # Using matrix operations for efficiency:
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    
    # sum_weights[i] = sum_j inv_dists_sq[i,j]
    sum_weights = np.sum(inv_dists_sq, axis=1)
    # weighted_sum[i] = sum_j inv_dists_sq[i,j] x_j
    weighted_sum = inv_dists_sq @ points
    
    # Raw Euclidean gradient
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    # Compute energy for the optimizer
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
    t0 = time.time()
    
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

def _compute_knn_graph(points, k=10):
    """
    Compute k-nearest neighbor graph using scipy's cKDTree.
    Returns a sparse matrix L (Laplacian) and the edge list.
    """
    from scipy.sparse import coo_matrix
    tree = cKDTree(points)
    dists, indices = tree.query(points, k=k+1)
    
    # Build edge list (undirected)
    edges = []
    for i in range(N):
        for j in indices[i, 1:]:  # Skip self
            if i < j:
                edges.append((i, j))
    
    # Build Laplacian
    num_edges = len(edges)
    rows = np.zeros(2 * num_edges, dtype=int)
    cols = np.zeros(2 * num_edges, dtype=int)
    data = np.zeros(2 * num_edges, dtype=float)
    
    for idx, (i, j) in enumerate(edges):
        rows[idx] = i
        cols[idx] = j
        data[idx] = -1.0
        
        rows[num_edges + idx] = j
        cols[num_edges + idx] = i
        data[num_edges + idx] = -1.0
    
    # Diagonal is degree
    degrees = np.zeros(N, dtype=int)
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    
    # Add diagonal
    rows = np.concatenate([rows, np.arange(N)])
    cols = np.concatenate([cols, np.arange(N)])
    data = np.concatenate([data, degrees.astype(float)])
    
    L = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsc()
    
    return L, edges

def _laplacian_flow_step(points, L, edges, step_size=0.01, deadline=None):
    """
    One step of Riemannian Laplacian flow.
    The force on point i is the sum of (p_j - p_i) for neighbors j, 
    weighted by the edge weights (here 1.0).
    We then project this force to the tangent space and take a step.
    """
    n = len(points)
    # Compute forces: f_i = sum_j L_ij p_j
    # Since L is the graph Laplacian, L @ p is the standard graph signal processing operation.
    # However, we want the force in the ambient space.
    # L is defined on the graph. The Laplacian flow on the manifold is slightly different,
    # but for small steps, the Euclidean Laplacian projected to the tangent space works well.
    
    # Actually, the standard gradient of the edge length energy is:
    # grad_i E_edges = sum_{j in N(i)} (p_i - p_j) / |p_i - p_j|
    # Our L is just the combinatorial Laplacian (weights 1).
    # Let's compute the forces explicitly for clarity and correctness.
    
    forces = np.zeros_like(points)
    for i, j in edges:
        d = points[i] - points[j]
        dist = np.linalg.norm(d)
        if dist > 1e-12:
            force = d / dist
            forces[i] += force
            forces[j] -= force
    
    # Project forces to tangent space
    dot_products = np.sum(forces * points, axis=1, keepdims=True)
    forces_tangent = forces - dot_products * points
    
    # Take step
    new_points = points + step_size * forces_tangent
    
    # Project to sphere
    new_points = _project_to_sphere(new_points)
    
    return new_points

def _basin_hopping_single(init_points, deadline, rng, max_iter=50, T_init=0.1):
    """
    Basin hopping with Laplacian flow perturbation option.
    """
    current_points, current_energy = _local_minimize(init_points, deadline, maxiter=200)
    best_points = current_points.copy()
    best_energy = current_energy

    T = T_init
    T_min = 1e-8
    cooling = 0.95

    # Pre-compute graph for perturbation? No, graph changes.
    # We'll compute it on the fly if needed, but that's expensive.
    # Let's stick to random tangent perturbations but maybe add a "structural" perturbation
    # that swaps two points or moves a cluster?
    
    # Alternative: Use the KNN graph to identify "clusters" and move them together?
    # Or use the Laplacian flow as a deterministic "relaxation" step between random kicks?
    
    # Let's try: In each iteration, do a random kick, then a few steps of Laplacian flow, then local minimize.
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        if T < T_min:
            break

        # Perturb a random subset of points
        n_perturb = rng.randint(2, 8)
        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        # Optional: Run a few steps of Laplacian flow to relax topological defects
        # This helps maintain uniformity of the point distribution
        if it % 5 == 0:  # Do it every 5th iteration to save time
            try:
                L, edges = _compute_knn_graph(new_points, k=6)
                for _ in range(5):
                    if time.time() > deadline - 2.0:
                        break
                    new_points = _laplacian_flow_step(new_points, L, edges, step_size=0.05, deadline=deadline)
            except Exception:
                pass

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
            options={'maxiter': 20000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 100000}
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
    
    # 2. Random uniform points
    for seed in range(5):
        r = np.random.RandomState(seed * 100 + 1)
        inits.append(_random_uniform_sphere(N, r))
        
    # 3. Perturbed Fibonacci
    for k in range(3):
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.1
        inits.append(_project_to_sphere(perturbed))
        
    # 4. Run Laplacian flow on a random start to create a "relaxed" uniform init
    try:
        rand_init = _random_uniform_sphere(N, np.random.RandomState(99))
        L, edges = _compute_knn_graph(rand_init, k=10)
        relaxed_init = rand_init.copy()
        for _ in range(50):
            relaxed_init = _laplacian_flow_step(relaxed_init, L, edges, step_size=0.1)
        # Re-compute graph as it changes significantly
        L, edges = _compute_knn_graph(relaxed_init, k=10)
        for _ in range(20):
            relaxed_init = _laplacian_flow_step(relaxed_init, L, edges, step_size=0.05)
        inits.append(relaxed_init)
    except Exception:
        pass

    # Run local minimization on each to find good local minima
    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
            
        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        x_res, e_res = _local_minimize(init, deadline, maxiter=300, ftol=1e-10, gtol=1e-8)
        
        if e_res < best_e:
            best_e = e_res
            best_x = x_res

    # Phase 2: Basin Hopping from the best local minimum found
    if best_x is not None and time.time() < deadline - 15.0:
        x_bh, e_bh = _basin_hopping_single(best_x, deadline, rng, max_iter=150, T_init=0.08)
        if e_bh < best_e:
            best_e = e_bh
            best_x = x_bh

    # Phase 3: Second round of basin hopping with different parameters
    if best_x is not None and time.time() < deadline - 10.0:
        # Use a smaller temperature for finer search
        x_bh2, e_bh2 = _basin_hopping_single(best_x, deadline, rng, max_iter=100, T_init=0.02)
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
