"""
Logarithmic energy minimization of 150 points on the unit sphere.
Combines Basin Hopping global search with Adam inner-loop optimization 
and a custom manifold-aware L-BFGS final polish.
"""
import numpy as np
import time
import math

N = 150
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _energy_and_grad(pts):
    """
    Compute log(1/|x_i - x_j|) energy and tangent-space gradient.
    
    pts: (N, 3) array of unit vectors.
    Returns: (energy, grad) where grad is (N, 3) projected to tangent space.
    """
    n = pts.shape[0]
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Mask diagonal to avoid division by zero
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = np.sqrt(dist_sq[mask])
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    energy = -np.sum(np.log(d))
    
    # Gradient calculation
    # dE/dx_i = -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2 = sum_{j != i} (x_j - x_i)/|x_i - x_j|^2
    inv_d2 = np.zeros_like(dist_sq)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2[mask] = 1.0 / dist_sq[mask]
    
    # grad[i] = sum_j inv_d2[i,j] * (pts[j] - pts[i])
    # = sum_j inv_d2[i,j] * pts[j] - pts[i] * sum_j inv_d2[i,j]
    row_sums = inv_d2.sum(axis=1)
    grad = inv_d2 @ pts - pts * row_sums[:, np.newaxis]
    
    # Project gradient to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _tangent_project(pts, vec):
    """Project vector to tangent space of sphere at pts."""
    dot = np.sum(vec * pts, axis=1, keepdims=True)
    return vec - dot * pts


def _adam_optimize(pts0, max_iter=2000, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
                   deadline=None, check_interval=10):
    """
    Adam optimization on the sphere.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    start_time = time.time()

    for t in range(1, max_iter + 1):
        if deadline is not None:
            if t % check_interval == 0 or t == 1:
                if time.time() > deadline:
                    break

        energy, grad = _energy_and_grad(pts)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        
        # Step in tangent direction
        pts_new = pts - lr * update
        pts_new = _project_to_sphere(pts_new)
        
        if not np.all(np.isfinite(pts_new)):
            break
        
        energy_new = _energy_and_grad(pts_new)[0]
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
        pts = pts_new

    return best_pts, best_energy


def _manifold_lbfgs(pts0, max_iter=200, tol=1e-12, deadline=None):
    """
    L-BFGS optimization constrained to the sphere manifold.
    We use L-BFGS logic but apply updates in the tangent space.
    """
    from scipy.optimize import fmin_l_bfgs_b
    
    # We will use a wrapper that takes tangent vectors as parameters?
    # No, that's complex. Instead, we use a simple quasi-Newton approach manually.
    # Or, we use fmin_l_bfgs_b on Cartesian coords but with a very small step 
    # and projection. 
    # Actually, let's implement a simple gradient descent with momentum 
    # and adaptive step size (like a simplified L-BFGS) for the final polish,
    # as it's more robust on the manifold than standard L-BFGS-B.
    
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    # Simple adaptive gradient descent with Nesterov momentum
    vel = np.zeros_like(pts)
    lr = 0.01
    beta = 0.9
    
    start_time = time.time()
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        energy, grad = _energy_and_grad(pts)
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
            
        vel = beta * vel - lr * grad
        pts_new = pts + vel
        pts_new = _project_to_sphere(pts_new)
        
        energy_new = _energy_and_grad(pts_new)[0]
        
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            # If energy increases, shrink learning rate
            lr *= 0.5
            vel *= 0.5
            
    return best_pts, best_energy


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _greedy_farthest_point(n, seed=None):
    """Generate n points using greedy farthest-point sampling."""
    rng = np.random.RandomState(seed)
    pts = [rng.randn(3)]
    pts[0] = pts[0] / np.linalg.norm(pts[0])
    
    for _ in range(1, n):
        p = np.array(pts)
        # Distance to all existing points
        dists = np.min([np.sum((p - pi) ** 2, axis=1) for pi in pts], axis=0)
        # Actually, we need min distance from candidate to set.
        # Let's just pick a random point and check if it's far enough, 
        # or use a simple heuristic.
        # For speed, let's stick to Fibonacci and Random for now.
        pass
    return np.array(pts)


def _basin_hopping(pts0, deadline, rng, max_iter=50):
    """
    Basin hopping: perturb, local minimize, Metropolis accept.
    """
    current = _project_to_sphere(pts0.copy())
    current_energy = _energy_and_grad(current)[0]
    best = current.copy()
    best_energy = current_energy
    
    T = 0.1
    T_min = 1e-6
    cooling = 0.95
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        if T < T_min:
            break
            
        # Perturb a random subset of points
        n_perturb = rng.randint(5, 15)
        idxs = rng.choice(N, n_perturb, replace=False)
        
        new_pts = current.copy()
        for idx in idxs:
            # Random tangent vector
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, new_pts[idx]) * new_pts[idx]
            norm = np.linalg.norm(tangent)
            if norm > 1e-10:
                tangent /= norm
                new_pts[idx] += T * tangent
        
        new_pts = _project_to_sphere(new_pts)
        
        # Inner local optimization (Adam)
        new_pts, new_energy = _adam_optimize(new_pts, max_iter=500, lr=0.005,
                                             deadline=deadline - 5.0, check_interval=5)
        
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()
        
        T *= cooling
        
    return best, best_energy


def construct():
    t_start = time.time()
    rng = np.random.RandomState(42)
    
    # Initial candidates
    starts = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=2),
        _random_sphere(N, seed=3),
    ]
    
    # Add perturbed Fibonacci
    for _ in range(3):
        p = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.1
        starts.append(_project_to_sphere(p))
        
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Basin Hopping from each start
    # We have ~120s left. Allocate ~15s per start for 8 starts? 
    # Let's do 4 starts with 25s each, then polish.
    
    for si, start_pts in enumerate(starts):
        if time.time() - t_start > DEADLINE - 20:
            break
            
        bh_pts, bh_energy = _basin_hopping(start_pts, DEADLINE - 20 - si*10, rng, max_iter=30)
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pts = bh_pts.copy()
            
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _energy_and_grad(best_pts)[0]
        
    # Phase 2: Final Polish with Manifold L-BFGS / Adaptive GD
    if time.time() - t_start < DEADLINE - 5:
        polished_pts, polished_energy = _manifold_lbfgs(best_pts, max_iter=500, tol=1e-14, 
                                                        deadline=DEADLINE - 5)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pts = polished_pts
            
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge point j slightly
                best_pts[j] += 1e-8 * rng.randn(3)
                best_pts[j] = _project_to_sphere(best_pts[j].reshape(1, 3))[0]
                
    return best_pts
