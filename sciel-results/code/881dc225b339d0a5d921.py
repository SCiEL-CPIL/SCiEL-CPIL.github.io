"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100
DEADLINE = 145.0  # wall-clock seconds (safety margin before 150s kill)


# EVOLVE-BLOCK-START

def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) and its gradient.
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where grad is (N, 3).
    """
    n = pts.shape[0]
    # Compute pairwise distances
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    d_upper = dist[iu]
    energy = -np.sum(np.log(d_upper))
    
    # Gradient of energy w.r.t. pts[i]:
    # d/dx_i [ -log|xi - xj| ] = -(xi - xj)/|xi - xj|^2 = (xj - xi)/|xi - xj|^2
    # For each i, grad_i = sum_{j != i} (xj - xi) / |xi - xj|^2
    # Note: |xi - xj|^2 = 2 - 2*dot(xi,xj) for unit vectors, but let's use dist^2
    dist2 = dist * dist  # (N, N)
    # grad[i] = sum_j (pts[j] - pts[i]) / dist2[i,j]
    # We can compute this as: for each i, sum over j of (pts[j] - pts[i]) / dist2[i,j]
    # = sum_j pts[j] / dist2[i,j] - pts[i] * sum_j 1/dist2[i,j]
    # But dist2[i,i] = 0, so we need to handle that.
    # Let's use the diff approach:
    # grad[i] = sum_{j != i} (pts[j] - pts[i]) / dist2[i,j]
    # Note: diff[i,j] = pts[i] - pts[j], so pts[j] - pts[i] = -diff[i,j]
    # grad[i] = -sum_{j != i} diff[i,j] / dist2[i,j]
    
    # diff[i,j] / dist2[i,j] = (pts[i] - pts[j]) / |pts[i] - pts[j]|^2
    # Let's compute this matrix
    # inv_dist2 = 1/dist2, with diagonal set to 0
    inv_dist2 = np.zeros_like(dist2)
    mask = ~np.eye(n, dtype=bool)
    inv_dist2[mask] = 1.0 / dist2[mask]
    
    # grad[i] = -sum_j diff[i,j] * inv_dist2[i,j]
    # diff is (N, N, 3), inv_dist2 is (N, N)
    # We want for each i: sum_j diff[i,j] * inv_dist2[i,j]
    # This is a batched dot product
    # grad = -np.einsum('ijk,ij->ik', diff, inv_dist2)
    grad = -np.einsum('ijk,ij->ik', diff, inv_dist2)
    
    return energy, grad


def _fibonacci_sphere(n, seed=0):
    """Generate n points on sphere using Fibonacci spiral."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    # Add small random perturbation to break symmetry
    if seed > 0:
        # Small angular perturbation
        delta_phi = rng.uniform(-0.01, 0.01, size=n)
        delta_z = rng.uniform(-0.001, 0.001, size=n)
        z = np.clip(z + delta_z, -1.0, 1.0)
        phi = phi + delta_phi
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _projected_gradient_descent(pts, max_iter=5000, lr=None, momentum=0.9, tol=1e-10):
    """
    Projected gradient descent on the sphere with momentum.
    Uses the gradient of the energy function.
    """
    n = pts.shape[0]
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    
    energy, grad = _log_energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    if lr is None:
        lr = 0.01
    
    velocity = np.zeros_like(pts)
    
    for it in range(max_iter):
        # Check deadline
        if time.time() > DEADLINE:
            break
        
        energy, grad = _log_energy_and_grad(pts)
        
        if energy < best_energy - 1e-12:
            best_energy = energy
            best_pts = pts.copy()
        
        # Momentum update
        velocity = momentum * velocity - lr * grad
        pts_new = pts + velocity
        pts_new = _project_to_sphere(pts_new)
        
        # Check for convergence
        step_size = np.linalg.norm(pts_new - pts)
        if step_size < tol:
            break
        
        pts = pts_new
    
    # Final check
    energy, _ = _log_energy_and_grad(best_pts)
    return best_pts, energy


def _annealing_schedule(t, t_start=1.0, t_end=0.01, n_steps=1000):
    """Exponential cooling schedule."""
    return t_start * (t_end / t_start) ** (t / n_steps)


def _simulated_annealing(pts, max_iter=3000, t_start=0.1, t_end=0.001):
    """
    Simulated annealing on the sphere.
    Perturb points and accept based on Metropolis criterion.
    """
    n = pts.shape[0]
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    
    energy, _ = _log_energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        t = t_start * (t_end / t_start) ** (it / max_iter)
        
        # Generate random perturbation
        # Perturb in a random direction on the tangent space
        num_perturbed = rng.randint(1, min(5, n))
        idx = rng.choice(n, size=num_perturbed, replace=False)
        
        # Random tangent vectors
        rand_vecs = rng.randn(num_perturbed, 3)
        # Project to tangent space: v - (v·x)x
        x_i = pts[idx]
        dot = np.sum(rand_vecs * x_i, axis=1, keepdims=True)
        tangent_vecs = rand_vecs - dot * x_i
        norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        tangent_vecs = tangent_vecs / norms
        
        # Step size
        step_size = 0.05 * t / t_start + 0.005
        new_pts = pts.copy()
        new_pts[idx] = pts[idx] + step_size * tangent_vecs
        new_pts = _project_to_sphere(new_pts)
        
        new_energy, _ = _log_energy_and_grad(new_pts)
        
        # Metropolis acceptance
        delta_e = new_energy - energy
        if delta_e < 0 or rng.rand() < math.exp(-delta_e / max(t, 1e-10)):
            pts = new_pts
            energy = new_energy
            if energy < best_energy - 1e-12:
                best_energy = energy
                best_pts = pts.copy()
    
    energy, _ = _log_energy_and_grad(best_pts)
    return best_pts, energy


def construct():
    """
    Construct 100 points on the unit sphere to minimize logarithmic energy.
    Uses multiple restarts with projected gradient descent and simulated annealing.
    """
    start_time = time.time()
    
    best_pts = None
    best_energy = np.inf
    
    # Strategy: multiple restarts with different initializations
    # Each restart: initialize, do annealing to escape local minima, then polish with PGD
    
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 5.0:
            break
        
        # Different initializations
        if restart == 0:
            # Fibonacci spiral
            pts = _fibonacci_sphere(N, seed=0)
        elif restart == 1:
            # Random points
            rng = np.random.RandomState(123)
            pts = rng.randn(N, 3)
            pts = _project_to_sphere(pts)
        elif restart == 2:
            # Fibonacci with perturbation
            pts = _fibonacci_sphere(N, seed=7)
        elif restart == 3:
            # Two-cluster initialization
            rng = np.random.RandomState(42)
            center1 = np.array([1.0, 0.0, 0.0])
            center2 = np.array([-1.0, 0.0, 0.0])
            pts = np.zeros((N, 3))
            for i in range(N):
                if i < N // 2:
                    pts[i] = center1 + rng.randn(3) * 0.3
                else:
                    pts[i] = center2 + rng.randn(3) * 0.3
            pts = _project_to_sphere(pts)
        else:
            # Random with seed based on restart
            rng = np.random.RandomState(1000 + restart)
            pts = rng.randn(N, 3)
            pts = _project_to_sphere(pts)
        
        # Phase 1: Simulated annealing to explore
        pts, energy = _simulated_annealing(pts, max_iter=1500, t_start=0.05, t_end=0.0005)
        
        # Phase 2: Projected gradient descent to polish
        pts, energy = _projected_gradient_descent(pts, max_iter=3000, lr=0.005, momentum=0.9, tol=1e-12)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() < DEADLINE - 2.0:
        # Very fine-tuned gradient descent
        best_pts, best_energy = _projected_gradient_descent(
            best_pts, max_iter=10000, lr=0.001, momentum=0.8, tol=1e-14
        )
    
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N, seed=0)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Check for any points too close
    # If any pair is closer than 1e-9, nudge them apart
    for _ in range(10):
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        iu = np.triu_indices(N, k=1)
        min_dist = np.min(dist[iu])
        if min_dist > 1e-9:
            break
        # Find the closest pair and nudge
        d_upper = dist[iu]
        min_idx = np.argmin(d_upper)
        i, j = iu[0][min_idx], iu[1][min_idx]
        # Nudge point j slightly
        rng = np.random.RandomState(0)
        v = rng.randn(3)
        v = v - np.dot(v, best_pts[j]) * best_pts[j]
        v = v / np.linalg.norm(v)
        best_pts[j] = best_pts[j] + 1e-8 * v
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
