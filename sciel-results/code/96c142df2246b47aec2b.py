"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
from math import log, fsum
import time

# EVOLVE-BLOCK-START
import numpy as np
from math import log, fsum
import time

N = 150
DEADLINE_FRACTION = 0.85  # safety margin

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e30  # penalty for coincident points
            terms.append(-log(d))
    return fsum(terms)

def _log_energy_vec(pts):
    """Vectorized log energy for faster evaluation during optimization."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)  # avoid log(0)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Clip to avoid log(0) or log(negative) due to numerical errors
    d = np.clip(d, 1e-15, None)
    return np.sum(-np.log(d))

def _log_energy_grad(pts):
    """Compute gradient of log energy."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    
    # Gradient of -log(d) is -1/d * dd
    # dE/dxi = sum_{j!=i} (xi - xj) / |xi - xj|^2
    # Note: d(|xi-xj|)/dxi = (xi-xj)/|xi-xj|
    # d(-log|xi-xj|)/dxi = -1/|xi-xj| * (xi-xj)/|xi-xj| = -(xi-xj)/|xi-xj|^2
    # Wait, let's re-verify.
    # E = sum -log(d_ij)
    # dE/dxi = sum_{j!=i} - (1/d_ij) * d(d_ij)/dxi
    # d(d_ij)/dxi = (xi - xj) / d_ij
    # So dE/dxi = sum_{j!=i} - (xi - xj) / d_ij^2
    # This is equivalent to sum_{j!=i} (xj - xi) / d_ij^2
    
    # We want the gradient of the energy to minimize.
    # Let's compute the force F_i = -grad E_i.
    # Actually, let's just compute grad E.
    
    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-15
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    
    # diff is xi - xj
    # grad_i = sum_j -diff_ij * inv_d2_ij
    # So grad = - sum_j diff_ij * inv_d2_ij
    grads = -np.sum(diff * inv_d2[:, np.newaxis, :], axis=1)
    
    # Project gradient to tangent space: grad - (grad . p) * p
    for i in range(N):
        g = grads[i]
        p = pts[i]
        grads[i] = g - np.dot(g, p) * p
        
    return grads

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_rotation_matrix(rng):
    """Generate a random 3x3 rotation matrix."""
    # Using QR decomposition of a random matrix
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = 1)
    d = np.sign(np.linalg.det(Q))
    Q[:, 0] *= d
    return Q

def _simulated_annealing(pts, time_budget, rng):
    """
    Run simulated annealing with periodic symmetry breaking.
    """
    best_pts = _normalize(pts.copy())
    best_energy = _log_energy(best_pts)
    current_energy = best_energy
    
    # Annealing parameters
    T = 0.1
    cooling_rate = 0.9999
    step_size = 0.01
    
    start_time = time.time()
    steps = 0
    stall_counter = 0
    
    while time.time() < time_budget:
        steps += 1
        
        # Compute gradient
        grad = _log_energy_grad(best_pts)
        
        # Gradient descent step with noise
        noise = rng.standard_normal((N, 3)) * T * step_size
        # Project noise to tangent space
        for i in range(N):
            n_vec = noise[i]
            p_vec = best_pts[i]
            noise[i] = n_vec - np.dot(n_vec, p_vec) * p_vec
        
        new_pts = best_pts - step_size * grad + noise
        
        # Normalize to sphere
        new_pts = _normalize(new_pts)
        
        # Check validity (no coincident points)
        if np.any(np.linalg.norm(new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :], axis=2) < 1e-9):
            # If points are too close, reject
            continue
            
        new_energy = _log_energy_vec(new_pts)
        
        # Metropolis criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            best_pts = new_pts
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                stall_counter = 0
            else:
                stall_counter += 1
        else:
            stall_counter += 1
            
        # Cool down
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6
            
        # Periodic symmetry breaking (random rotation) to escape local minima
        if steps % 500 == 0 and stall_counter > 100:
            R = _random_rotation_matrix(rng)
            rotated_pts = best_pts @ R.T
            # This doesn't change energy, but might help if we combine with other perturbations?
            # Actually, rotating the whole set doesn't change the relative distances, so energy is invariant.
            # We need a perturbation that breaks the symmetry of the *local* minimum structure.
            # Let's do a small random displacement instead of global rotation for escape.
            # Or, we can try a "kick":
            kick = rng.standard_normal((N, 3)) * 0.05
            for i in range(N):
                k = kick[i]
                p = best_pts[i]
                kick[i] = k - np.dot(k, p) * p
            kicked_pts = _normalize(best_pts + kick)
            
            # If the kick improves energy, accept it. Otherwise, we might still accept it with probability exp(-dE/T)
            kicked_energy = _log_energy_vec(kicked_pts)
            dE = kicked_energy - current_energy
            if dE < 0 or rng.random() < np.exp(-dE / T):
                best_pts = kicked_pts
                current_energy = kicked_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                stall_counter = 0
                T = max(T * 1.5, 0.01) # Reheat slightly
        
    return best_pts, best_energy

def construct():
    start_time = time.time()
    total_time = 150.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_spiral(N)
    best_energy = _log_energy(best_pts)
    
    # Run simulated annealing
    pts, energy = _simulated_annealing(best_pts, deadline, rng)
    if energy < best_energy:
        best_pts = pts
        best_energy = energy
        
    if time.time() >= deadline:
        return best_pts
        
    # If time remains, try a second round from a different start
    # e.g. random points
    random_pts = _normalize(rng.standard_normal((N, 3)))
    pts2, energy2 = _simulated_annealing(random_pts, deadline, rng)
    if energy2 < best_energy:
        best_pts = pts2
        best_energy = energy2
        
    return best_pts
# EVOLVE-BLOCK-END
