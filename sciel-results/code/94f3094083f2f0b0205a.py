"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 150
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi-xj|)
    and gradient w.r.t. each point on the sphere.
    
    For a pair (i,j) with distance d = |xi-xj|:
      energy contribution: log(1/d) = -log(d)
      gradient contribution to xi: d/dd (-log(d)) * dxi/dd = (-1/d) * (xi-xj)/d = -(xi-xj)/d^2
      gradient contribution to xj: +(xi-xj)/d^2
    """
    # Pairwise differences
    # pts shape (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid self-pairs and zero distances
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of -log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    e = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, sum over j != i of -(xi - xj) / d_ij^2
    # grad_i = -sum_{j!=i} (xi - xj) / d_ij^2
    # Note: (xi - xj) / d_ij^2 = diff[i,j] / dist[i,j]^2
    inv_d2 = 1.0 / (dist * dist)  # (N, N)
    # Zero out diagonal
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_d2[i,j]
    # diff[i,j] = xi - xj
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return e, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _fibonacci_spiral(n):
    """Generate n points on sphere using golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _simulate_anneal(pts, time_budget):
    """
    Simulated annealing on the sphere.
    Moves: random small perturbations, projected back to sphere.
    Acceptance: Metropolis criterion with cooling schedule.
    """
    n = len(pts)
    current_energy, _ = _log_energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    # Temperature schedule: start high, cool exponentially
    T0 = 1.0
    T_min = 1e-8
    # Number of steps to do within budget
    # Estimate time per step
    t_start = time.time()
    
    # Precompute some constants
    rng = np.random.default_rng(42)
    
    # We'll do iterations in batches
    batch_size = 500  # steps per batch
    step_size_base = 0.1  # initial step size
    
    while time.time() - t_start < time_budget:
        T = max(T_min, T0 * np.exp(-0.001 * 0))  # We'll adjust cooling per iteration
        # Actually, let's use a linear cooling schedule based on iteration count
        # Total iterations estimate: budget / time_per_step
        # Let's just use a simple geometric cooling
        
        for _ in range(batch_size):
            if time.time() - t_start >= time_budget:
                break
            
            # Pick a random point to move
            idx = rng.integers(0, n)
            
            # Random perturbation in tangent space
            # Generate random vector, project to tangent space at pts[idx]
            rand_vec = rng.standard_normal(3)
            tangent = rand_vec - np.dot(rand_vec, pts[idx]) * pts[idx]
            norm_t = np.linalg.norm(tangent)
            if norm_t < 1e-12:
                continue
            tangent /= norm_t
            
            # Step size decreases with temperature
            step_size = step_size_base * np.sqrt(T)
            new_pt = pts[idx] + step_size * tangent
            new_pt = new_pt / np.linalg.norm(new_pt)
            
            # Compute new energy efficiently: only changes involving idx
            old_energy = current_energy
            
            # Old contributions involving idx
            old_contrib = 0.0
            for j in range(n):
                if j == idx:
                    continue
                d = np.linalg.norm(pts[idx] - pts[j])
                if d < 1e-15:
                    old_contrib -= np.inf
                    break
                old_contrib += -np.log(d)
            
            # New contributions involving idx
            new_contrib = 0.0
            for j in range(n):
                if j == idx:
                    continue
                d = np.linalg.norm(new_pt - pts[j])
                if d < 1e-15:
                    new_contrib = -np.inf
                    break
                new_contrib += -np.log(d)
            
            delta_e = new_contrib - old_contrib
            
            # Metropolis acceptance
            if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / T)):
                pts[idx] = new_pt
                current_energy = old_energy + delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
            
            # Cool temperature
            T *= 0.99999
            if T < T_min:
                T = T_min
    
    return best_pts, best_energy


def _newton_polish(pts, max_iter=200, time_budget=30.0):
    """
    Newton's method / BFGS-like optimization on the sphere.
    Uses gradient projection to stay on the sphere.
    """
    n = len(pts)
    current_energy, grad = _log_energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    # BFGS Hessian approximation
    H = np.eye(n * 3)  # (N*3, N*3)
    x_flat = pts.flatten()
    
    t_start = time.time()
    
    for it in range(max_iter):
        if time.time() - t_start >= time_budget:
            break
        
        e, grad = _log_energy_and_grad(pts)
        
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-10:
            break
        
        # BFGS step: solve H * delta = -grad
        try:
            delta = np.linalg.solve(H, -grad.flatten())
        except np.linalg.LinAlgError:
            delta = -grad.flatten()
        
        # Line search
        alpha = 0.1
        for _ in range(50):
            new_pts = pts + alpha * delta.reshape(n, 3)
            new_pts = _project_to_sphere(new_pts)
            new_e, _ = _log_energy_and_grad(new_pts)
            
            if new_e < e:
                break
            alpha *= 0.5
        
        # Update BFGS
        s = (new_pts - pts).flatten()
        g_new, _ = _log_energy_and_grad(new_pts)
        y = (g_new - grad).flatten()
        
        sy = np.dot(s, y)
        if sy > 1e-12:
            Hy = H @ y
            H = H + np.outer(s, s) / sy - np.outer(Hy, Hy) / np.dot(y, Hy)
        
        pts = new_pts
    
    return best_pts, best_energy


def construct():
    t_start = time.time()
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    
    # Phase 1: Simulated annealing
    remaining = TIME_LIMIT - (time.time() - t_start)
    sa_budget = min(60.0, remaining * 0.5)
    pts, e_sa = _simulate_anneal(pts, sa_budget)
    
    # Phase 2: Newton polish
    remaining = TIME_LIMIT - (time.time() - t_start)
    np_budget = min(remaining * 0.7, 50.0)
    pts, e_np = _newton_polish(pts, max_iter=500, time_budget=np_budget)
    
    # Final check
    pts = _project_to_sphere(pts)
    
    # Validate
    if not np.all(np.isfinite(pts)):
        pts = _fibonacci_spiral(N)
    
    return pts

# EVOLVE-BLOCK-END
