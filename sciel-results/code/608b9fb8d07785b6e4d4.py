"""Optimized logarithmic energy minimizer for 100 points on the unit sphere."""
import numpy as np
import time
import math

N = 100
DEADLINE = 145.0  # safety margin before 150s kill

def _fibonacci_spiral(n):
    """Golden-angle spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _project_to_sphere(pts):
    """Normalize each row to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return -1e18
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _log_energy_and_grad(pts):
    """Compute log energy and gradient via pairwise distances.
    
    Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    d_ij = |x_i - x_j|
    d/dx_i [log(1/d_ij)] = d/dx_i [-log(d_ij)] = -(1/d_ij) * d_ij/(d_ij^2) * (x_i - x_j)
    = -(x_i - x_j) / d_ij^2  ... wait let me redo.
    
    log(1/d) = -log(d), d = |x_i - x_j|
    d(-log d)/dx_i = -(1/d) * (x_i - x_j)/d = -(x_i - x_j)/d^2
    """
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                return -1e18, grad
            inv_d2 = 1.0 / (d * d)
            energy -= math.log(d)  # log(1/d) = -log(d)
            g = diff * inv_d2
            grad[i] += g
            grad[j] -= g
    return energy, grad

def _log_energy_and_grad_fast(pts):
    """Vectorized computation of log energy and gradient."""
    n = len(pts)
    # Compute all pairwise differences
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    d = np.linalg.norm(diff, axis=2)  # (n, n)
    
    # Mask diagonal and only use upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_masked = d[mask]
    
    if np.any(d_masked < 1e-12):
        return -1e18, np.zeros_like(pts)
    
    energy = -np.sum(np.log(d_masked))
    
    # Gradient: for pair (i,j), contribution to grad[i] is (pts[i]-pts[j])/d^2
    # grad[j] gets negative
    inv_d2 = np.zeros_like(d)
    inv_d2[mask] = 1.0 / (d_masked ** 2)
    
    # grad[i] = sum_j (pts[i] - pts[j]) / d_ij^2  for j != i
    # Using broadcasting: grad = sum over j of (pts[i]-pts[j]) * inv_d2[i,j]
    # diff[i,j] = pts[i] - pts[j], so grad[i] = sum_j diff[i,j] * inv_d2[i,j]
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _log_energy_and_grad_vec(pts):
    """Efficient vectorized log energy and gradient."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]  # (n,n,3)
    d2 = np.einsum('ijk,ijk->ij', diff, diff)  # (n,n)
    np.fill_diagonal(d2, 1.0)  # avoid div by zero on diagonal
    
    d = np.sqrt(d2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_masked = d[mask]
    
    if np.any(d_masked < 1e-12):
        return -1e18, np.zeros_like(pts)
    
    energy = -math.fsum(np.log(d_masked).tolist())
    
    # Gradient computation
    # For each pair (i,j) with i<j: grad[i] += (pts[i]-pts[j])/d^2, grad[j] -= (pts[i]-pts[j])/d^2
    inv_d2 = np.zeros_like(d2)
    inv_d2[mask] = 1.0 / (d_masked ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j (pts[i] - pts[j]) * inv_d2[i,j]
    # = pts[i] * sum_j inv_d2[i,j] - sum_j pts[j] * inv_d2[i,j]
    row_sum = np.sum(inv_d2, axis=1)  # (n,)
    # sum_j pts[j] * inv_d2[i,j] = inv_d2 @ pts  (n x n) @ (n x 3) = (n x 3)
    weighted_sum = inv_d2 @ pts  # (n, 3)
    grad = pts * row_sum[:, np.newaxis] - weighted_sum
    
    return energy, grad

def _simulated_annealing(pts, max_iters, T_start, T_end, lr_start, lr_end, 
                         n_restarts=1, time_limit=None, t0=None):
    """Simulated annealing with gradient descent on the sphere.
    
    Uses the fact that for points on the sphere, the gradient of the energy
    projected onto the tangent space drives the optimization.
    """
    if t0 is None:
        t0 = time.time()
    
    best_pts = pts.copy()
    best_energy = _log_energy_and_grad_vec(best_pts)[0]
    
    current = pts.copy()
    current_energy = best_energy
    
    n = len(current)
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - t0 > time_limit:
            break
        
        # Annealing schedule
        progress = it / max(max_iters - 1, 1)
        T = T_start * (T_end / T_start) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress
        
        # Compute gradient
        _, grad = _log_energy_and_grad_vec(current)
        
        # Project gradient to tangent space: grad_tangent = grad - (grad . x) * x
        dot = np.sum(current * grad, axis=1, keepdims=True)
        grad_tan = grad - dot * current
        
        # Update: move in negative gradient direction
        step = current - lr * grad_tan
        
        # Add random perturbation for simulated annealing
        if T > 1e-10:
            noise = np.random.randn(n, 3) * T
            step = step + noise
        
        # Project back to sphere
        step = _project_to_sphere(step)
        
        # Compute new energy
        new_energy = _log_energy_and_grad_vec(step)[0]
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current = step
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = step.copy()
        else:
            # Accept with probability exp(-delta/T)
            delta = new_energy - current_energy
            if delta < T * 20:  # avoid overflow
                prob = math.exp(-delta / max(T, 1e-15))
                if np.random.random() < prob:
                    current = step
                    current_energy = new_energy
    
    return best_pts, best_energy

def _gradient_descent_polish(pts, max_iters=5000, tol=1e-12, time_limit=None, t0=None):
    """High-precision gradient descent polish with adaptive step size."""
    if t0 is None:
        t0 = time.time()
    
    current = pts.copy()
    energy = _log_energy_and_grad_vec(current)[0]
    best_pts = current.copy()
    best_energy = energy
    
    # Backtracking line search parameters
    c1 = 1e-4
    rho = 0.5
    max_line_iters = 20
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - t0 > time_limit:
            break
        
        _, grad = _log_energy_and_grad_vec(current)
        
        # Project to tangent space
        dot = np.sum(current * grad, axis=1, keepdims=True)
        grad_tan = grad - dot * current
        
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < tol:
            break
        
        # Adaptive step size
        alpha = min(1.0, 1.0 / (grad_norm + 1e-10))
        
        # Line search
        accepted = False
        for _ in range(max_line_iters):
            step = current - alpha * grad_tan
            step = _project_to_sphere(step)
            new_energy = _log_energy_and_grad_vec(step)[0]
            
            if new_energy < energy - c1 * alpha * grad_norm ** 2:
                current = step
                energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = step.copy()
                accepted = True
                break
            alpha *= rho
        
        if not accepted:
            # Try smaller step
            alpha = 1e-6
            step = current - alpha * grad_tan
            step = _project_to_sphere(step)
            new_energy = _log_energy_and_grad_vec(step)[0]
            if new_energy < energy:
                current = step
                energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = step.copy()
            else:
                break
    
    return best_pts, best_energy

def construct():
    t0 = time.time()
    remaining = lambda: DEADLINE - time.time()
    
    # Initial point: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    
    # Phase 1: Coarse simulated annealing
    time_left = remaining()
    if time_left > 30:
        max_iters = min(3000, int(time_left * 30))  # ~30 iters/sec estimate
        pts, e = _simulated_annealing(
            pts, max_iters=max_iters,
            T_start=0.05, T_end=1e-5,
            lr_start=0.01, lr_end=1e-5,
            time_limit=remaining(), t0=t0
        )
    
    # Phase 2: Fine simulated annealing from best
    time_left = remaining()
    if time_left > 20:
        max_iters = min(5000, int(time_left * 25))
        pts, e = _simulated_annealing(
            pts, max_iters=max_iters,
            T_start=0.01, T_end=1e-7,
            lr_start=0.005, lr_end=1e-6,
            time_limit=remaining(), t0=t0
        )
    
    # Phase 3: Gradient descent polish
    time_left = remaining()
    if time_left > 5:
        max_iters = min(10000, int(time_left * 20))
        pts, e = _gradient_descent_polish(
            pts, max_iters=max_iters, tol=1e-14,
            time_limit=remaining() - 2, t0=t0
        )
    
    # Final check: ensure valid output
    pts = _project_to_sphere(pts)
    
    # Verify no points too close
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-9:
                # Nudge apart
                pts[j] += 1e-8 * np.random.randn(3)
                pts = _project_to_sphere(pts)
                break
    
    return pts
