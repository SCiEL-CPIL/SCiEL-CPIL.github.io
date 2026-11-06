"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    vals = dists[iu]
    vals = np.maximum(vals, 1e-15)
    return math.fsum(np.log(1.0 / vals).tolist())

def _energy_and_grad(points):
    """Compute energy and gradient (Euclidean, not projected)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum over i<j of log(1/d_ij)
    iu = np.triu_indices(n, k=1)
    d_vals = dists[iu]
    d_vals = np.maximum(d_vals, 1e-15)
    energy = math.fsum(np.log(1.0 / d_vals).tolist())
    
    # Gradient: for each i, grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = np.zeros_like(dists)
    inv_d2[iu] = 1.0 / (d_vals ** 2)
    iu_rev = (iu[1], iu[0])
    inv_d2[iu_rev] = inv_d2[iu]
    np.fill_diagonal(inv_d2, 0.0)
    
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)

def _icosahedron_sphere(n):
    """Generate points based on icosahedron vertices, then subdivide."""
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Subdivide faces to get more points
    # For simplicity, just use icosahedron + random perturbations
    # Actually, let's just use a better starting point
    # Use a combination of structured and random
    if n <= 12:
        return verts[:n]
    
    # For larger n, use Fibonacci spiral as base and add some random
    pts = _fibonacci_spiral(n)
    return pts

def _optimize_lbfgs(points, max_iter=1000, deadline=None):
    """Optimize using L-BFGS-B with parameterization on sphere."""
    n = len(points)
    
    # Parameterize using tangent space: x = x0 + t where t is tangent
    # Actually, let's use a different approach: optimize in R^3 with constraint
    # Use scipy minimize with SLSQP or trust-constr
    
    def obj(x_flat):
        if deadline is not None and time.time() > deadline:
            return 1e100
        pts = x_flat.reshape(n, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e
    
    def grad(x_flat):
        if deadline is not None and time.time() > deadline:
            return np.zeros_like(x_flat)
        pts = x_flat.reshape(n, 3)
        pts = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts)
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return g_tangent.flatten()
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            obj, x0, jac=grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success or result.fun < 1e100:
            best_pts = _project_to_sphere(result.x.reshape(n, 3))
            best_e = _energy(best_pts)
            return best_pts, best_e
    except Exception:
        pass
    
    # Fallback to manual gradient descent
    return _manual_gradient_descent(points, max_iter=max_iter, deadline=deadline)

def _manual_gradient_descent(points, max_iter=2000, lr=0.01, tol=1e-12, deadline=None):
    """Manual projected gradient descent with line search."""
    n = len(points)
    best_energy = _energy(points)
    best_points = points.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        grad_norm = np.linalg.norm(grad_tangent)
        
        if grad_norm < tol:
            break
        
        # Backtracking line search
        step = lr
        improved = False
        for _ in range(30):
            new_points = _project_to_sphere(points - step * grad_tangent)
            new_energy = _energy(new_points)
            if new_energy < energy:
                points = new_points
                improved = True
                break
            step *= 0.5
        
        if not improved:
            # Try smaller step
            step = lr * 0.1
            new_points = _project_to_sphere(points - step * grad_tangent)
            new_energy = _energy(new_points)
            if new_energy < energy:
                points = new_points
            else:
                break
        
        # Decay learning rate slowly
        if it % 200 == 199:
            lr *= 0.95
    
    return best_points, best_energy

def _momentum_descent(points, max_iter=3000, lr=0.01, momentum=0.9, deadline=None):
    """Gradient descent with momentum on the sphere."""
    n = len(points)
    best_energy = _energy(points)
    best_points = points.copy()
    velocity = np.zeros_like(points)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Update velocity with momentum
        velocity = momentum * velocity - lr * grad_tangent
        
        # Project velocity to tangent space
        velocity_tangent = velocity - np.sum(velocity * points, axis=1, keepdims=True) * points
        
        # Take step
        new_points = _project_to_sphere(points + velocity_tangent)
        new_energy = _energy(new_points)
        
        if new_energy < energy:
            points = new_points
        else:
            # Reduce step
            velocity *= 0.5
            new_points = _project_to_sphere(points + velocity)
            new_energy = _energy(new_points)
            if new_energy < energy:
                points = new_points
            else:
                velocity = np.zeros_like(points)
    
    return best_points, best_energy

def _simulated_annealing(points, max_iter=3000, t_start=0.1, t_end=0.001, deadline=None):
    """Simulated annealing on the sphere."""
    n = len(points)
    best_energy = _energy(points)
    best_points = points.copy()
    current_energy = best_energy
    current_points = points.copy()
    
    t = t_start
    cooling = (t_end / t_start) ** (1.0 / max_iter)
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random perturbation
        perturbation = rng.randn(n, 3) * 0.02
        new_points = _project_to_sphere(current_points + perturbation)
        
        new_energy = _energy(new_points)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(t, 1e-10)):
            current_points = new_points
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_points.copy()
        
        t *= cooling
    
    return best_points, best_energy

def construct():
    deadline = time.time() + 140.0  # Leave 10s safety margin
    
    # Start from Fibonacci spiral
    points = _fibonacci_spiral(N)
    
    # Phase 1: L-BFGS optimization
    best_points, best_energy = _optimize_lbfgs(points, max_iter=2000, deadline=deadline)
    
    # Phase 2: Try multiple random restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        seed = 12345 + restart * 1000
        rand_points = _random_points(N, seed)
        
        # Short optimization from random start
        pts, e = _manual_gradient_descent(rand_points, max_iter=500, lr=0.01, deadline=deadline)
        
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 3: Momentum descent for further refinement
    if time.time() < deadline - 20.0:
        pts, e = _momentum_descent(best_points, max_iter=1000, lr=0.005, deadline=deadline)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 4: Simulated annealing for global search
    if time.time() < deadline - 15.0:
        pts, e = _simulated_annealing(best_points, max_iter=1000, t_start=0.05, t_end=0.001, deadline=deadline)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 5: Final polish with tight tolerance
    if time.time() < deadline - 10.0:
        pts, e = _manual_gradient_descent(best_points, max_iter=2000, lr=0.001, tol=1e-14, deadline=deadline)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Ensure valid output
    best_points = _project_to_sphere(best_points)
    
    # Check for any points too close
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-9:
                # Nudge them apart
                rng = np.random.RandomState(i * 1000 + j)
                best_points[j] += rng.randn(3) * 1e-6
                best_points = _project_to_sphere(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
