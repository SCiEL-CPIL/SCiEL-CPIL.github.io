"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-9:
                return 1e10  # penalty for too-close points
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_grad(points):
    """Compute gradient of logarithmic energy with respect to points on sphere (projected)."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = points[i] - points[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-9:
                continue
            factor = 1.0 / (d * d)
            grad[i] -= d_vec * factor
            grad[j] += d_vec * factor
    # Project gradient onto tangent space of sphere: grad_tangent = grad - (grad . x) * x
    for i in range(n):
        proj = np.dot(grad[i], points[i])
        grad[i] -= proj * points[i]
    return grad


def _riemannian_descent(points, max_iter=50000, lr_init=0.01, momentum=0.9, tol=1e-15):
    """Riemannian gradient descent with momentum on the sphere."""
    pts = points.copy()
    n = len(pts)
    velocity = np.zeros_like(pts)
    lr = lr_init
    
    for it in range(max_iter):
        grad = _log_energy_grad(pts)
        
        # Update with momentum in tangent space
        velocity = momentum * velocity - lr * grad
        
        # Project velocity onto tangent space
        for i in range(n):
            proj = np.dot(velocity[i], pts[i])
            velocity[i] -= proj * pts[i]
        
        # Update points and renormalize
        pts_new = pts + velocity
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts_new = pts_new / norms
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Adaptive learning rate: reduce if energy increases
        e_old = _log_energy(pts)
        e_new = _log_energy(pts_new)
        if e_new > e_old:
            lr *= 0.5
            pts = pts  # keep old
        else:
            pts = pts_new
            if it % 1000 == 0 and it > 0:
                lr *= 0.95  # slow decay
        
        if lr < 1e-10:
            break
    
    return pts, _log_energy(pts)


def construct():
    # Generate multiple initial configurations
    rng = np.random.default_rng(12345)
    
    initializations = []
    
    # 1. Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib)
    
    # 2. Perturbed Fibonacci spheres
    for trial in range(5):
        noise = rng.normal(0, 0.15, size=(N, 3))
        perturbed = fib + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        perturbed = perturbed / norms
        initializations.append(perturbed)
    
    # 3. Random initializations
    for trial in range(5):
        random_pts = rng.normal(0, 1, size=(N, 3))
        norms = np.linalg.norm(random_pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        random_pts = random_pts / norms
        initializations.append(random_pts)
    
    # 4. Spherical codes (icosahedron-like for small N, but for N=100 use random)
    # Add a few more diverse random starts
    for trial in range(3):
        random_pts = rng.normal(0, 1, size=(N, 3))
        norms = np.linalg.norm(random_pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        random_pts = random_pts / norms
        initializations.append(random_pts)
    
    best_pts = None
    best_energy = float('inf')
    
    # Run Riemannian descent from each initialization
    for init_pts in initializations:
        pts, energy = _riemannian_descent(init_pts, max_iter=30000, lr_init=0.005, momentum=0.9, tol=1e-14)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final polishing: run a longer descent from the best point
    if best_pts is not None:
        pts, energy = _riemannian_descent(best_pts, max_iter=50000, lr_init=0.002, momentum=0.95, tol=1e-15)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts
    
    return best_pts
# EVOLVE-BLOCK-END
