"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient (tangent): grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2, projected to tangent space.
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist2, 1.0)
    
    # Compute energy
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())
    
    # Compute gradient
    # grad_i = sum_j (x_j - x_i) / dist_ij^2
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad = inv_dist2 @ pts - pts * row_sums
    row_sums = np.sum(inv_dist2, axis=1)
    grad = inv_dist2 @ pts
    grad -= pts * row_sums[:, np.newaxis]
    
    # Project gradient to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _energy_only(pts):
    """Compute only the energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    return -math.fsum(np.log(dists).tolist())


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    """Generate points using a Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron (geodesic dome),
    then project to sphere. Provides a different structure than Fibonacci spiral.
    """
    rng = np.random.RandomState(seed)
    # Start with icosahedron vertices
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Edges of icosahedron
    edges = [(0,11),(0,5),(0,1),(0,7),(0,10),(1,5),(5,11),(11,10),(10,7),
             (7,1),(1,6),(6,10),(2,11),(2,5),(2,9),(2,4),(2,8),(3,5),
             (3,11),(3,4),(3,9),(3,6),(4,9),(5,4),(11,4),(10,7),(6,1),
             (8,4),(9,8),(8,5),(6,9),(7,2),(8,3)]
    
    # Recursively subdivide until we have enough points
    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)])
    
    # If we have more than n, randomly select n
    if len(verts) > n:
        idx = rng.choice(len(verts), size=n, replace=False)
        verts = verts[idx]
    
    return verts


def _random_points(n, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _adam_optimize(pts0, max_iter=3000, lr_start=0.01, lr_end=0.001,
                   beta1=0.9, beta2=0.999, eps=1e-8, deadline=None, check_interval=20):
    """
    Adam optimization on the sphere with learning rate decay.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    
    # Cosine decay for learning rate
    def lr_schedule(t):
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t / max_iter))

    for t in range(1, max_iter + 1):
        if deadline is not None and (t % check_interval == 0 or t == 1):
            if time.time() > deadline:
                break

        energy, grad = _energy_and_grad(pts)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        
        lr = lr_schedule(t)
        
        # Step in tangent direction
        pts_new = pts - lr * update
        pts_new = _project_to_sphere(pts_new)

        if not np.all(np.isfinite(pts_new)):
            break

        energy_new = _energy_only(pts_new)
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
            
        pts = pts_new

    return best_pts, best_energy


def _velocity_verlet_sphere(points, deadline, dt=0.005, n_steps=5000, damping=0.98):
    """
    Velocity-Verlet integration on the sphere.
    """
    pts = _project_to_sphere(points.copy())
    vel = np.zeros_like(pts)
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Initial force
    _, grad0 = _energy_and_grad(pts)
    force = -grad0
    
    for step in range(n_steps):
        if time.time() > deadline - 0.5:
            break
        
        # Velocity-Verlet half step
        vel_half = vel + 0.5 * dt * force
        # Project velocity to tangent space
        dot_v = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half = vel_half - dot_v * pts
        
        # Position update
        pts_new = pts + dt * vel_half
        pts_new = _project_to_sphere(pts_new)
        
        # New force
        _, grad_new = _energy_and_grad(pts_new)
        force_new = -grad_new
        # Project force to tangent space
        dot_f = np.sum(force_new * pts_new, axis=1, keepdims=True)
        force_new = force_new - dot_f * pts_new
        
        # Velocity update with damping
        vel = (vel_half + 0.5 * dt * force_new) * damping
        dot_v2 = np.sum(vel * pts_new, axis=1, keepdims=True)
        vel = vel - dot_v2 * pts_new
        
        pts = pts_new
        force = force_new
        
        energy = _energy_only(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Adaptive damping or step size if energy increases
        if step > 100 and energy > best_energy * 1.05:
            dt *= 0.9
            damping *= 0.99

    return best_pts, best_energy


def _lbfgs_polish(pts, deadline, maxiter=500):
    """
    L-BFGS-B polish using flat Cartesian coordinates with projection.
    """
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 10, 'gtol': 1e-10, 'ftol': 1e-12})
        final_pts = _project_to_sphere(res.x.reshape(N, 3))
        return final_pts, _energy_only(final_pts)
    except Exception:
        return pts, _energy_only(pts)


def _gradient_descent_polish(pts, deadline, max_iter=2000, initial_lr=0.001):
    """
    Gradient descent polish with adaptive step size.
    Helps escape shallow local minima and fine-tune to tighter local minimum.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    lr = initial_lr
    
    for i in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        # Adaptive step size: reduce if energy increases, increase if it decreases
        # Simple heuristic: use gradient norm to scale step
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
            
        # Try a step
        step = lr / (1.0 + grad_norm * 10)
        pts_new = pts - step * grad
        pts_new = _project_to_sphere(pts_new)
        
        new_energy = _energy_only(pts_new)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = pts_new.copy()
            pts = pts_new
            # Increase learning rate slightly if we're making progress
            lr = min(lr * 1.1, initial_lr * 5)
        else:
            # Reduce learning rate
            lr *= 0.7
            if lr < 1e-8:
                break
                
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    best_points = None
    best_energy = float('inf')
    
    # Multiple diverse initializations
    initializers = []
    
    # 1. Fibonacci spiral
    initializers.append(_fibonacci_spiral(N))
    
    # 2. Geodesic dome
    initializers.append(_geodesic_dome(N, seed=42))
    
    # 3. Random points
    rng = np.random.default_rng(123)
    for _ in range(5):
        pts = rng.standard_normal((N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        initializers.append(pts)
    
    # 4. Perturbed Fibonacci
    for _ in range(3):
        base = _fibonacci_spiral(N)
        noise = rng.standard_normal((N, 3)) * 0.1
        pts = base + noise
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        initializers.append(pts)
    
    for idx, init_pts in enumerate(initializers):
        if time.time() > deadline - 30.0:
            break
        
        current_pts = init_pts.copy()
        current_energy = _energy_only(current_pts)
        
        # Phase 1: Velocity-Verlet for coarse optimization
        if time.time() < deadline - 20.0:
            try:
                pts, e = _velocity_verlet_sphere(current_pts, deadline=deadline - 15.0, dt=0.005, n_steps=3000)
                if e < current_energy:
                    current_pts, current_energy = pts, e
            except Exception:
                pass
        
        # Phase 2: Adam for medium-scale optimization
        if time.time() < deadline - 15.0:
            try:
                pts, e = _adam_optimize(current_pts, max_iter=2000, deadline=deadline - 10.0)
                if e < current_energy:
                    current_pts, current_energy = pts, e
            except Exception:
                pass
        
        # Phase 3: L-BFGS polish
        if time.time() < deadline - 10.0:
            try:
                pts, e = _lbfgs_polish(current_pts, deadline=deadline - 5.0, maxiter=300)
                if e < current_energy:
                    current_pts, current_energy = pts, e
            except Exception:
                pass
        
        # Phase 4: Gradient descent polish for fine-tuning
        if time.time() < deadline - 5.0:
            try:
                pts, e = _gradient_descent_polish(current_pts, deadline=deadline - 2.0, max_iter=1000)
                if e < current_energy:
                    current_pts, current_energy = pts, e
            except Exception:
                pass
        
        # Update global best
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = current_pts.copy()
    
    # Final high-precision polish on the best solution found
    if best_points is not None and time.time() < deadline - 3.0:
        try:
            # Tighter L-BFGS
            pts, e = _lbfgs_polish(best_points, deadline=deadline - 1.0, maxiter=500)
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            pass
        
        # Final gradient descent
        if time.time() < deadline - 1.0:
            try:
                pts, e = _gradient_descent_polish(best_points, deadline=deadline - 0.5, max_iter=500, initial_lr=0.0005)
                if e < best_energy:
                    best_energy = e
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Fallback to Fibonacci if nothing worked
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    # Ensure validity
    best_points = _project_to_sphere(best_points)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Perturb slightly
                best_points[j] += np.random.normal(0, 1e-6, 3)
                best_points[j] = best_points[j] / np.linalg.norm(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END
