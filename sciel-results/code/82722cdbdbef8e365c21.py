"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on the sphere.
    
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-30:
                continue
            d = math.sqrt(d2)
            # dE/dx_i contribution from pair (i,j): (x_i - x_j) / d^2
            grad[i] += diff / d2
    # Project gradient to tangent space of the sphere
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    return grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _golden_spiral():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points uniformly on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _gradient_descent(pts, max_iter=5000, tol=1e-12, deadline=None):
    """Gradient descent on the sphere with adaptive step size."""
    pts = _project_to_sphere(pts.copy())
    current_energy = _log_energy(pts)
    
    # Adaptive step size
    lr = 0.1
    best_energy = current_energy
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _log_energy_grad(pts)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        
        # Take a step
        step = lr * grad / max(grad_norm, 1e-10)
        new_pts = pts + step
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _log_energy(new_pts)
        
        if new_energy < current_energy:
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
            # Increase step size slightly on success
            lr = min(lr * 1.05, 0.5)
        else:
            # Decrease step size on failure
            lr *= 0.7
            if lr < 1e-8:
                break
    
    return best_pts, best_energy


def _simulated_annealing(pts, max_iter=3000, deadline=None):
    """Simulated annealing on the sphere."""
    pts = _project_to_sphere(pts.copy())
    current_energy = _log_energy(pts)
    best_energy = current_energy
    best_pts = pts.copy()
    
    T = 0.1
    T_min = 1e-6
    cooling = 0.999
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Propose a random move
        idx = np.random.randint(N)
        direction = np.random.standard_normal(3)
        # Project direction to tangent space
        direction -= np.dot(direction, pts[idx]) * pts[idx]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            continue
        direction /= dir_norm
        
        step_size = 0.1 * math.sqrt(T)
        new_pts = pts.copy()
        new_pts[idx] += step_size * direction
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _log_energy(new_pts)
        delta = new_energy - current_energy
        
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        T = max(T * cooling, T_min)
    
    return best_pts, best_energy


def _refine(pts, deadline):
    """Final refinement using gradient descent with tight tolerance."""
    best_pts, best_energy = _gradient_descent(pts, max_iter=20000, tol=1e-14, deadline=deadline)
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Golden spiral
    initial_configs.append(_golden_spiral())
    
    # 2. Random configurations
    for _ in range(5):
        initial_configs.append(_random_sphere(N, rng))
    
    # 3. Perturbed golden spiral
    gs = _golden_spiral()
    for scale in [0.1, 0.3, 0.5]:
        perturbed = gs + scale * np.random.standard_normal((N, 3))
        initial_configs.append(_project_to_sphere(perturbed))
    
    # Run gradient descent on each configuration
    best_pts = None
    best_energy = float('inf')
    
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Gradient descent
        refined_pts, energy = _gradient_descent(init_pts, max_iter=3000, tol=1e-12, deadline=deadline)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = refined_pts
        
        # If we have time, also try simulated annealing from this result
        if time.time() < deadline - 20.0:
            sa_pts, sa_energy = _simulated_annealing(refined_pts, max_iter=2000, deadline=deadline)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts
    
    # Final high-precision polish
    if best_pts is not None and time.time() < deadline - 5.0:
        best_pts, best_energy = _refine(best_pts, deadline)
    
    # Ensure validity
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them
                diff = best_pts[i] - best_pts[j]
                if np.linalg.norm(diff) < 1e-15:
                    diff = np.array([1.0, 0.0, 0.0])
                diff /= np.linalg.norm(diff)
                best_pts[i] += 1e-8 * diff
                best_pts[j] -= 1e-8 * diff
                best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
