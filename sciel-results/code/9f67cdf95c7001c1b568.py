"""Optimize logarithmic energy of 150 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150
DEADLINE = 145.0  # seconds, with 5s safety margin before 150s limit


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    where d_ij is chordal distance.
    
    Gradient w.r.t. x_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2 * (-1) ... 
    Let's derive: E = -sum_{i<j} log|d_ij|, d_ij = |x_i - x_j|
    dE/dx_i = -sum_{j!=i} (1/d_ij) * (x_i - x_j)/d_ij = -sum_{j!=i} (x_i - x_j)/d_ij^2
    But we need to project onto the tangent space of the sphere.
    """
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    # Note: dE/dx_i = -sum_{j!=i} (x_i - x_j)/d_ij^2 ... wait let me re-derive
    # E = sum_{i<j} -log|d_ij|
    # dE/dx_i = sum_{j: j>i} -d/dx_i log|d_ij| + sum_{j: j<i} -d/dx_i log|d_ji|
    # d/dx_i log|d_ij| = (x_i - x_j) / |d_ij|^2
    # So dE/dx_i = -sum_{j!=i} (x_i - x_j) / |d_ij|^2
    # This is the negative of the Coulomb-like force, which makes sense.
    
    # grad[i] = -sum_{j!=i} (x_i - x_j) / dist_sq[i,j]
    # = -sum_{j!=i} x_i/dist_sq[i,j] + sum_{j!=i} x_j/dist_sq[i,j]
    # = -x_i * sum_{j!=i} 1/dist_sq[i,j] + sum_{j!=i} x_j / dist_sq[i,j]
    
    inv_dist_sq = 1.0 / dist_sq  # (N, N), diagonal is 1.0 but we'll zero it out
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -x_i * sum_j inv_dist_sq[i,j] + sum_j x_j * inv_dist_sq[i,j]
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    # sum_j x_j * inv_dist_sq[i,j] = inv_dist_sq @ pts
    weighted_sum = inv_dist_sq @ pts  # (N, 3)
    
    grad = -pts * row_sums[:, np.newaxis] + weighted_sum  # (N, 3)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _optimize_single(pts0, max_time):
    """Optimize a single starting configuration using momentum-based projected gradient descent."""
    start_time = time.time()
    
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_e = _energy_and_grad(pts)[0]
    
    lr = 0.02  # Initial learning rate
    min_lr = 1e-10
    max_lr = 0.5
    momentum = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    # Momentum and adaptive learning rate state
    velocity = np.zeros_like(pts)
    m = np.zeros_like(pts)  # First moment
    v = np.zeros_like(pts)  # Second moment
    t = 0
    patience = 0
    max_patience = 100
    last_improve = 0
    
    while time.time() - start_time < max_time:
        e, grad = _energy_and_grad(pts)
        
        if e < best_e - 1e-12:
            best_e = e
            best_pts = pts.copy()
            patience = 0
            last_improve = t
        else:
            patience += 1
            if patience > max_patience and t - last_improve > 50:
                break
        
        # Project gradient onto tangent space
        g_tan = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Check convergence
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-14:
            break
        
        # Adam-style adaptive learning rate
        t += 1
        m = beta2 * m + (1 - beta2) * g_tan
        v = beta2 * v + (1 - beta2) * (g_tan ** 2)
        m_hat = m / (1 - beta2 ** t)
        v_hat = v / (1 - beta2 ** t)
        adaptive_lr = lr / (np.sqrt(v_hat) + eps)
        
        # Momentum update
        velocity = momentum * velocity - adaptive_lr * m_hat
        
        # Project velocity onto tangent space
        velocity = velocity - np.sum(velocity * pts, axis=1, keepdims=True) * pts
        
        # Backtracking line search
        alpha = 1.0
        accepted = False
        for _ in range(20):
            # Move in tangent direction and project back to sphere
            new_pts = pts + alpha * velocity
            new_pts = _project_to_sphere(new_pts)
            new_e, _ = _energy_and_grad(new_pts)
            
            # Use a more lenient Armijo condition
            if new_e < e + 1e-6:
                pts = new_pts
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            # Reduce learning rate and reset momentum
            lr *= 0.5
            velocity *= 0.5
            if lr < min_lr:
                break
        else:
            # Occasionally increase learning rate if stuck
            if patience > 20 and t - last_improve > 30:
                lr = min(max_lr, lr * 1.2)
    
    return best_pts, best_e


def _lbfgs_polish(pts0, max_time, tol=1e-12):
    """Final high-precision polish using L-BFGS-B with projection."""
    start_time = time.time()
    
    def objective_and_grad(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, grad = _energy_and_grad(pts)
        return e, grad.flatten()
    
    x0 = pts0.flatten()
    
    # Use L-BFGS-B with tight tolerance
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={
            'maxiter': 5000,
            'ftol': tol,
            'gtol': 1e-14,
            'maxfun': 50000,
            'maxls': 50
        }
    )
    
    pts = _project_to_sphere(result.x.reshape(N, 3))
    e, _ = _energy_and_grad(pts)
    
    return pts, e


def _simulated_annealing(pts, max_time, rng):
    """Simulated annealing step to escape local minima."""
    start_time = time.time()
    current = pts.copy()
    current_e, _ = _energy_and_grad(current)
    best = current.copy()
    best_e = current_e
    
    T = 0.1
    T_min = 1e-6
    cooling = 0.99
    
    while time.time() - start_time < max_time and T > T_min:
        # Pick a random point and perturb it
        idx = rng.integers(N)
        angle = rng.uniform(0, 2 * np.pi)
        # Random direction in tangent space
        rand_dir = rng.standard_normal(3)
        rand_dir -= np.dot(rand_dir, current[idx]) * current[idx]
        norm = np.linalg.norm(rand_dir)
        if norm > 1e-15:
            rand_dir /= norm
        
        # Move along random direction
        step_size = T * 0.5
        new_pt = current[idx].copy()
        new_pt += step_size * rand_dir
        new_pt = new_pt / np.linalg.norm(new_pt)
        
        new_pts = current.copy()
        new_pts[idx] = new_pt
        
        new_e, _ = _energy_and_grad(new_pts)
        delta_e = new_e - current_e
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current = new_pts
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best = current.copy()
        
        T *= cooling
    
    return best, best_e


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci spiral (best single start)
    starts.append(_fibonacci_sphere(N))
    
    # 2. Slightly perturbed Fibonacci variants
    for scale in [0.01, 0.03, 0.08]:
        fib = _fibonacci_sphere(N)
        noise = scale * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))
    
    # 3. A few random starts for diversity
    for _ in range(3):
        starts.append(_random_sphere(N, rng))
    
    # Optimize each start with gradient descent
    best_pts = None
    best_energy = float('inf')
    
    # Allocate more time to the best starts
    time_weights = [0.35, 0.15, 0.15, 0.10, 0.10, 0.08, 0.07]  # Sum to 1.0
    total_available = DEADLINE - (time.time() - start_time) - 20  # Reserve 20s for final polish
    
    for i, s in enumerate(starts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        remaining = DEADLINE - (time.time() - start_time) - 15
        if remaining < 3:
            break
        
        # Allocate time based on weight, with minimum 3s
        w = time_weights[i] if i < len(time_weights) else 0.05
        opt_time = max(3.0, min(w * total_available, remaining - 2))
        
        try:
            pts_opt, e_opt = _optimize_single(s, opt_time)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            continue
    
    # Final high-precision polish with L-BFGS for tight convergence
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        remaining = DEADLINE - (time.time() - start_time) - 3
        if remaining > 5:
            try:
                # First do a quick gradient polish
                pts_quick, e_quick = _optimize_single(best_pts, min(remaining * 0.4, 10))
                if e_quick < best_energy:
                    best_energy = e_quick
                    best_pts = pts_quick.copy()
                
                # Then L-BFGS polish with remaining time
                remaining2 = DEADLINE - (time.time() - start_time) - 3
                if remaining2 > 3:
                    pts_final, e_final = _lbfgs_polish(best_pts, remaining2)
                    if e_final < best_energy:
                        best_energy = e_final
                        best_pts = pts_final.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    assert best_pts.shape == (N, 3)
    assert np.all(np.isfinite(best_pts))
    assert np.all(np.linalg.norm(best_pts, axis=1) > 1e-15)
    
    # Check no two points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them slightly
                best_pts[j] += 1e-8 * rng.standard_normal(3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
