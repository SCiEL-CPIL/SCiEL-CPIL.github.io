"""Logarithmic energy minimization on the unit sphere via simulated annealing + polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0  # seconds before the hard limit


def _log_energy(pts):
    """Compute the logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    d = cdist(pts, pts)
    np.fill_diagonal(d, 1.0)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    dists = d[iu]
    # Guard against zero distances
    dists = np.maximum(dists, 1e-15)
    return float(np.sum(np.log(1.0 / dists)))


def _log_energy_grad(pts):
    """Gradient of the logarithmic energy with respect to points on the sphere.
    
    For E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, 1.0)
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Using broadcasting: diff[i,j] = x_i - x_j, shape (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(d2, 1.0)
    # weight[i,j] = 1 / |x_i - x_j|^2
    weights = 1.0 / d2  # (N, N)
    np.fill_diagonal(weights, 0.0)
    # grad[i] = sum_j weights[i,j] * (x_i - x_j)
    grad = np.sum(weights[:, :, np.newaxis] * diff, axis=1)  # (N, 3)
    return grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _random_rotation():
    """Generate a random rotation matrix (Haar measure on SO(3))."""
    # Use quaternion method for uniform random rotation
    u1, u2, u3 = np.random.rand(3)
    a = 2.0 * math.pi * u1
    b = math.acos(2.0 * u2 - 1.0)
    c = 2.0 * math.pi * u3
    # Build rotation matrix from axis-angle
    ax = math.sin(b) * math.cos(a)
    ay = math.sin(b) * math.sin(a)
    az = math.cos(b)
    # Rodrigues formula
    K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
    R = np.eye(3) + math.sin(math.pi) * K + (1 - math.cos(math.pi)) * (K @ K)
    # Actually let's use a simpler approach: two random unit vectors
    return _random_rotation_simple()


def _random_rotation_simple():
    """Generate a random rotation matrix using two random unit vectors."""
    v1 = np.random.randn(3)
    v1 /= np.linalg.norm(v1)
    v2 = np.random.randn(3)
    v2 = v2 - np.dot(v2, v1) * v1
    n2 = np.linalg.norm(v2)
    if n2 < 1e-10:
        v2 = np.random.randn(3)
        v2 = v2 - np.dot(v2, v1) * v1
        n2 = np.linalg.norm(v2)
    v2 /= n2
    v3 = np.cross(v1, v2)
    R = np.column_stack([v1, v2, v3])
    return R


def construct():
    start_time = time.time()
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _project_to_sphere(pts)
    
    current_energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    # Simulated annealing parameters
    T_initial = 50.0
    T_final = 0.01
    # Number of SA iterations - we'll adapt based on time
    # Each iteration: propose move, evaluate
    # Estimate time per iteration
    max_iterations = 200000  # upper bound, will be limited by time
    
    # Cooling schedule: geometric
    if max_iterations > 1:
        cool_rate = (T_final / T_initial) ** (1.0 / max_iterations)
    else:
        cool_rate = 0.99
    
    rng = np.random.RandomState(42)
    
    iteration = 0
    while iteration < max_iterations:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Check time every 1000 iterations to reduce overhead
        if iteration % 1000 == 0 and elapsed > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Temperature
        T = T_initial * (cool_rate ** iteration)
        if T < T_final:
            T = T_final
        
        # Propose a move: random rotation of a random subset of points
        # Strategy: pick k random points, apply a small random rotation to them
        # Or: move a single point in a random direction on the sphere
        
        # Mix of strategies:
        # 1. Single point move (most common)
        # 2. Pair swap of angular positions
        # 3. Small global rotation
        
        strategy = rng.random()
        
        if strategy < 0.7:
            # Move a single point in a random tangent direction
            idx = rng.randint(N)
            # Random tangent direction at pts[idx]
            tangent = np.random.randn(3)
            tangent -= np.dot(tangent, pts[idx]) * pts[idx]
            tn = np.linalg.norm(tangent)
            if tn < 1e-10:
                continue
            tangent /= tn
            # Step size proportional to temperature
            step = T * 0.1
            new_pts = pts.copy()
            new_pts[idx] = pts[idx] + step * tangent
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
            
        elif strategy < 0.9:
            # Move two points toward/away from each other
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            step = T * 0.05
            # Move idx1 toward idx2 and idx2 toward idx1 (or away)
            direction = pts[idx2] - pts[idx1]
            dn = np.linalg.norm(direction)
            if dn < 1e-10:
                continue
            direction /= dn
            # Tangential components
            t1 = direction - np.dot(direction, pts[idx1]) * pts[idx1]
            t2 = -direction - np.dot(-direction, pts[idx2]) * pts[idx2]
            t1n = np.linalg.norm(t1)
            t2n = np.linalg.norm(t2)
            if t1n > 1e-10:
                t1 /= t1n
            else:
                t1 = np.random.randn(3)
                t1 -= np.dot(t1, pts[idx1]) * pts[idx1]
                t1 /= np.linalg.norm(t1)
            if t2n > 1e-10:
                t2 /= t2n
            else:
                t2 = np.random.randn(3)
                t2 -= np.dot(t2, pts[idx2]) * pts[idx2]
                t2 /= np.linalg.norm(t2)
            
            new_pts = pts.copy()
            new_pts[idx1] = pts[idx1] + step * t1
            new_pts[idx1] /= np.linalg.norm(new_pts[idx1])
            new_pts[idx2] = pts[idx2] + step * t2
            new_pts[idx2] /= np.linalg.norm(new_pts[idx2])
            
        else:
            # Small global rotation
            R = _random_rotation_simple()
            angle = T * 0.01
            # Use Rodrigues for small angle around random axis
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R_small = np.eye(3) + angle * K + (angle ** 2 / 2) * (K @ K)
            new_pts = (R_small @ pts.T).T
            new_pts = _project_to_sphere(new_pts)
        
        # Evaluate new energy
        new_energy = _log_energy(new_pts)
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-10)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        iteration += 1
    
    # Final polish with L-BFGS-B or trust-constr on the sphere
    # Use parameterization: optimize on tangent space with projection
    # Or use scipy minimize with a constraint
    
    # Better approach: use the fact that we can do gradient descent with projection
    # Use L-BFGS-B on the flattened coordinates with a sphere constraint via penalty or retraction
    
    # Simple approach: do a few rounds of projected gradient descent
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 10.0:
        # Use scipy minimize with SLSQP or trust-constr
        # Parameterize using the fact that points are on the sphere
        # We'll use the approach of minimizing with a constraint
        
        # Actually, let's do a simple projected gradient method
        lr = 0.01
        for polish_iter in range(200):
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT - 5.0:
                break
            
            grad = _log_energy_grad(best_pts)
            # Project gradient to tangent space
            tangent_grad = grad - np.sum(grad * best_pts, axis=1, keepdims=True) * best_pts
            # Update
            new_pts = best_pts - lr * tangent_grad
            new_pts = _project_to_sphere(new_pts)
            new_e = _log_energy(new_pts)
            if new_e < best_energy:
                best_energy = new_e
                best_pts = new_pts
                # Decay learning rate
                lr *= 0.99
            else:
                lr *= 0.5
    
    # Final check: ensure all points are on the unit sphere and finite
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    d = cdist(best_pts, best_pts)
    np.fill_diagonal(d, 1.0)
    min_dist = np.min(d)
    if min_dist < 1e-9:
        # This shouldn't happen, but if it does, add small perturbation
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-9:
                    best_pts[j] += np.random.randn(3) * 1e-6
                    best_pts[j] /= np.linalg.norm(best_pts[j])
    
    # Ensure finite
    if not np.all(np.isfinite(best_pts)):
        # Fallback to initial
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
