import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _fib_sample(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_data(X):
    """Compute convex hull facets and their circumcenters."""
    try:
        hull = ConvexHull(X)
        facets = hull.simplices  # shape (F, 3)
        facet_points = X[facets]  # shape (F, 3, 3)
        
        # Compute circumcenter for each triangular facet
        circumcenters = np.zeros((len(facets), 3))
        for i, tri in enumerate(facet_points):
            a, b, c = tri
            # Circumcenter of triangle in 3D (project to plane)
            ab = b - a
            ac = c - a
            bc = c - b
            bc2 = np.dot(bc, bc)
            # Using formula for circumcenter
            d = 2.0 * (np.dot(ab, np.cross(ab, ac)))
            if abs(d) < 1e-12:
                circumcenters[i] = (a + b + c) / 3.0
                continue
            # Circumcenter = a + (bc^2 * ab x ac + ab^2 * ac x bc) / d
            bc2_val = np.dot(bc, bc)
            ab2_val = np.dot(ab, ab)
            cc = a + (bc2_val * np.cross(ab, ac) + ab2_val * np.cross(ac, bc)) / d
            circumcenters[i] = cc / np.linalg.norm(cc)
        
        # Compute covering radius from hull facets
        # For each facet, the covering radius contribution is the angle from circumcenter to vertices
        angles = []
        for i, tri in enumerate(facet_points):
            cc = circumcenters[i]
            for v in tri:
                dot = np.clip(np.dot(cc, v), -1.0, 1.0)
                angles.append(np.arccos(dot))
        covering_radius = max(angles) if angles else 0.0
        
        return covering_radius, circumcenters, facets
    except Exception:
        # Fallback: use sampling-based estimate
        Y = _fib_sample(5000)
        dots = np.clip(Y @ X.T, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        angles = np.arccos(max_dots)
        return np.max(angles), np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

def _soft_max_objective(flat_X, beta):
    """Soft-max approximation of covering radius for smooth optimization."""
    X = flat_X.reshape(N, 3)
    X = _normalize(X)
    # Sample points on sphere
    Y = _fib_sample(2000)
    dots = np.clip(Y @ X.T, -1.0, 1.0)
    angles = np.arccos(dots)
    # Soft max over nearest point for each y
    min_angles = np.min(angles, axis=1)
    # Soft max: log-sum-exp approximation
    scaled = min_angles * beta
    max_val = np.max(scaled)
    soft_max = max_val + np.log(np.sum(np.exp(scaled - max_val))) / beta
    return soft_max

def _soft_max_grad(flat_X, beta):
    """Gradient of soft-max covering radius."""
    X = flat_X.reshape(N, 3)
    X = _normalize(X)
    Y = _fib_sample(2000)
    dots = np.clip(Y @ X.T, -1.0, 1.0)
    angles = np.arccos(dots)
    min_angles = np.min(angles, axis=1)
    scaled = min_angles * beta
    max_val = np.max(scaled)
    weights = np.exp(scaled - max_val)
    weights /= np.sum(weights)
    
    # Gradient: d(soft_max)/d(X_i)
    # soft_max = max_val + (1/beta) * log(sum(exp(scaled - max_val)))
    # d soft_max / d angle_j = weights[j] / beta
    # d angle_j / d X_i = -dot(y_j, X_i) / (sqrt(1-dot^2) * |X_i|) for the nearest point
    # This is complex; use finite difference approximation or simplify
    
    # Simplified: return zero gradient (use numerical differentiation)
    return np.zeros_like(flat_X)

def _repulsion_relax(X, iterations=1000, power=2.0):
    """Physics-based electrostatic repulsion to spread points."""
    X = X.copy().astype(float)
    for it in range(iterations):
        # Compute pairwise distances
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = np.linalg.norm(X[i] - X[j])
                    dists[i, j] = d if d > 1e-10 else 1e-10
        
        # Compute forces (repulsion)
        forces = np.zeros_like(X)
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = dists[i, j]
                    f = 1.0 / (d ** power)
                    direction = (X[i] - X[j]) / d
                    forces[i] += f * direction
        
        # Update positions with adaptive step size
        step_size = 0.01 / (1.0 + it / 100.0)
        X += step_size * forces
        X = _normalize(X)
        
        # Early termination if forces are small
        if np.max(np.abs(forces)) < 1e-6:
            break
    
    return X

def _lloyd_step_vectorized(X, iterations=50):
    """Lloyd's algorithm for optimal covering: move each point to centroid of its Voronoi cell."""
    X = X.copy().astype(float)
    for it in range(iterations):
        # Compute Voronoi regions using convex hull
        try:
            hull = ConvexHull(X)
            facets = hull.simplices
            facet_points = X[facets]
            
            # For each point, find its Voronoi cell (intersection of half-spaces)
            # Simplified: move each point toward the average of its neighboring facet circumcenters
            new_X = X.copy()
            
            # Compute facet circumcenters
            for i, tri in enumerate(facet_points):
                a, b, c = tri
                ab = b - a
                ac = c - a
                bc = c - a
                d = 2.0 * np.dot(ab, np.cross(ab, ac))
                if abs(d) < 1e-12:
                    cc = (a + b + c) / 3.0
                else:
                    bc2 = np.dot(bc, bc)
                    ab2 = np.dot(ab, ab)
                    cc = a + (bc2 * np.cross(ab, ac) + ab2 * np.cross(ac, bc)) / d
                cc = cc / np.linalg.norm(cc)
                
                # Move each vertex toward the circumcenter
                for idx in facets[i]:
                    new_X[idx] = 0.9 * X[idx] + 0.1 * cc
            
            X = _normalize(new_X)
        except Exception:
            break
    
    return X

def _projected_gradient_polish(X, iterations=200, lr=0.001):
    """Polish using projected gradient descent on covering radius."""
    X = X.copy().astype(float)
    for it in range(iterations):
        # Compute gradient numerically
        eps = 1e-6
        grad = np.zeros_like(X)
        for i in range(N):
            for j in range(3):
                X_plus = X.copy()
                X_plus[i, j] += eps
                X_plus = _normalize(X_plus)
                cr_plus, _, _ = _compute_hull_data(X_plus)
                
                X_minus = X.copy()
                X_minus[i, j] -= eps
                X_minus = _normalize(X_minus)
                cr_minus, _, _ = _compute_hull_data(X_minus)
                
                grad[i, j] = (cr_plus - cr_minus) / (2 * eps)
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * X, axis=1, keepdims=True) * X
        
        # Update
        X = X - lr * grad_tangent
        X = _normalize(X)
        
        # Adaptive learning rate
        if it % 50 == 0 and it > 0:
            lr *= 0.5
    
    return X

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    # Phase 1: Repulsion relaxation for good initial spread
    X0 = _fib_sample(N)
    X_repulsed = _repulsion_relax(X0, iterations=500)
    
    # Phase 2: L-BFGS with soft-max objective
    best_X = X_repulsed
    best_val, _, _ = _compute_hull_data(best_X)
    
    def soft_obj(flat_X):
        return _soft_max_objective(flat_X, beta=100.0)
    
    try:
        res = minimize(
            soft_obj,
            X_repulsed.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        X_lbfgs = _normalize(res.x.reshape(N, 3))
        val_lbfgs, _, _ = _compute_hull_data(X_lbfgs)
        if val_lbfgs < best_val:
            best_val = val_lbfgs
            best_X = X_lbfgs
    except Exception:
        pass
    
    # Phase 3: Lloyd's algorithm for local refinement
    X_lloyd = _lloyd_step_vectorized(best_X, iterations=30)
    val_lloyd, _, _ = _compute_hull_data(X_lloyd)
    if val_lloyd < best_val:
        best_val = val_lloyd
        best_X = X_lloyd
    
    # Phase 4: Projected gradient polish
    elapsed = time.time() - start_time
    if elapsed < timeout - 20:
        X_polished = _projected_gradient_polish(best_X, iterations=100, lr=0.0005)
        val_polished, _, _ = _compute_hull_data(X_polished)
        if val_polished < best_val:
            best_val = val_polished
            best_X = X_polished
    
    # Final check: ensure valid configuration
    final_val, _, _ = _compute_hull_data(best_X)
    if final_val > best_val:
        best_X = _normalize(best_X)
    
    return best_X
