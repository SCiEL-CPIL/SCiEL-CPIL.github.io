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
            ab = b - a
            ac = c - a
            bc = c - b
            d = 2.0 * (np.dot(ab, np.cross(ab, ac)))
            if abs(d) < 1e-12:
                circumcenters[i] = (a + b + c) / 3.0
                continue
            bc2_val = np.dot(bc, bc)
            ab2_val = np.dot(ab, ab)
            cc = a + (bc2_val * np.cross(ab, ac) + ab2_val * np.cross(ac, bc)) / d
            circumcenters[i] = cc / np.linalg.norm(cc)
        
        # Compute covering radius from hull facets
        angles = []
        for i, tri in enumerate(facet_points):
            cc = circumcenters[i]
            for v in tri:
                dot = np.clip(np.dot(cc, v), -1.0, 1.0)
                angles.append(np.arccos(dot))
        covering_radius = max(angles) if angles else 0.0
        
        return covering_radius, circumcenters, facets
    except Exception:
        Y = _fib_sample(5000)
        dots = np.clip(Y @ X.T, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        angles = np.arccos(max_dots)
        return np.max(angles), np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

def _soft_max_objective_and_grad(flat_X, beta, Y):
    """Soft-max covering radius with analytical gradient."""
    X = flat_X.reshape(N, 3)
    X = _normalize(X)
    
    # dots: (M, N) - dot products of sample points with configuration points
    dots = np.clip(Y @ X.T, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # For each sample point, find the nearest configuration point (min angle)
    min_angles = np.min(angles, axis=1)  # (M,)
    min_idx = np.argmin(angles, axis=1)  # (M,)
    
    # Soft max over sample points
    scaled = min_angles * beta
    max_val = np.max(scaled)
    weights = np.exp(scaled - max_val)
    weights_sum = np.sum(weights)
    soft_max = max_val + np.log(weights_sum) / beta
    
    # Gradient computation
    # d(soft_max)/d(min_angles[m]) = weights[m] / weights_sum
    # d(min_angles[m])/d(X[i,j]) where i = min_idx[m]:
    #   min_angles[m] = arccos(dots[m, i])
    #   d(arccos(d))/dX = -1/sqrt(1-d^2) * d(d)/dX
    #   d(dots[m,i])/dX[i,j] = Y[m,j]
    #   But we need to project onto tangent space of sphere
    
    grad = np.zeros((N, 3))
    # For each sample point m, the nearest point is i = min_idx[m]
    # d(soft_max)/dX[i] = (weights[m]/weights_sum) * d(min_angles[m])/dX[i]
    # d(min_angles[m])/dX[i] = -Y[m] / sqrt(1 - dots[m,i]^2)
    
    for m in range(len(Y)):
        i = min_idx[m]
        d = dots[m, i]
        denom = np.sqrt(np.maximum(1e-12, 1.0 - d * d))
        w = weights[m] / weights_sum
        grad[i] -= w * Y[m] / denom
    
    # Project gradient to tangent space: grad - (grad·X)X
    grad_tangent = grad - np.sum(grad * X, axis=1, keepdims=True) * X
    
    return soft_max, grad_tangent.flatten()

def _repulsion_relax(X, iterations=500, power=2.0):
    """Vectorized physics-based electrostatic repulsion."""
    X = X.copy().astype(float)
    for it in range(iterations):
        # Compute pairwise distances vectorized
        # dists[i,j] = ||X[i] - X[j]||
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
        dists = np.linalg.norm(diff, axis=2)  # (N, N)
        np.fill_diagonal(dists, 1.0)  # Avoid division by zero
        
        # Forces: F[i] = sum_j (1/d[i,j]^power) * (X[i]-X[j])/d[i,j]
        # = sum_j (X[i]-X[j]) / d[i,j]^(power+1)
        inv_d = 1.0 / dists  # (N, N)
        force_weights = inv_d ** (power + 1)  # (N, N)
        np.fill_diagonal(force_weights, 0.0)
        
        # forces[i] = sum_j force_weights[i,j] * (X[i] - X[j])
        # = X[i] * sum_j force_weights[i,j] - sum_j force_weights[i,j] * X[j]
        row_sums = np.sum(force_weights, axis=1, keepdims=True)  # (N, 1)
        forces = X * row_sums - force_weights @ X  # (N, 3)
        
        # Update with adaptive step
        step_size = 0.01 / (1.0 + it / 100.0)
        X = X + step_size * forces
        X = _normalize(X)
        
        if np.max(np.abs(forces)) < 1e-6:
            break
    
    return X

def _lloyd_step(X, iterations=50):
    """Lloyd's algorithm: move each point toward circumcenter of its Voronoi cell."""
    X = X.copy().astype(float)
    for it in range(iterations):
        try:
            hull = ConvexHull(X)
            facets = hull.simplices
            facet_points = X[facets]
            
            # Accumulate circumcenter contributions for each vertex
            new_X = X.copy()
            counts = np.zeros(N)
            
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
                cc_norm = np.linalg.norm(cc)
                if cc_norm > 1e-10:
                    cc = cc / cc_norm
                    for idx in facets[i]:
                        new_X[idx] += cc
                        counts[idx] += 1
            
            # Average and blend
            mask = counts > 0
            new_X[mask] /= counts[mask, np.newaxis]
            new_X = 0.8 * X + 0.2 * new_X
            X = _normalize(new_X)
        except Exception:
            break
    
    return X

def construct():
    start_time = time.time()
    timeout = 140.0
    
    # Phase 1: Repulsion relaxation for good initial spread
    X0 = _fib_sample(N)
    X_repulsed = _repulsion_relax(X0, iterations=500)
    
    best_X = X_repulsed.copy()
    best_val, _, _ = _compute_hull_data(best_X)
    
    elapsed = time.time() - start_time
    if elapsed > timeout - 30:
        return best_X
    
    # Phase 2: L-BFGS with soft-max objective and analytical gradient
    # Use a fixed set of sample points for consistency
    Y_samples = _fib_sample(3000)
    
    def objective_and_grad(flat_X):
        val, grad = _soft_max_objective_and_grad(flat_X, beta=80.0, Y=Y_samples)
        return val, grad
    
    try:
        res = minimize(
            objective_and_grad,
            X_repulsed.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
        )
        X_lbfgs = _normalize(res.x.reshape(N, 3))
        val_lbfgs, _, _ = _compute_hull_data(X_lbfgs)
        if val_lbfgs < best_val:
            best_val = val_lbfgs
            best_X = X_lbfgs.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > timeout - 20:
        return best_X
    
    # Phase 3: Lloyd's algorithm for local refinement
    X_lloyd = _lloyd_step(best_X, iterations=40)
    val_lloyd, _, _ = _compute_hull_data(X_lloyd)
    if val_lloyd < best_val:
        best_val = val_lloyd
        best_X = X_lloyd.copy()
    
    elapsed = time.time() - start_time
    if elapsed > timeout - 15:
        return best_X
    
    # Phase 4: Second L-BFGS pass with higher beta for sharper approximation
    Y_samples2 = _fib_sample(5000)
    
    def objective_and_grad2(flat_X):
        val, grad = _soft_max_objective_and_grad(flat_X, beta=150.0, Y=Y_samples2)
        return val, grad
    
    try:
        res2 = minimize(
            objective_and_grad2,
            best_X.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 300}
        )
        X_lbfgs2 = _normalize(res2.x.reshape(N, 3))
        val_lbfgs2, _, _ = _compute_hull_data(X_lbfgs2)
        if val_lbfgs2 < best_val:
            best_val = val_lbfgs2
            best_X = X_lbfgs2.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > timeout - 10:
        return best_X
    
    # Phase 5: Final Lloyd polish
    X_final = _lloyd_step(best_X, iterations=30)
    val_final, _, _ = _compute_hull_data(X_final)
    if val_final < best_val:
        best_val = val_final
        best_X = X_final.copy()
    
    return _normalize(best_X)
