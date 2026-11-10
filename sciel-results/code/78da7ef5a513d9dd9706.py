import numpy as np
import time

N = 50

def _score(centers):
    """Compute the radius score: min of pairwise distances/2 and edge distances."""
    n = len(centers)
    # Edge distances: min over each center of distance to each edge
    # Square is [-0.5, 0.5]^2
    edge_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    min_edge = np.min(edge_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_edge, min_pair)

def _neg_score(centers_flat):
    centers = centers_flat.reshape(-1, 2)
    if not np.all(np.isfinite(centers)):
        return 1e10
    if not (np.all(centers >= -0.5) and np.all(centers <= 0.5)):
        return 1e10
    return -_score(centers)

def _gradient(centers_flat):
    centers = centers_flat.reshape(-1, 2)
    n = len(centers)
    
    # Edge distances
    edge_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    min_edge = np.min(edge_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_full = np.min(dists)
    min_pair = min_pair_full / 2.0
    
    grad = np.zeros_like(centers_flat)
    
    if min_edge < min_pair:
        # Gradient comes from edge constraints
        for i in range(n):
            if edge_dists[i] <= min_edge + 1e-12:
                d_left = 0.5 - centers[i, 0]
                d_right = centers[i, 0] + 0.5
                d_bottom = 0.5 - centers[i, 1]
                d_top = centers[i, 1] + 0.5
                
                min_d = min(d_left, d_right, d_bottom, d_top)
                
                if d_left <= min_d + 1e-12:
                    grad[2*i] = -1.0
                if d_right <= min_d + 1e-12:
                    grad[2*i] = 1.0
                if d_bottom <= min_d + 1e-12:
                    grad[2*i+1] = -1.0
                if d_top <= min_d + 1e-12:
                    grad[2*i+1] = 1.0
    else:
        # Gradient comes from pairwise constraints
        min_dist_threshold = min_pair_full + 1e-10
        for i in range(n):
            for j in range(i+1, n):
                if dists[i, j] <= min_dist_threshold:
                    vec = centers[i] - centers[j]
                    d_vec = vec / (2.0 * dists[i, j])
                    grad[2*i:2*i+2] += d_vec
                    grad[2*j:2*j+2] -= d_vec
    
    return -grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    rng = np.random.RandomState(42)
    
    # Initial guess: Hexagonal lattice scaled to fit
    # This provides a better starting point than a square grid
    s = 1.0
    points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
    points = np.array(points)
    
    # Select N points closest to origin
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    init_positions = points[indices].copy()
    
    # Scale to fit in [-0.5, 0.5]^2
    max_abs = np.max(np.abs(init_positions))
    if max_abs > 0:
        scale = 0.49 / max_abs  # Slight margin to avoid boundary issues initially
        init_positions = init_positions * scale
    
    best_centers = init_positions.copy()
    best_score = _score(best_centers)
    
    # Strategy: Multiple restarts with L-BFGS optimization
    # We will use the hex lattice as the primary restart, but also try
    # slightly perturbed versions to escape local minima.
    
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            centers = best_centers.copy()
        elif restart == 1:
            # Jittered hex lattice
            jitter = 0.01
            centers = best_centers + rng.uniform(-jitter, jitter, size=best_centers.shape)
            centers = np.clip(centers, -0.499, 0.499)
        elif restart == 2:
            # Square grid (fallback/alternative structure)
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            centers = grid[:N].copy()
        else:
            # Random perturbations of the best solution found so far
            jitter = 0.05 / (restart + 1)
            centers = best_centers + rng.uniform(-jitter, jitter, size=best_centers.shape)
            centers = np.clip(centers, -0.499, 0.499)
        
        # Optimize using L-BFGS-B
        try:
            from scipy.optimize import minimize
            result = minimize(
                _neg_score,
                centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2*N),
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100},
                jac=_gradient
            )
            
            if result.fun < -best_score:
                new_score = -result.fun
                if new_score > best_score:
                    best_score = new_score
                    best_centers = result.x.reshape(-1, 2).copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Final high-precision polish on the best solution found
    if time.time() < deadline - 2.0:
        try:
            from scipy.optimize import minimize
            result = minimize(
                _neg_score,
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.49999, 0.49999)] * (2*N),
                options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100},
                jac=_gradient
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_centers = result.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    # Ensure strict validity: clip to slightly inside the boundary
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers
