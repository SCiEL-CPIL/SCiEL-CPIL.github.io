import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_hull_radius(pts):
    """
    Compute the true covering radius using the convex hull.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Compute normals for each face
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        # Filter degenerate
        valid = norms.flatten() > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid] / (norms[valid] + _EPS)
        
        # Ensure outward direction
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        
        # The pole is the normalized normal.
        # Distance is arccos(pole . vertex)
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        return np.max(radii)
    except Exception:
        return np.pi

def _potential_and_grad(pts, beta):
    """
    Compute the potential energy and gradient for the exponential repulsive potential.
    Potential: sum_{i<j} exp(-beta * d_ij)
    Gradient: sum_{j!=i} -beta * exp(-beta * d_ij) * (p_j - p_i) / d_ij
    Note: On the sphere, the gradient of the distance d_ij with respect to p_i 
    is roughly (p_j - p_i)/d_ij projected onto the tangent space.
    However, since we normalize after the step, we can use the vector difference directly 
    as a proxy for the direction of increasing distance, scaled by the derivative of the potential.
    
    Actually, let's be more precise.
    f(d) = exp(-beta * d)
    df/dd = -beta * exp(-beta * d)
    d d_ij / d p_i = (p_i - p_j) / d_ij  (for Euclidean, but on sphere it's similar for small d)
    Wait, p_i and p_j are on the sphere.
    d_ij = arccos(p_i . p_j)
    grad_{p_i} d_ij = - (p_j - (p_i.p_j)p_i) / sqrt(1 - (p_i.p_j)^2)
    This is the projection of p_j onto the tangent plane at p_i, normalized by sin(d_ij).
    
    Let's use a simpler approximation that works well for optimization on spheres:
    Move p_i away from p_j proportional to the repulsive force.
    Force vector F_ij = -beta * exp(-beta * d_ij) * (p_j - p_i)
    This is not exactly the gradient, but it points in the correct direction on the sphere.
    To be more rigorous, we can compute the gradient of the potential with respect to the Cartesian coordinates,
    then project onto the tangent space.
    
    Let's stick to the vector difference method which is robust:
    grad_i = sum_{j!=i} -beta * exp(-beta * d_ij) * (p_j - p_i)
    Then step: p_i_new = p_i + lr * grad_i
    Then normalize.
    """
    # Compute pairwise distances
    dots = pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Compute weights: w_ij = beta * exp(-beta * d_ij)
    # We want to move i away from j.
    # The force on i due to j is proportional to -grad_{p_i} exp(-beta d_ij)
    # grad_{p_i} exp(-beta d_ij) = -beta exp(-beta d_ij) grad_{p_i} d_ij
    # So force = beta exp(-beta d_ij) grad_{p_i} d_ij
    # grad_{p_i} d_ij points from j to i (roughly).
    # Let's use the vector (p_i - p_j) as the direction.
    
    weights = beta * np.exp(-beta * dists)
    weights[~mask] = 0.0
    
    # Sum of weighted vectors: sum_j w_ij (p_i - p_j)
    # = p_i * sum_j w_ij - sum_j w_ij p_j
    # sum_j w_ij is row sum of weights
    # sum_j w_ij p_j is matrix product weights @ pts
    
    row_sums = weights.sum(axis=1, keepdims=True)
    weighted_sum = weights @ pts
    
    # Gradient direction for each point i: row_sums[i] * p_i - weighted_sum[i]
    grad = row_sums * pts - weighted_sum
    
    # Potential value
    pot = 0.5 * weights.sum()
    
    return pot, grad

def _optimize(pts, beta_start, beta_end, n_iters, lr_start, lr_end, deadline):
    """
    Run gradient descent with annealing beta and learning rate.
    """
    pts = pts.copy()
    best_pts = pts.copy()
    best_pot = np.inf
    
    # Precompute time step
    time_per_iter = 0.01 # rough estimate, will adjust
    
    for i in range(n_iters):
        if time.time() > deadline:
            break
            
        # Anneal beta and lr
        t = i / max(n_iters - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        lr = lr_start * (lr_end / lr_start) ** t
        
        pot, grad = _potential_and_grad(pts, beta)
        
        # Update
        pts = pts + lr * grad
        
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / (norms + _EPS)
        
        if pot < best_pot:
            best_pot = pot
            best_pts = pts.copy()
            
    return best_pts

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    best_pts = _initial_fib()
    best_r = _compute_hull_radius(best_pts)
    
    # Strategy: Multi-restart with different initializations and annealing schedules
    # The key is to use a smooth potential that allows escaping local minima better than Lloyd's
    # We use a range of beta values to balance global exploration (low beta) and local refinement (high beta)
    
    n_restarts = 15
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        else:
            # Mix of random and perturbed Fibonacci
            if restart % 2 == 0:
                pts = _random_init(rng)
            else:
                pts = _initial_fib()
                # Add noise
                noise = rng.standard_normal((N, 3)) * 0.5
                pts = pts + noise
                pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Determine time budget for this restart
        # Leave 15s for final evaluation and safety
        remaining = deadline - time.time() - 15.0
        if remaining < 5.0:
            continue
            
        # Annealing parameters
        # Start with low beta (broad exploration) and end with high beta (fine structure)
        beta_start = 2.0
        beta_end = 20.0
        
        lr_start = 0.05
        lr_end = 0.001
        
        # Estimate number of iterations based on time
        # Each iteration involves O(N^2) operations. N=130, so N^2 ~ 17000.
        # This is fast. We can do thousands of iterations.
        # Let's allocate time dynamically.
        iter_deadline = time.time() + remaining
        
        # We don't know exact time per iter, so we run until deadline
        # But we need to pass n_iters to the loop. We can just make it large and break on time.
        n_iters = 10000
        
        optimized_pts = _optimize(pts, beta_start, beta_end, n_iters, lr_start, lr_end, iter_deadline)
        
        # Evaluate
        curr_r = _compute_hull_radius(optimized_pts)
        if curr_r < best_r:
            best_r = curr_r
            best_pts = optimized_pts.copy()
            
    # Final polish: Use a very high beta for local refinement
    if time.time() < deadline - 5.0:
        final_deadline = time.time() + 10.0
        # High beta to lock in structure
        polished_pts = _optimize(best_pts, 50.0, 100.0, 5000, 0.01, 0.0001, final_deadline)
        final_r = _compute_hull_radius(polished_pts)
        if final_r < best_r:
            best_r = final_r
            best_pts = polished_pts.copy()
            
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
