"""Optimized sphere covering with 130 points via Sinkhorn Divergence Minimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
DEADLINE = time.time() + 140  # 10s safety margin for the 150s limit

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the exact covering radius in radians using ConvexHull.
    This is the ground truth metric for evaluation.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2  # Fallback if hull fails (e.g., points in hemisphere)
    
    max_r = 0.0
    for simplex in hull.simplices:
        p = pts[simplex]
        a, b, c = p[0], p[1], p[2]
        v1 = b - a
        v2 = c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        # Ensure normal points outward
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.dot(x, a)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
    return max_r

def _fibonacci_spiral(n):
    j = np.arange(n) + 0.5
    z = 1.0 - 2.0 * j / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _sinkhorn_gradient(pts, samples, epsilon=0.1, iterations=20):
    """
    Compute the gradient of the Sinkhorn divergence between the empirical measure
    of pts and the uniform measure on samples.
    
    The Sinkhorn divergence is a smooth approximation to the Wasserstein distance.
    Minimizing it encourages the points to be uniformly distributed, which is
    related to minimizing the covering radius.
    
    Args:
        pts: (N, 3) array of points on the sphere.
        samples: (M, 3) array of sample points on the sphere (uniform).
        epsilon: Entropic regularization parameter.
        iterations: Number of Sinkhorn iterations.
    
    Returns:
        Gradient of the Sinkhorn divergence w.r.t. pts.
    """
    M = len(samples)
    N = len(pts)
    
    # Compute cost matrix: C[i, j] = 1 - cos(theta_ij) = 1 - pts[i] @ samples[j]
    # This is a valid cost function for the sphere (chordal distance squared is also common,
    # but angular distance is more directly related to covering radius).
    # Actually, for Sinkhorn, we usually use squared distances. Let's use squared chordal distance:
    # d^2 = 2 - 2 * cos(theta) = ||p - s||^2
    # So C[i, j] = 2 - 2 * (pts[i] @ samples[j])
    
    # To save memory and time, we can compute in chunks, but for N=130 and M~1000, it's small.
    # Let's use M = 1000 for a good approximation of the uniform measure.
    
    # Compute the matrix of dot products
    # P: (N, 3), S: (M, 3)
    # Dots: (N, M)
    dots = pts @ samples.T
    C = 2.0 - 2.0 * dots
    
    # Initialize potentials u and v
    u = np.zeros(N)
    v = np.zeros(M)
    
    # Marginals
    mu = np.ones(N) / N
    nu = np.ones(M) / M
    
    # Sinkhorn iteration
    # We want to solve: min_K <K, C> + epsilon * (KL(K, mu) + KL(K, nu))
    # K = exp(-(C - u - v) / epsilon)
    # Normalize rows and columns.
    
    for _ in range(iterations):
        # Update u
        # K = exp(-(C - u - v) / epsilon)
        # To avoid overflow, we can shift, but for small C and epsilon, it's usually fine.
        # C is in [0, 4]. epsilon is 0.1. exp(-40) is very small.
        # Let's use log-sum-exp style stability if needed, but for now simple exp.
        # Actually, standard Sinkhorn is:
        # u = -epsilon * log( (K^T @ nu) / mu )
        # where K = exp(-(C - u - v)/epsilon)
        
        # Compute K
        K = np.exp(-(C - u[:, None] - v[None, :]) / epsilon)
        
        # Normalize columns to match nu
        # K_col_sums = K.sum(axis=0)
        # v = epsilon * log(nu / K_col_sums)
        # This is equivalent to: v = -epsilon * log(K_col_sums / nu)
        
        col_sums = K.sum(axis=0)
        # Avoid log(0)
        col_sums = np.maximum(col_sums, 1e-12)
        v = -epsilon * np.log(col_sums / nu)
        
        # Normalize rows to match mu
        # K_row_sums = K.sum(axis=1)
        # u = epsilon * log(mu / K_row_sums)
        # This is equivalent to: u = -epsilon * log(K_row_sums / mu)
        
        row_sums = K.sum(axis=1)
        row_sums = np.maximum(row_sums, 1e-12)
        u = -epsilon * np.log(row_sums / mu)
        
    # After convergence, compute the gradient
    # The gradient of the Sinkhorn divergence w.r.t. C is K - mu nu^T (approximately, at optimum)
    # But we need gradient w.r.t. pts.
    # C[i, j] = 2 - 2 * pts[i] @ samples[j]
    # dC/dpts[i] = -2 * samples[j]
    # So grad_pts[i] = sum_j (dL/dC[i,j]) * dC[i,j]/dpts[i]
    # dL/dC[i,j] = K[i,j] - mu[i]*nu[j] (from the dual problem structure, or simply the optimal plan K*)
    
    # Let's recompute K with final u, v
    K = np.exp(-(C - u[:, None] - v[None, :]) / epsilon)
    
    # Optimal plan K*
    # The gradient of the objective w.r.t. C is K* - mu nu^T
    grad_C = K - np.outer(mu, nu)
    
    # grad_pts[i] = sum_j grad_C[i,j] * (-2 * samples[j])
    # grad_pts = -2 * (grad_C @ samples)
    grad_pts = -2.0 * (grad_C @ samples)
    
    return grad_pts

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_pts = _fibonacci_spiral(N)
    best_cr = _covering_radius(best_pts)
    
    # Generate uniform samples on the sphere for the target measure
    # Use Fibonacci spiral for good uniformity
    M = 2000
    samples = _fibonacci_spiral(M)
    
    max_restarts = 5
    iters_per_restart = 200
    
    for restart in range(max_restarts):
        if time.time() > DEADLINE - 10:
            break
            
        # Initialize
        if restart == 0:
            pts = _fibonacci_spiral(N)
            pts += rng.randn(N, 3) * 0.1
        else:
            pts = rng.randn(N, 3)
        
        pts = _normalize(pts)
        current_cr = _covering_radius(pts)
        
        if current_cr < best_cr:
            best_cr = current_cr
            best_pts = pts.copy()
        
        # Sinkhorn Divergence Gradient Descent
        # Use adaptive learning rate
        lr = 0.05
        for it in range(iters_per_restart):
            if time.time() > DEADLINE - 5:
                break
            
            # Compute gradient
            grad = _sinkhorn_gradient(pts, samples, epsilon=0.1, iterations=10)
            
            # Normalize gradient to prevent large steps
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            
            # Gradient step
            pts = pts - lr * grad
            pts = _normalize(pts)
            
            # Evaluate
            current_cr = _covering_radius(pts)
            if current_cr < best_cr:
                best_cr = current_cr
                best_pts = pts.copy()
                
            # Adaptive learning rate: reduce if no improvement
            if current_cr > best_cr * 1.01:
                lr *= 0.5
            else:
                lr = min(lr * 1.05, 0.1) # Slowly increase if improving
            
            # Early stopping if gradient is very small
            if grad_norm < 1e-6:
                break
    
    # Final polish: run a few more iterations on the best candidate
    if time.time() < DEADLINE - 2:
        pts = best_pts.copy()
        lr = 0.01
        for _ in range(50):
            if time.time() > DEADLINE - 1:
                break
            
            grad = _sinkhorn_gradient(pts, samples, epsilon=0.1, iterations=10)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            
            pts = pts - lr * grad
            pts = _normalize(pts)
            
            current_cr = _covering_radius(pts)
            if current_cr < best_cr:
                best_cr = current_cr
                best_pts = pts.copy()
    
    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius(pts)))
