import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130
DEADLINE = 145.0  # safety margin before 150s kill
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return max(0.0, DEADLINE - _elapsed())

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius_deg(pts):
    """Compute the covering radius in degrees by checking the convex hull facets."""
    pts = _normalize(pts)
    # Check if all points are in a closed hemisphere (invalid)
    # For a valid covering, points should span the sphere
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.inf
    
    # The covering radius is the maximum circumradius of the hull facets
    # For each facet, we need to check if it contains the origin in its cap
    # and compute the angular distance from origin to the facet plane
    
    max_rad = 0.0
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        # Compute the normal to the facet
        if len(facet_pts) == 3:
            v1 = facet_pts[1] - facet_pts[0]
            v2 = facet_pts[2] - facet_pts[0]
            normal = np.cross(v1, v2)
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-12:
                continue
            normal = normal / n_norm
            
            # Check if origin is on the same side as the center of the facet
            center = np.mean(facet_pts, axis=0)
            # Distance from origin to the plane
            dist = np.dot(center, normal)
            
            # The angular distance from origin to the facet is arccos(dist) if dist > 0
            # but we need the cap that doesn't contain other points
            # For covering radius, we want the maximum distance to the nearest point
            # which corresponds to the circumradius of each facet
            
            # Circumradius of the triangle on the unit sphere
            # The circumcenter in 3D
            a = facet_pts[0]
            b = facet_pts[1]
            c = facet_pts[2]
            
            # Solve for circumcenter
            # |x - a|^2 = |x - b|^2 = |x - c|^2
            # This gives linear equations
            A_mat = 2.0 * np.array([b - a, c - a])
            b_vec = np.array([np.dot(b, b) - np.dot(a, a), np.dot(c, c) - np.dot(a, a)])
            
            try:
                circum = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                circum = a + circum
            except:
                continue
            
            circ_norm = np.linalg.norm(circum)
            if circ_norm < 1e-12:
                # Facet is a great circle, covering radius is 90 degrees
                rad = np.pi / 2.0
            else:
                # The angular distance from origin to circumcenter
                # The circumradius in angular terms
                cos_theta = 1.0 / circ_norm if circ_norm > 1.0 else 1.0
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                rad = np.arccos(cos_theta)
            
            max_rad = max(max_rad, rad)
    
    return np.degrees(max_rad)

def _soft_covering_radius(pts, temperature=0.1):
    """
    Soft approximation of covering radius using log-sum-exp.
    We sample points on the sphere and compute the max distance to nearest point.
    """
    pts = _normalize(pts)
    
    # Sample many points on the sphere to estimate the covering radius
    n_samples = 5000
    rng = np.random.RandomState(42)
    # Uniform sampling on sphere
    z = rng.uniform(-1, 1, n_samples)
    phi = rng.uniform(0, 2*np.pi, n_samples)
    r = np.sqrt(1 - z*z)
    samples = np.column_stack([r*np.cos(phi), r*np.sin(phi), z])
    
    # Compute distances from each sample to all points
    # dist[i,j] = angular distance from samples[i] to pts[j]
    dots = samples @ pts.T  # (n_samples, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (n_samples, N)
    
    # For each sample, the nearest point distance
    min_angles = np.min(angles, axis=1)  # (n_samples,)
    
    # Soft max using log-sum-exp for smoothness
    # We want to approximate max(min_angles)
    # Use: T * log(sum(exp(min_angles/T)))
    scaled = min_angles / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = temperature * log_sum
    
    return soft_max

def _grad_soft_covering_radius(pts, temperature=0.1):
    """
    Gradient of the soft covering radius w.r.t. points.
    """
    pts = _normalize(pts)
    
    n_samples = 2000
    rng = np.random.RandomState(42)
    z = rng.uniform(-1, 1, n_samples)
    phi = rng.uniform(0, 2*np.pi, n_samples)
    r = np.sqrt(1 - z*z)
    samples = np.column_stack([r*np.cos(phi), r*np.sin(phi), z])
    
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    min_idx = np.argmin(angles, axis=1)
    min_angles = angles[np.arange(n_samples), min_idx]
    
    # Soft max weights
    scaled = min_angles / temperature
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # Gradient: d/dpts[nearest] of the soft max
    # For each sample i, the gradient w.r.t. pts[j] where j is nearest to sample i
    grad = np.zeros_like(pts)
    for i in range(n_samples):
        j = min_idx[i]
        w = weights[i] / temperature
        # d(angles)/d(pts[j]) where angles = arccos(dot(sample_i, pts[j]))
        # d(arccos(x))/dx = -1/sqrt(1-x^2)
        # x = dot(sample_i, pts[j])
        # d(x)/d(pts[j]) = sample_i
        dot_val = dots[i, j]
        sin_val = np.sqrt(np.clip(1 - dot_val*dot_val, 1e-12, 1.0))
        grad[j] -= w * samples[i] / sin_val
    
    # Project gradient to tangent space
    for i in range(N):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return grad

def _spectral_whitening(pts):
    """Apply spectral whitening to make the covariance matrix isotropic."""
    pts = _normalize(pts)
    # Center the points
    center = np.mean(pts, axis=0)
    centered = pts - center
    
    # Compute covariance
    cov = np.cov(centered.T)
    
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)
    
    # Whitening: transform to make covariance identity
    # P = eigenvectors @ diag(1/sqrt(eigenvalues)) @ eigenvectors.T
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    P = eigenvectors @ D_inv_sqrt @ eigenvectors.T
    
    whitened = centered @ P.T
    
    # Re-center and normalize
    whitened = whitened - np.mean(whitened, axis=0)
    whitened = _normalize(whitened)
    
    return whitened

def construct():
    global _start_time
    _start_time = time.time()
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_cr = _covering_radius_deg(best_pts)
    
    # Optimization loop
    temperature = 0.2
    lr = 0.01
    min_lr = 1e-6
    
    while _remaining() > 5.0:
        # Compute gradient of soft covering radius
        grad = _grad_soft_covering_radius(pts, temperature)
        
        # Adaptive step size
        step = lr * grad
        
        # Update points
        new_pts = pts + step
        new_pts = _normalize(new_pts)
        
        # Check if improvement
        new_cr = _covering_radius_deg(new_pts)
        
        if new_cr < best_cr:
            best_pts = new_pts.copy()
            best_cr = new_cr
            pts = new_pts
            # Decrease temperature slowly
            temperature *= 0.99
        else:
            # Rejection: revert and reduce step size
            lr *= 0.95
            if lr < min_lr:
                lr = min_lr
                # Try spectral whitening to escape local minimum
                if _remaining() > 10.0:
                    pts = _spectral_whitening(pts)
                    cr_whitened = _covering_radius_deg(pts)
                    if cr_whitened < best_cr:
                        best_pts = pts.copy()
                        best_cr = cr_whitened
                    lr = 0.01  # Reset learning rate after whitening
        
        # Periodic spectral whitening
        if _remaining() > 15.0 and np.random.random() < 0.05:
            pts_whitened = _spectral_whitening(pts)
            cr_whitened = _covering_radius_deg(pts_whitened)
            if cr_whitened < best_cr:
                best_pts = pts_whitened.copy()
                best_cr = cr_whitened
                pts = pts_whitened
    
    # Final polish: try a few local refinements
    if _remaining() > 2.0:
        for _ in range(10):
            if _remaining() < 1.0:
                break
            grad = _grad_soft_covering_radius(pts, 0.05)
            step = 0.001 * grad
            new_pts = _normalize(pts + step)
            cr_new = _covering_radius_deg(new_pts)
            if cr_new < best_cr:
                best_pts = new_pts.copy()
                best_cr = cr_new
                pts = new_pts
    
    return _normalize(best_pts)
