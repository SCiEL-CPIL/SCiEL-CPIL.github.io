import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 130
PROBE_COUNT = 2000

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _fibonacci_spiral(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(points)

def _covering_radius_deg(pts):
    """Compute exact covering radius in degrees via convex hull."""
    try:
        hull = ConvexHull(pts)
        max_rad = 0.0
        for simplex in hull.simplices:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Circumcenter of triangle on sphere
            # n = (a-b) x (a-c)
            n = np.cross(a - b, a - c)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            centroid = (a + b + c) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
            dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
            rad = np.arccos(dot_val)
            if rad > max_rad:
                max_rad = rad
        return np.degrees(max_rad)
    except Exception:
        return 180.0

def _smooth_objective_and_grad(x_flat, probes, T_in, T_out):
    """
    x_flat: (3*N,)
    probes: (M, 3)
    Returns: (value, grad_flat)
    """
    N = len(x_flat) // 3
    pts = x_flat.reshape(-1, 3)
    
    # Vectorized distance calculation
    # dists_sq[y, i] = ||probes[y] - pts[i]||^2
    # Use (a-b)^2 = a^2 - 2ab + b^2
    # Since on sphere, a^2=1, b^2=1, so d^2 = 2 - 2 a.b
    dots = np.dot(probes, pts.T) # (M, N)
    dists_sq = 2.0 - 2.0 * dots
    
    # Clip for numerical stability
    dists_sq = np.clip(dists_sq, 1e-10, 4.0)
    
    # Inner softmin
    # exp(-dists_sq / T_in)
    exp_inner = np.exp(-dists_sq / T_in) # (M, N)
    Q = np.sum(exp_inner, axis=1) # (M,)
    
    # m_y = -T_in * ln(Q_y)
    # Avoid log(0)
    Q_safe = np.maximum(Q, 1e-15)
    m = -T_in * np.log(Q_safe) # (M,)
    
    # Outer softmax max
    # exp(m / T_out)
    exp_outer = np.exp(m / T_out) # (M,)
    S = np.sum(exp_outer)
    S_safe = max(S, 1e-15)
    
    value = T_out * np.log(S_safe)
    
    # Gradient
    # dJ/dp_i = sum_y (e_y / S) * (2 / Q_y) * sum_k w_yk (p_k - y) ... wait
    # From derivation:
    # grad_m_y = (2 / Q_y) * V_y where V_y = sum_k w_yk (p_k - y)
    # grad_J = (1/S) * sum_y e_y * grad_m_y
    
    grad_pts = np.zeros_like(pts)
    
    # We need to accumulate gradients for all points
    # For each probe y, we have a vector V_y of shape (3*N) effectively?
    # V_y = sum_k w_yk (p_k - y)
    # This can be computed as: 
    # V_y = (sum_k w_yk * p_k) - y * (sum_k w_yk)
    # sum_k w_yk = Q_y
    # sum_k w_yk * p_k = exp_inner[y] @ pts
    
    # So V_y = exp_inner[y] @ pts - Q[y] * probes[y]
    # grad_contrib_y = (e_y / S) * (2 / Q_y) * V_y
    # This is a (3,) vector added to all p_k? No.
    # V_y is a vector sum. But the gradient w.r.t p_k is the k-th term in the sum?
    # Let's re-verify the gradient chain.
    # m_y = -T_in ln (sum_k exp(-d_yc_k^2/T_in))
    # d m_y / d p_k = -T_in * (1/Q_y) * d Q_y / d p_k
    # d Q_y / d p_k = exp(-d_yc_k^2/T_in) * d(-d_yc_k^2/T_in)/d p_k
    # d(-d^2/T_in)/d p_k = (-1/T_in) * 2 (p_k - y_k) / d_yc_k * d_yc_k ?? No.
    # d/dp (p-y)^2 = 2(p-y).
    # So d Q_y / d p_k = w_yk * (-2/T_in) * (p_k - y)
    # Thus d m_y / d p_k = -T_in * (1/Q_y) * w_yk * (-2/T_in) * (p_k - y)
    # = (2 w_yk / Q_y) * (p_k - y)
    
    # So the gradient of m_y w.r.t p_k is (2 w_yk / Q_y) * (p_k - y).
    # The gradient of J w.r.t p_k is sum_y (e_y / S) * (2 w_yk / Q_y) * (p_k - y).
    
    # This means for each probe y, we iterate over all k and add to grad_pts[k].
    # This is O(M * N).
    
    # Optimization:
    # grad_pts[k] = sum_y coeff_y * (2 w_yk / Q_y) * (p_k - y)
    # = p_k * sum_y (coeff_y * 2 w_yk / Q_y) - y * sum_y (coeff_y * 2 w_yk / Q_y)
    # Let C_yk = (e_y / S) * (2 w_yk / Q_y)
    # grad_pts[k] = p_k * (sum_y C_yk) - (sum_y C_yk * y)
    
    # We can compute C matrix (M, N)
    # C = exp_outer[:, None] / S_safe * 2.0 * exp_inner / Q_safe[:, None]
    
    C = (exp_outer / S_safe)[:, None] * (2.0 * exp_inner / Q_safe[:, None])
    
    # sum_y C_yk
    sum_C = np.sum(C, axis=0) # (N,)
    
    # sum_y C_yk * y
    # C is (M, N), probes is (M, 3)
    # We want (N, 3)
    # sum_y C_yk * probes[y] = (C.T @ probes)
    sum_Cy = C.T @ probes # (N, 3)
    
    grad_pts = sum_C[:, None] * pts - sum_Cy
    
    return value, grad_pts.flatten()

def construct():
    start_time = time.time()
    max_time = 110.0  # Safety margin
    
    # Initialize
    pts = _fibonacci_spiral(N)
    x0 = pts.flatten()
    
    # Probe points
    np.random.seed(42)
    # Use Fibonacci for probes too for uniformity
    probes = _fibonacci_spiral(PROBE_COUNT)
    
    T_in = 0.05
    T_out = 0.1
    
    # Wrapper for L-BFGS-B
    def obj(x):
        val, _ = _smooth_objective_and_grad(x, probes, T_in, T_out)
        return val
        
    def grad(x):
        _, g = _smooth_objective_and_grad(x, probes, T_in, T_out)
        return g
        
    # L-BFGS-B
    try:
        result = minimize(
            obj,
            x0,
            jac=grad,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        pts = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    pts = _normalize(pts)
    
    # Check time
    if time.time() - start_time > max_time:
        return pts
    
    # Polish with local search on exact covering radius
    # Simple Lloyd-like step for the worst facets
    for _ in range(20):
        if time.time() - start_time > max_time:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        max_rad = 0.0
        worst_facet = None
        worst_circ = None
        
        for simplex in hull.simplices:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            n = np.cross(a - b, a - c)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            centroid = (a + b + c) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
            dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
            rad = np.arccos(dot_val)
            
            if rad > max_rad:
                max_rad = rad
                worst_facet = simplex
                worst_circ = n
                
        if worst_facet is None:
            break
            
        # Move vertices towards circumcenter
        step = 0.05
        for idx in worst_facet:
            v = pts[idx]
            grad = worst_circ - v
            grad = grad / np.linalg.norm(grad + 1e-10)
            new_v = v + step * grad
            pts[idx] = new_v / np.linalg.norm(new_v)
            
        pts = _normalize(pts)
        
    return pts
