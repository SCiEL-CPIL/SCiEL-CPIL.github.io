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

def _compute_hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull method.
    Returns the radius in radians.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Vectors for the triangle edges
        AB = B - A
        AC = C - A
        
        # Area of triangle: 0.5 * || AB x AC ||
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Handle degenerate facets
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
            
        # Normal vector for each facet
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward (away from origin)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # Angular circumradius: distance from pole to any vertex (e.g., A)
        # cos(theta) = P @ A
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        return np.max(ang_radii)
    except Exception:
        return np.pi

def _get_grid(G=4000):
    """
    Generate a dense spherical grid for the surrogate objective.
    We use a Fibonacci spiral for good uniformity.
    """
    i = np.arange(G) + 0.5
    z = 1.0 - 2.0 * i / G
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _soft_hull_objective_and_grad(pts, grid_pts, beta=10.0):
    """
    Computes a differentiable surrogate for the covering radius.
    
    The covering radius is R = max_{y in S} min_{i} d(y, p_i).
    We approximate this by sampling the sphere with grid_pts.
    For each grid point y_j, let d_j = min_i d(y_j, p_i).
    We want to minimize a smooth approximation of max_j d_j.
    
    Using log-sum-exp:
    LSE(d) = (1/beta) * log( sum_j exp(beta * d_j) )
    
    d_j is the squared chord distance (or angular distance). 
    Using squared chord distance ||y - p||^2 = 2 - 2 y.p is smooth and monotonic with angular distance.
    Let u_{ij} = 2 - 2 y_j . p_i.
    d_j = min_i u_{ij}.
    
    The min operation is non-differentiable. We use a soft-min:
    soft_min(u) = - (1/beta) log( sum_i exp(-beta * u_i) )
    
    So the objective is:
    F = (1/beta) log( sum_j exp( beta * soft_min_i(u_{ij}) ) )
    
    This is deeply nested and expensive to compute gradients for analytically without care.
    
    Alternative simpler strategy:
    Minimize the sum of squared "worst-case" distances? No, that's L2.
    
    Let's use the "soft-max" of the distances directly.
    For each grid point j, the distance to the nearest point is d_j.
    We approximate max(d_j) using LSE.
    But d_j itself is a min.
    
    Let's simplify:
    Objective: J = sum_j ( min_i (2 - 2 y_j . p_i) )^2 ? No, that minimizes average distance, not max.
    
    Better Surrogate:
    J = max_j ( min_i d(y_j, p_i) )
    
    We can compute the gradient of the "soft max" of the "soft mins".
    
    Let u_{ij} = 2 - 2 y_j . p_i.
    Let d_j = soft_min_i(u_{ij}) = - (1/beta) log( sum_i exp(-beta u_{ij}) ).
    
    Let w_{ij} = exp(-beta u_{ij}) / sum_k exp(-beta u_{ik}). (Weights for the min)
    Then d_j = sum_i w_{ij} u_{ij}.
    
    Now we want M = soft_max_j(d_j) = (1/beta) log( sum_j exp(beta d_j) ).
    Let v_j = exp(beta d_j) / sum_k exp(beta d_k). (Weights for the max)
    Then M = sum_j v_j d_j.
    
    Gradient of M w.r.t p_k:
    dM/dp_k = sum_j v_j * d(d_j)/dp_k
    d(d_j)/dp_k = sum_i w_{ij} * d(u_{ij})/dp_k
    d(u_{ij})/dp_k = -2 y_j if i == k, else 0.
    
    So d(d_j)/dp_k = w_{kj} * (-2 y_j).
    
    Thus:
    grad_M[p_k] = sum_j v_j * w_{kj} * (-2 y_j)
    
    This is O(N * G) for the weights and O(N * G) for the gradient accumulation.
    With N=130, G=4000, this is 520k ops per iteration. Very fast.
    """
    # pts: (N, 3)
    # grid_pts: (G, 3)
    
    # 1. Compute u_{ij} = 2 - 2 y_j . p_i
    # y_j is (G, 3), p_i is (N, 3)
    # dots = grid_pts @ pts.T -> (G, N)
    dots = grid_pts @ pts.T
    u = 2.0 - 2.0 * dots
    
    # 2. Compute w_{ij} (weights for soft min)
    # w_{ij} = exp(-beta * u_{ij}) / sum_k exp(-beta * u_{ik})
    # To avoid overflow, subtract max of each row? 
    # exp(-beta * u) is exp of a negative number (since u > 0 usually), so it's small.
    # Actually, u can be close to 0, so exp(0)=1.
    # If u is large, exp is small.
    # Numerical stability:
    # log_w = -beta * u
    # max_log_w = max(log_w, axis=1, keepdims=True)
    # w = exp(log_w - max_log_w)
    # w = w / sum(w, axis=1, keepdims=True)
    
    log_w = -beta * u
    max_log_w = np.max(log_w, axis=1, keepdims=True)
    w = np.exp(log_w - max_log_w)
    w_sum = np.sum(w, axis=1, keepdims=True)
    w = w / w_sum # (G, N)
    
    # 3. Compute d_j = sum_i w_{ij} u_{ij}
    d = np.sum(w * u, axis=1) # (G,)
    
    # 4. Compute v_j (weights for soft max)
    # v_j = exp(beta * d_j) / sum_k exp(beta * d_k)
    # d_j is positive. beta * d_j can be large.
    log_v = beta * d
    max_log_v = np.max(log_v)
    v = np.exp(log_v - max_log_v)
    v_sum = np.sum(v)
    v = v / v_sum # (G,)
    
    # 5. Compute Gradient
    # grad[p_k] = sum_j v_j * w_{kj} * (-2 y_j)
    # v_j is (G,), w_{kj} is (G, N), y_j is (G, 3)
    # We need to sum over j: v_j * w_{kj} * y_j
    # Let's form a matrix W_v = v[:, None] * w # (G, N)
    # Then grad = -2 * (W_v.T @ grid_pts) # (N, 3)
    
    W_v = v[:, None] * w # (G, N)
    grad = -2.0 * (W_v.T @ grid_pts) # (N, 3)
    
    # Objective value (for monitoring, not strictly needed for descent if we just use grad)
    # M = sum_j v_j d_j
    obj = np.sum(v * d)
    
    return obj, grad

def _riemannian_grad_descent(pts, grid_pts, deadline, lr=0.01, beta=10.0, max_iters=2000):
    """
    Perform Riemannian Gradient Descent.
    """
    best_pts = pts.copy()
    best_obj = _soft_hull_objective_and_grad(pts, grid_pts, beta)[0]
    
    # Adaptive learning rate? Or fixed with decay?
    # Let's use a simple decay.
    current_lr = lr
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        obj, grad = _soft_hull_objective_and_grad(pts, grid_pts, beta)
        
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
            
        # Project gradient onto tangent space
        # grad is in R^3. Tangent space at p is {v : p . v = 0}.
        # p.T @ grad is (N,)
        proj_grad = grad - (pts * (grad * pts).sum(axis=1, keepdims=True))
        
        # Update
        cand = pts - current_lr * proj_grad
        
        # Normalize
        norms = np.linalg.norm(cand, axis=1, keepdims=True)
        cand = cand / (norms + _EPS)
        
        # Check if it's valid (no coincident points? Hull works?)
        # For now, just accept if objective improves or is close.
        # Since this is gradient descent on a smooth proxy, it should generally improve.
        # But the proxy is an approximation. 
        # Let's just take the step if the proxy objective decreased?
        # We already computed obj for the *current* pts.
        # We need the obj for cand? That's another O(N*G) cost.
        # Let's assume the step is good if lr is small.
        
        pts = cand
        
        # Decay learning rate
        if i % 100 == 0:
            current_lr *= 0.95
            
    return best_pts

def _local_polish(pts, deadline):
    """
    Perform a few iterations of facet-based local optimization to refine the GD result.
    """
    best_pts = pts.copy()
    best_r = _compute_hull_radius(pts)
    
    # Simple gradient descent on the hull facets
    step = 0.05
    for _ in range(50):
        if time.time() > deadline:
            break
            
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            M = len(faces)
            
            A = pts[faces[:, 0]]
            B = pts[faces[:, 1]]
            C = pts[faces[:, 2]]
            
            AB = B - A
            AC = C - A
            cross = np.cross(AB, AC)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            valid = area > _EPS
            if not np.any(valid):
                break
                
            normals = cross[valid]
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            centroids = (A[valid] + B[valid] + C[valid]) / 3.0
            signs = np.sum(normals * centroids, axis=1, keepdims=True)
            normals = normals * np.sign(signs)
            normals = normals / (norms + _EPS)
            
            cos_theta = np.sum(normals * A[valid], axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            ang_radii = np.arccos(cos_theta)
            
            curr_r = np.max(ang_radii)
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
                
            # Find worst facet
            worst_idx = np.argmax(ang_radii)
            v_indices = faces[valid][worst_idx]
            pole = normals[worst_idx]
            
            cand = pts.copy()
            for v in v_indices:
                dir_vec = pole - pts[v]
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > _EPS:
                    dir_vec /= dir_norm
                else:
                    dir_vec = 0.0
                cand[v] = pts[v] + step * dir_vec
                cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
                
            # Check improvement
            cand_r = _compute_hull_radius(cand)
            if cand_r < curr_r:
                pts = cand
            else:
                step *= 0.9
        except Exception:
            break
            
    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    # Create a dense spherical grid for the energy function
    # Use a Fibonacci spiral with more points for the grid
    G = 4000
    grid_pts = _get_grid(G)
    
    # Initial configuration
    pts = _initial_fib()
    
    # Phase 1: Riemannian Gradient Descent on the soft surrogate
    # This is fast and smooth.
    gd_deadline = deadline - 20.0 # Leave 20s for polishing
    pts = _riemannian_grad_descent(pts, grid_pts, gd_deadline, lr=0.05, beta=20.0)
    
    # Phase 2: Local polish using the exact hull metric
    if time.time() < deadline - 5.0:
        best_pts, best_r = _local_polish(pts, deadline)
        pts = best_pts
        
    # Final normalization
    pts = np.array(pts, dtype=float)
    pts /= (np.linalg.norm(pts, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(pts)) or pts.shape != (N, 3):
        return _initial_fib()
        
    return pts
