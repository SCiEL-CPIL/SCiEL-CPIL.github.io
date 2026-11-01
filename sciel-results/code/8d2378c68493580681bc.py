"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _initial_fib():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_exact(pts):
    """Compute the exact covering radius using convex hull facets."""
    if len(pts) < 4:
        return 180.0 # Fallback, shouldn't happen for N=130
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_radius = 0.0
    for simplex in hull.simplices:
        v1 = pts[simplex[0]]
        v2 = pts[simplex[1]]
        v3 = pts[simplex[2]]
        
        e1 = v2 - v1
        e2 = v3 - v1
        n = np.cross(e1, e2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-15:
            continue
        
        d = np.dot(n, v1)
        dist = abs(d) / n_norm
        
        if dist > 1.0:
            dist = 1.0
        elif dist < -1.0:
            dist = -1.0
        
        angle = np.arccos(min(max(dist, -1.0), 1.0))
        if angle > max_radius:
            max_radius = angle
    
    return np.degrees(max_radius)

def _lloyd_iteration(pts, n_theta=360, n_phi=720):
    """One Lloyd iteration: move each point to the centroid of its Voronoi cell."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    dots = grid @ pts.T
    assignments = dots.argmax(axis=1)
    
    new_pts = pts.copy()
    for i in range(len(pts)):
        mask = assignments == i
        if np.sum(mask) > 0:
            centroid = grid[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_pts[i] = centroid / norm
    
    return new_pts

def _refine_lloyd(pts, start_time, max_time=120.0, max_iters=20):
    """Lloyd-based refinement."""
    current = pts.copy()
    best = pts.copy()
    best_radius = _covering_radius_exact(current)
    
    for it in range(max_iters):
        if time.time() - start_time > max_time - 5.0:
            break
        
        new_pts = _lloyd_iteration(current)
        
        # Check for coincident points and nudge apart
        for i in range(N):
            for j in range(i + 1, N):
                dot = np.dot(new_pts[i], new_pts[j])
                if dot > 0.999999:
                    diff = new_pts[i] - new_pts[j]
                    norm = np.linalg.norm(diff)
                    if norm < 1e-10:
                        diff = np.array([1.0, 0.0, 0.0])
                        norm = 1.0
                    diff /= norm
                    new_pts[i] = new_pts[i] + 0.01 * diff
                    new_pts[j] = new_pts[j] - 0.01 * diff
                    new_pts[i] /= np.linalg.norm(new_pts[i])
                    new_pts[j] /= np.linalg.norm(new_pts[j])
        
        new_radius = _covering_radius_exact(new_pts)
        if new_radius < best_radius:
            current = new_pts
            best = new_pts.copy()
            best_radius = new_radius
        else:
            # If no improvement, try a small blend or break
            break
            
    return best, best_radius

def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    max_time = 120.0
    
    # Start with Fibonacci spiral
    pts = _initial_fib()
    best_pts = pts.copy()
    best_radius = _covering_radius_exact(best_pts)
    
    # Phase 1: Continuous Differentiable Optimization
    # We parameterize points as (x, y, z) and normalize them in the objective.
    # Objective: Smooth approximation of covering radius.
    # We use a grid for the smooth objective to make it differentiable and fast.
    # Grid size: 120x240 is a good balance.
    n_theta = 120
    n_phi = 240
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    def objective_and_grad(params):
        # params: (3*N,)
        p = params.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        p = p / norms
        
        # Compute max dot product for each grid point
        dots = grid @ p.T
        dots = np.clip(dots, -1.0, 1.0)
        
        # Soft-min of dots (which corresponds to soft-max of angles)
        # We want to minimize the max angle, which is equivalent to minimizing the max (1 - cos(angle))
        # Or simply maximizing the min cos(angle).
        # Let's minimize -min_cos.
        # min_cos = min(dots) is non-differentiable.
        # Use log-sum-exp trick for min: -logsumexp(-dots/beta) * beta
        # Actually, we want to minimize the covering radius, which is max angle.
        # max angle = arccos(min dot).
        # Minimizing max angle is equivalent to maximizing min dot.
        # So we want to maximize min(dots).
        # Let's minimize -min(dots).
        # Smooth min: -beta * log(sum(exp(-dots/beta)))
        # This is a smooth lower bound on -min(dots).
        
        beta = 0.05 # Temperature for soft-min
        # To avoid overflow in exp, subtract max
        # We are computing -min(dots) approx.
        # min(dots) = -max(-dots)
        # max(-dots) = beta * log(sum(exp(-dots/beta)))
        # So min(dots) approx = -beta * log(sum(exp(-dots/beta)))
        
        # We want to maximize min(dots), so minimize -min(dots).
        # obj = -min(dots) approx = beta * log(sum(exp(-dots/beta)))
        
        # Gradient of obj wrt p:
        # Let u = -dots/beta.
        # obj = beta * log(sum(exp(u)))
        # d(obj)/d(u) = sum(exp(u)) / sum(exp(u)) = exp(u)/sum(exp(u)) = softmax(u)
        # d(u)/d(p) = -1/beta * grid (since dots = grid @ p.T, d(dots)/d(p) = grid)
        # So d(obj)/d(p) = softmax(u) @ grid / beta
        
        u = -dots / beta
        u_max = np.max(u, axis=1, keepdims=True)
        u_shifted = u - u_max
        exp_u = np.exp(u_shifted)
        sum_exp = np.sum(exp_u, axis=1, keepdims=True)
        softmax_u = exp_u / sum_exp
        
        # obj_value
        obj_val = beta * np.sum(np.log(sum_exp) + u_max) # log(sum(exp(u))) = log(sum(exp(u_shifted))) + u_max
        
        # Gradient
        # d(obj)/d(p) = (softmax_u @ grid) / beta
        # Note: grid shape (M, 3), softmax_u shape (M, N)
        # (softmax_u @ grid) shape (N, 3)
        grad_p = (softmax_u.T @ grid) / beta
        
        # Chain rule for normalization
        # p = p_raw / norm
        # d(p)/d(p_raw) = (I - p p^T) / norm
        # grad_params = sum_i grad_p[i] @ (I - p[i]p[i]^T) / norm[i]
        
        # We compute grad_params directly
        # grad_params[i] = (grad_p[i] - np.dot(grad_p[i], p[i]) * p[i]) / norm[i]
        
        dot_grad_p = np.sum(grad_p * p, axis=1, keepdims=True)
        grad_params = (grad_p - dot_grad_p * p) / norms
        
        return obj_val, grad_params.ravel()
    
    # Run L-BFGS-B
    try:
        res = minimize(
            objective_and_grad,
            pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-9, 'gtol': 1e-6}
        )
        
        opt_pts = res.x.reshape(N, 3)
        opt_pts /= np.linalg.norm(opt_pts, axis=1, keepdims=True)
        
        # Check validity and radius
        # Ensure no coincident points
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(opt_pts[i] - opt_pts[j]) < 1e-6:
                    opt_pts[j] += np.random.randn(3) * 0.01
                    opt_pts[j] /= np.linalg.norm(opt_pts[j])
        
        opt_radius = _covering_radius_exact(opt_pts)
        if opt_radius < best_radius:
            best_pts = opt_pts
            best_radius = opt_radius
            
    except Exception:
        pass # If optimization fails, keep initial points
    
    # Phase 2: Lloyd Refinement
    remaining = max_time - (time.time() - start_time)
    if remaining > 10.0:
        lloyd_pts, lloyd_radius = _refine_lloyd(best_pts, start_time, max_time)
        if lloyd_radius < best_radius:
            best_pts = lloyd_pts
            best_radius = lloyd_radius
    
    # Final normalization and cleanup
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
