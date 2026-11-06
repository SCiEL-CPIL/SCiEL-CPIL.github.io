import numpy as np
import time
from scipy.optimize import minimize

N = 130

def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return pts / norms

def _soft_soft_objective_and_grad(flat_pts, test_pts, beta, gamma):
    """
    Computes the soft-soft covering radius and its gradient.
    flat_pts: (N*3,)
    test_pts: (M, 3)
    """
    pts = _normalize(flat_pts.reshape(N, 3))
    
    # Compute distances d(y, p_i)
    # test_pts: (M, 3), pts: (N, 3)
    dots = test_pts @ pts.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (M, N)
    
    # Inner soft-min: m_gamma(y)
    # m_gamma = -1/gamma * log(sum(exp(-gamma * dists)))
    # For numerical stability, subtract max
    max_d = np.max(dists, axis=1, keepdims=True)
    exp_term = np.exp(-gamma * (dists - max_d))
    S = np.sum(exp_term, axis=1, keepdims=True)
    m_gamma = -1.0 / gamma * (np.log(S) - gamma * max_d) # (M, 1)
    m_gamma = m_gamma.flatten() # (M,)
    
    # Weights w_yi = exp(-gamma * (dists - max_d)) / S
    w = exp_term / S # (M, N)
    
    # Outer soft-max: J
    # J = 1/beta * log(sum(exp(beta * m_gamma)))
    max_m = np.max(m_gamma)
    exp_m = np.exp(beta * (m_gamma - max_m))
    T = np.sum(exp_m)
    J = max_m + np.log(T) / beta
    
    # Weights v_y = exp(beta * (m_gamma - max_m)) / T
    v = exp_m / T # (M,)
    
    # Gradient calculation
    # dJ/dp_k = sum_y v_y * sum_i w_yi * dd_yi/dp_k
    # dd_yi/dp_k = -1/sqrt(1-dot^2) * (y - dot*p_k)
    
    # We need to compute this for all k.
    # This is expensive if done naively.
    # Let's compute the gradient for all points at once.
    
    # grad_pts: (N, 3)
    # For each point k, grad[k] = sum_y v_y * w[y,k] * grad_d[y,k]
    
    # grad_d[y,k] = - (y - dot[y,k]*p_k) / sqrt(1 - dot[y,k]^2)
    
    # Let's compute the term (y - dot*p_k)
    # y: (M, 3), p_k: (N, 3)
    # This is a tensor operation.
    # We can iterate over test points y, or use broadcasting.
    # M can be large (e.g., 10000). N=130.
    # Iterating over M is 10000 * (N*3) ops. ~4 million ops. Fast enough.
    
    grad_pts = np.zeros((N, 3))
    
    # Precompute 1/sqrt(1-dot^2)
    # Avoid division by zero
    denom = np.sqrt(np.maximum(1e-10, 1.0 - dots * dots))
    
    # Loop over test points for gradient accumulation
    # This might be slow in Python if M is very large.
    # Vectorized approach:
    # We want G[k] = sum_y v_y * w[y,k] * (-(test_pts[y] - dots[y,k]*pts[k]) / denom[y,k])
    
    # Let's try to vectorize.
    # term_y_k = v[:, None] * w * (1.0 / denom)
    # term_y_k is (M, N)
    # We need to multiply this by (y - dot*p_k) which is (M, N, 3)
    
    # y_broadcast: (M, 1, 3)
    # p_k_broadcast: (1, N, 3)
    # dot_broadcast: (M, N, 1)
    
    y_b = test_pts[:, np.newaxis, :] # (M, 1, 3)
    p_b = pts[np.newaxis, :, :] # (1, N, 3)
    dot_b = dots[:, :, np.newaxis] # (M, N, 1)
    
    # diff = y - dot * p
    diff = y_b - dot_b * p_b # (M, N, 3)
    
    # coeff = v_y * w_yk / denom_yk
    coeff = (v[:, np.newaxis] * w / denom[:, :, np.newaxis]) # (M, N, 1)
    
    # grad_contrib = coeff * diff
    grad_contrib = coeff * diff # (M, N, 3)
    
    # Sum over y (axis 0)
    grad_pts = np.sum(grad_contrib, axis=0) # (N, 3)
    
    # Note: The gradient of arccos(x) is -1/sqrt(1-x^2).
    # So we have included the negative sign in the formula above?
    # d/dp arccos(y.p) = -1/sqrt(1-(y.p)^2) * y
    # Wait, d/dp (y.p) = y.
    # So grad_d = -y / sqrt(1-dot^2).
    # My formula: -(y - dot*p) / sqrt(1-dot^2).
    # Let's re-verify.
    # d = arccos(y.p).
    # dd/dp = -1/sqrt(1-(y.p)^2) * d(y.p)/dp = -y / sqrt(1-dot^2).
    # However, this is the gradient in R3.
    # We are on the sphere. The effective gradient is the projection.
    # But since we are using the flat vector in the optimizer, we just return this gradient.
    # The optimizer will update the flat vector, and we normalize.
    # This is a standard trick.
    
    # Wait, the term (y - dot*p) is the projection of y onto the tangent space at p.
    # The gradient of the distance on the sphere is indeed proportional to this.
    # Let's check the sign.
    # If we move p towards y, distance decreases.
    # y - dot*p points from p towards y (tangential component).
    # So moving p in the direction of (y - dot*p) decreases distance.
    # So gradient should point in the direction of (y - dot*p).
    # My formula had a minus sign: -(y - dot*p).
    # Let's check: d = arccos(dot).
    # dd/ddot = -1/sqrt(1-dot^2).
    # ddot/dp = y.
    # So dd/dp = -y/sqrt(1-dot^2).
    # This vector -y points from y to origin? No, y is a unit vector.
    # The component of y along p is dot*p.
    # The component orthogonal to p is y - dot*p.
    # -y = -(dot*p + (y - dot*p)) = -dot*p - (y - dot*p).
    # The -dot*p part is radial. The -(y - dot*p) part is tangential.
    # So the tangential part of the gradient is -(y - dot*p)/sqrt(...).
    # This means the gradient points *away* from y?
    # If I move p in the direction of the gradient, I increase the function.
    # If I move p in the direction of -gradient, I decrease the function.
    # So -gradient should point towards y.
    # -gradient_tangential = (y - dot*p)/sqrt(...).
    # This points towards y. Correct.
    # So the gradient returned to the optimizer should be such that moving in that direction increases the cost.
    # So returning `grad_pts` calculated as `sum ... * (-(y - dot*p)/...)` is correct for a minimization algorithm (which moves opposite to gradient).
    
    # Flatten gradient
    grad_flat = grad_pts.flatten()
    
    return J, grad_flat

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initial configuration
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_score = np.inf
    
    # Test points
    # Use a large number of test points for accuracy
    n_test = 20000
    test_points = _fibonacci_spiral(n_test)
    
    # Phase 1: Coarse optimization
    beta = 5.0
    gamma = 5.0
    
    def objective_coarse(flat_pts):
        val, _ = _soft_soft_objective_and_grad(flat_pts, test_points, beta, gamma)
        return val
    
    def grad_coarse(flat_pts):
        _, g = _soft_soft_objective_and_grad(flat_pts, test_points, beta, gamma)
        return g
        
    try:
        res = minimize(
            objective_coarse,
            grad_coarse,
            pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 50, 'gtol': 1e-5, 'maxfun': 100}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_pts = _normalize(res.x.reshape(N, 3))
    except Exception:
        pass
        
    if time.time() > deadline - 30:
        return best_pts

    # Phase 2: Fine optimization
    beta = 20.0
    gamma = 20.0
    
    def objective_fine(flat_pts):
        val, _ = _soft_soft_objective_and_grad(flat_pts, test_points, beta, gamma)
        return val
        
    def grad_fine(flat_pts):
        _, g = _soft_soft_objective_and_grad(flat_pts, test_points, beta, gamma)
        return g

    try:
        res2 = minimize(
            objective_fine,
            grad_fine,
            best_pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 50, 'gtol': 1e-6, 'maxfun': 100}
        )
        if res2.fun < best_score:
            best_score = res2.fun
            best_pts = _normalize(res2.x.reshape(N, 3))
    except Exception:
        pass

    if time.time() > deadline - 10:
        return best_pts

    # Phase 3: Very fine polish
    beta = 50.0
    gamma = 50.0
    
    def objective_vf(flat_pts):
        val, _ = _soft_soft_objective_and_grad(flat_pts, test_points, beta, gamma)
        return val
        
    def grad_vf(flat_pts):
        _, g = _soft_soft_objective_and_grad(flat_pts, test_points, beta, gamma)
        return g

    try:
        res3 = minimize(
            objective_vf,
            grad_vf,
            best_pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 50, 'gtol': 1e-7, 'maxfun': 50}
        )
        if res3.fun < best_score:
            best_score = res3.fun
            best_pts = _normalize(res3.x.reshape(N, 3))
    except Exception:
        pass

    return best_pts
