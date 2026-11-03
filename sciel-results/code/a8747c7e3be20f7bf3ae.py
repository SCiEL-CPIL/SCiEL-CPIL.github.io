<<<<<<< SEARCH
def _gradient_descent(pts, max_iter=500, lr=0.01, beta=500.0):
    """
    Gradient descent on the smooth covering radius objective.
    Uses finite differences to compute gradients.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        # Compute current objective
        obj = _soft_covering_radius(pts, beta=beta)
        
        # Compute gradient via finite differences
        grad = np.zeros_like(pts)
        eps = 1e-5
        for i in range(n):
            for d in range(3):
                pts_p = pts.copy()
                pts_p[i, d] += eps
                pts_p[i] = pts_p[i] / np.linalg.norm(pts_p[i])
                obj_p = _soft_covering_radius(pts_p, beta=beta)
                
                pts_m = pts.copy()
                pts_m[i, d] -= eps
                pts_m[i] = pts_m[i] / np.linalg.norm(pts_m[i])
                obj_m = _soft_covering_radius(pts_m, beta=beta)
                
                grad[i, d] = (obj_p - obj_m) / (2 * eps)
        
        # Project gradient onto tangent space
        for i in range(n):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Gradient descent step
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        # Reduce learning rate over time
        lr *= 0.995
        
    return pts
=======
def _gradient_descent(pts, max_iter=500, lr=0.01, beta=500.0, time_limit=None):
    """
    Gradient descent on the smooth covering radius objective.
    Uses finite differences to compute gradients.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    start_time = time.time()
    
    for it in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
            
        # Compute current objective
        obj = _soft_covering_radius(pts, beta=beta)
        
        # Compute gradient via finite differences
        grad = np.zeros_like(pts)
        eps = 1e-5
        for i in range(n):
            for d in range(3):
                pts_p = pts.copy()
                pts_p[i, d] += eps
                pts_p[i] = pts_p[i] / np.linalg.norm(pts_p[i])
                obj_p = _soft_covering_radius(pts_p, beta=beta)
                
                pts_m = pts.copy()
                pts_m[i, d] -= eps
                pts_m[i] = pts_m[i] / np.linalg.norm(pts_m[i])
                obj_m = _soft_covering_radius(pts_m, beta=beta)
                
                grad[i, d] = (obj_p - obj_m) / (2 * eps)
        
        # Project gradient onto tangent space
        for i in range(n):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Gradient descent step
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        # Reduce learning rate over time
        lr *= 0.995
        
    return pts
>>>>>>> REPLACE

