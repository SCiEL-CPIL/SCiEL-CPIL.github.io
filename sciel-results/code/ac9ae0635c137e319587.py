"""Heilbronn triangles: place 11 points to maximize the minimum triangle area."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 135  # safety margin from 150s limit


def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the equilateral triangle area."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    
    min_area = np.min(valid_areas)
    return min_area / TRI_AREA


def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u, v, w) with u+v+w=1 to Cartesian."""
    # Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack([x, y])


def _cartesian_to_barycentric(points):
    """Convert Cartesian to barycentric coordinates."""
    x = points[:, 0]
    y = points[:, 1]
    # w = y / S3H
    w = y / S3H
    # u = 1 - x - w (since x = v + 0.5*w, so v = x - 0.5*w, u = 1 - v - w = 1 - x + 0.5*w - w = 1 - x - 0.5*w)
    # Actually: x = v + 0.5*w => v = x - 0.5*w
    # u = 1 - v - w = 1 - (x - 0.5*w) - w = 1 - x - 0.5*w
    u = 1.0 - x - 0.5 * w
    v = x - 0.5 * w
    return np.column_stack([u, v, w])


def _soft_min_objective_flat(flat_params, alpha=100.0):
    """
    Objective in barycentric coordinates.
    flat_params: 2*N array of (u_i, v_i) for i=0..N-1, where w_i = 1 - u_i - v_i.
    We maximize min area, so minimize -soft_min(areas).
    Add penalty for points outside triangle.
    """
    n = len(flat_params) // 2
    uv = flat_params.reshape(n, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # Penalty for negative barycentric coords (outside triangle)
    neg_u = np.maximum(0.0, -u)
    neg_v = np.maximum(0.0, -v)
    neg_w = np.maximum(0.0, -w)
    penalty = 1000.0 * (np.sum(neg_u**2) + np.sum(neg_v**2) + np.sum(neg_w**2))
    
    # Convert to Cartesian
    x = v + 0.5 * w
    y = S3H * w
    
    # Compute all triangle areas
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return penalty
    
    # Soft minimum: -log(sum(exp(-alpha * areas))) / alpha
    # This is differentiable and approximates the min
    min_val = -np.log(np.sum(np.exp(-alpha * valid_areas))) / alpha
    return -min_val + penalty


def _min_area_objective_flat(flat_params):
    """Non-smooth objective: negative min area plus penalty."""
    n = len(flat_params) // 2
    uv = flat_params.reshape(n, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    neg_u = np.maximum(0.0, -u)
    neg_v = np.maximum(0.0, -v)
    neg_w = np.maximum(0.0, -w)
    penalty = 1000.0 * (np.sum(neg_u**2) + np.sum(neg_v**2) + np.sum(neg_w**2))
    
    x = v + 0.5 * w
    y = S3H * w
    
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return penalty
    
    min_val = np.min(valid_areas) / TRI_AREA
    return -min_val + penalty


def _is_valid(points):
    """Check if all points are inside the triangle."""
    bary = _cartesian_to_barycentric(points)
    return np.all(bary >= -1e-9)


def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    best_points = None
    best_score = -float('inf')
    
    # Phase 1: L-BFGS-B with soft-min objective from multiple random starts in barycentric space
    rng = np.random.RandomState(42)
    
    def random_barycentric_start():
        """Generate random barycentric coordinates inside the triangle."""
        u = rng.random(N)
        v = rng.random(N)
        # Ensure u + v <= 1 by scaling
        s = u + v
        s[s == 0] = 1.0
        scale = np.minimum(1.0, 0.99 / s)
        u = u * scale
        v = v * scale
        return np.column_stack([u, v]).flatten()
    
    n_starts = 15
    
    for s in range(n_starts):
        if elapsed() > DEADLINE - 30:
            break
        
        x0 = random_barycentric_start()
        
        # Use soft-min objective for smoothness
        alpha = 50.0 + 10.0 * s  # vary sharpness
        
        def obj_soft(x):
            return _soft_min_objective_flat(x, alpha=alpha)
        
        try:
            res = minimize(
                obj_soft,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
        except Exception:
            continue
        
        if elapsed() > DEADLINE - 20:
            break
        
        # Convert back to Cartesian
        uv = res.x.reshape(N, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        
        # Check validity
        if np.min(u) < -1e-6 or np.min(v) < -1e-6 or np.min(w) < -1e-6:
            # Clip and re-evaluate
            u = np.maximum(u, 0)
            v = np.maximum(v, 0)
            w = 1.0 - u - v
            if np.min(w) < -1e-6:
                continue
        
        x_cart = v + 0.5 * w
        y_cart = S3H * w
        points = np.column_stack([x_cart, y_cart])
        
        if not _is_valid(points):
            continue
        
        score = _min_area_normalized(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Phase 2: Polish the best candidate with Nelder-Mead in Cartesian space
    if best_points is not None and elapsed() < DEADLINE - 5:
        def cart_objective(flat):
            pts = flat.reshape(N, 2)
            # Penalty for out of bounds
            bary = _cartesian_to_barycentric(pts)
            penalty = 1000.0 * (np.sum(np.maximum(0, -bary[:, 0])**2) +
                               np.sum(np.maximum(0, -bary[:, 1])**2) +
                               np.sum(np.maximum(0, -bary[:, 2])**2))
            min_a = _min_area_normalized(pts)
            return -min_a + penalty
        
        try:
            res = minimize(
                cart_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            polished = res.x.reshape(N, 2)
            if _is_valid(polished):
                score = _min_area_normalized(polished)
                if score > best_score:
                    best_score = score
                    best_points = polished.copy()
        except Exception:
            pass
    
    # Phase 3: Additional L-BFGS-B refinement on the best candidate
    if best_points is not None and elapsed() < DEADLINE - 3:
        bary = _cartesian_to_barycentric(best_points)
        x0 = np.column_stack([bary[:, 0], bary[:, 1]]).flatten()
        
        def obj_refine(x):
            return _soft_min_objective_flat(x, alpha=200.0)
        
        try:
            res = minimize(
                obj_refine,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            uv = res.x.reshape(N, 2)
            u = uv[:, 0]
            v = uv[:, 1]
            w = 1.0 - u - v
            if np.min(u) >= -1e-8 and np.min(v) >= -1e-8 and np.min(w) >= -1e-8:
                x_cart = v + 0.5 * w
                y_cart = S3H * w
                points = np.column_stack([x_cart, y_cart])
                if _is_valid(points):
                    score = _min_area_normalized(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        # Use a simple grid-like configuration
        best_points = np.zeros((N, 2))
        # Place points at vertices and midpoints and center
        best_points[0] = [0.0, 0.0]
        best_points[1] = [1.0, 0.0]
        best_points[2] = [0.5, S3H]
        best_points[3] = [0.25, S3H/4]
        best_points[4] = [0.75, S3H/4]
        best_points[5] = [0.5, S3H/2]
        best_points[6] = [0.125, S3H/8]
        best_points[7] = [0.625, S3H/8]
        best_points[8] = [0.375, 3*S3H/8]
        best_points[9] = [0.875, S3H/8]
        best_points[10] = [0.5, S3H/4]
        # Clip to ensure inside
        for i in range(N):
            best_points[i, 0] = np.clip(best_points[i, 0], 0.001, 0.999)
            best_points[i, 1] = np.clip(best_points[i, 1], 0.001, S3H * 0.999)
        # Ensure inside triangle
        for i in range(N):
            x, y = best_points[i]
            if y > S3H * x:
                best_points[i, 1] = S3H * x * 0.999
            if y > S3H * (1 - x):
                best_points[i, 1] = S3H * (1 - x) * 0.999
    
    return best_points
