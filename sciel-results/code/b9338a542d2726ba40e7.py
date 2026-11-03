import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2

# Precompute indices for all unique triples (i, j, k) with i < j < k
# There are C(11, 3) = 165 such triples
_triples = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_triples = np.array(_triples)
_N_TRIPLES = len(_triples)

def _min_triangle_area(pts):
    """
    pts: array of shape (N, 2) with Cartesian coordinates.
    Returns the minimum area of a triangle formed by any 3 points.
    """
    pi = pts[_triples[:, 0]]
    pj = pts[_triples[:, 1]]
    pk = pts[_triples[:, 2]]
    
    dx1 = pi[:, 0] - pk[:, 0]
    dy1 = pi[:, 1] - pk[:, 1]
    dx2 = pj[:, 0] - pk[:, 0]
    dy2 = pj[:, 1] - pk[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _is_inside_triangle(pts):
    """
    Check if all points are inside or on the boundary of the equilateral triangle
    with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    x = pts[:, 0]
    y = pts[:, 1]
    
    cond1 = y >= -1e-9
    cond2 = SQRT3 * x - y >= -1e-9
    cond3 = -SQRT3 * x - y + SQRT3 >= -1e-9
    
    return np.all(cond1) and np.all(cond2) and np.all(cond3)

def _project_to_triangle(pts):
    """
    Project points strictly inside the triangle.
    Uses barycentric coordinates for robust projection.
    """
    # Vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, H])
    
    # Convert to barycentric coordinates
    # P = u*A + v*B + w*C, u+v+w=1
    # P_x = v + 0.5w
    # P_y = H*w
    # w = P_y / H
    # v = P_x - 0.5w
    # u = 1 - v - w
    
    w = pts[:, 1] / H
    v = pts[:, 0] - 0.5 * w
    u = 1.0 - v - w
    
    # Clip to [0, 1]
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    
    # Normalize if sum != 1 (due to clipping)
    sums = u + v + w
    # Avoid division by zero (should not happen if input is not all far away, but just in case)
    sums[sums == 0] = 1.0
    u /= sums
    v /= sums
    w /= sums
    
    # Convert back to Cartesian
    x = v + 0.5 * w
    y = H * w
    
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def _is_time_up():
        return time.time() > deadline

    best_area = 0.0
    best_pts = None
    
    # Objective function: Maximize smooth approximation of min area
    # We minimize -f(pts) where f is log-sum-exp of areas
    # f(x) = (1/beta) * log(sum(exp(beta * a_i)))
    # As beta -> inf, f(x) -> max(a_i). Wait, we want min area.
    # Let's define g(x) = -min(a_i). We want to minimize g(x).
    # Smooth approximation of min: min(a_i) ~= -(1/beta) * log(sum(exp(-beta * a_i)))
    # So we minimize h(x) = (1/beta) * log(sum(exp(-beta * a_i)))
    
    def objective_and_grad(pts_flat):
        if _is_time_up():
            return 1e9, np.zeros_like(pts_flat)
        
        pts = pts_flat.reshape(N, 2)
        
        # Compute areas
        pi = pts[_triples[:, 0]]
        pj = pts[_triples[:, 1]]
        pk = pts[_triples[:, 2]]
        
        dx1 = pi[:, 0] - pk[:, 0]
        dy1 = pi[:, 1] - pk[:, 1]
        dx2 = pj[:, 0] - pk[:, 0]
        dy2 = pj[:, 1] - pk[:, 1]
        
        cross = dx1 * dy2 - dy1 * dx2
        areas = 0.5 * np.abs(cross)
        
        # Smooth min
        beta = 50.0 # Temperature for log-sum-exp
        # To avoid overflow, subtract max
        max_a = np.max(areas)
        # We want min(areas).
        # min(a) = - max(-a)
        # -max(-a) = - (1/beta) log sum exp(beta * (-a))
        # Let z = -beta * areas
        # f = - (1/beta) log sum exp(z)
        
        z = -beta * areas
        max_z = np.max(z)
        sum_exp = np.sum(np.exp(z - max_z))
        
        # Objective value
        val = -(max_z + np.log(sum_exp)) / beta
        
        # Gradient
        # d/da_i [ -(1/beta) log sum exp(-beta * a_j) ]
        # = -(1/beta) * (1/sum_exp) * sum_j exp(-beta a_j) * d/da_i (-beta a_j)
        # For i=j: term is exp(-beta a_i) * (-beta)
        # For i!=j: term is 0
        # So d/da_i f = -(1/beta) * (1/sum_exp) * exp(-beta a_i) * (-beta)
        # = exp(-beta a_i) / sum_exp
        
        # Let p_i = exp(z_i) / sum_exp (normalized probabilities)
        # Note: sum_exp here is sum(exp(z - max_z)).
        # exp(z_i) = exp(z_i - max_z) * exp(max_z)
        # p_i = exp(z_i - max_z) / sum_exp
        
        p = np.exp(z - max_z) / sum_exp
        
        # Gradient with respect to area a_i is p_i
        # Now we need gradient with respect to pts.
        # a_i = 0.5 * |cross_i|
        # da_i/dpts = 0.5 * sign(cross_i) * dcross_i/dpts
        
        # dcross/dpts_k:
        # cross = (x_i - x_k)(y_j - y_k) - (y_i - y_k)(x_j - x_k)
        # d/dx_i: (y_j - y_k)
        # d/dy_i: -(x_j - x_k)
        # d/dx_j: -(y_i - y_k)
        # d/dy_j: (x_i - x_k)
        # d/dx_k: (x_i - x_j)
        # d/dy_k: -(y_i - y_j)
        
        # Let's compute gradients for each triple and accumulate
        # grad_pts = zeros((N, 2))
        
        # Vectorized accumulation is tricky. Let's use loops for gradient or sparse accumulation.
        # Given N=11 and 165 triples, a simple loop might be okay if optimized, 
        # but vectorized is better.
        
        # Let's compute the contribution of each triple to the gradient of the objective.
        # grad_obj = sum_i p_i * (da_i/dpts)
        
        # da_i/dx_i = 0.5 * sign(cross_i) * (y_j - y_k)
        # da_i/dy_i = 0.5 * sign(cross_i) * (-(x_j - x_k))
        # ...
        
        signs = np.sign(cross)
        # Handle zero cross (degenerate). If cross=0, area=0. Gradient is undefined but subgradient exists.
        # For optimization, we can treat sign(0) as 0 or random. Let's use 0 to avoid jumps.
        # Actually, if area is 0, p_i will be large (if it's the min). 
        # The gradient of |x| at 0 is undefined. 
        # We can use a smoothed absolute value or just assume generic position.
        # For now, signs = np.where(cross == 0, 1.0, signs) to ensure some direction? 
        # No, let's stick to standard sign. If stuck at 0, we rely on restarts.
        
        factor = 0.5 * signs * p
        
        # Accumulate gradients
        grad_pts = np.zeros((N, 2))
        
        # We can use np.add.at for efficient accumulation
        # Indices: _triples[:, 0], _triples[:, 1], _triples[:, 2]
        
        # Contributions to point i (index 0 in triple)
        # d/dx_i: factor * (y_j - y_k)
        # d/dy_i: factor * (-(x_j - x_k))
        
        # Let's extract coordinates again for gradient calc
        # pi, pj, pk are already defined
        
        # dx_jk = pj[:, 0] - pk[:, 0]
        # dy_jk = pj[:, 1] - pk[:, 1]
        # dx_ik = pi[:, 0] - pk[:, 0]
        # dy_ik = pi[:, 1] - pk[:, 1]
        # dx_ij = pi[:, 0] - pj[:, 0]
        # dy_ij = pi[:, 1] - pj[:, 1]
        
        # Grad for pi (i):
        # gx_i = factor * (y_j - y_k)
        # gy_i = factor * (-(x_j - x_k))
        
        # Grad for pj (j):
        # gx_j = factor * (-(y_i - y_k))
        # gy_j = factor * (x_i - x_k)
        
        # Grad for pk (k):
        # gx_k = factor * (x_i - x_j)
        # gy_k = factor * (-(y_i - y_j))
        
        gx_i = factor * (pj[:, 1] - pk[:, 1])
        gy_i = factor * (-(pj[:, 0] - pk[:, 0]))
        
        gx_j = factor * (-(pi[:, 1] - pk[:, 1]))
        gy_j = factor * (pi[:, 0] - pk[:, 0])
        
        gx_k = factor * (pi[:, 0] - pj[:, 0])
        gy_k = factor * (-(pi[:, 1] - pj[:, 1]))
        
        # Accumulate
        np.add.at(grad_pts, (_triples[:, 0], 0), gx_i)
        np.add.at(grad_pts, (_triples[:, 0], 1), gy_i)
        np.add.at(grad_pts, (_triples[:, 1], 0), gx_j)
        np.add.at(grad_pts, (_triples[:, 1], 1), gy_j)
        np.add.at(grad_pts, (_triples[:, 2], 0), gx_k)
        np.add.at(grad_pts, (_triples[:, 2], 1), gy_k)
        
        # Project gradient? No, we project the points after the step.
        # But we must ensure the optimizer doesn't step far outside.
        # L-BFGS-B handles bounds well. Let's use bounds.
        
        return val, grad_pts.flatten()

    # Generate diverse initial configurations
    rng = np.random.default_rng(42)
    
    inits = []
    
    # 1. Random barycentric
    for _ in range(5):
        bary = rng.dirichlet([1, 1, 1], size=N)
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = H * bary[:, 2]
        pts = np.column_stack([x, y])
        inits.append(pts)
        
    # 2. Structured grid-like
    # 3x3 grid mapped to triangle?
    # Just use a few deterministic starts
    
    # Start with a known decent configuration if possible?
    # Let's try a "center + ring" heuristic
    center = np.array([0.5, H/3])
    ring_radius = 0.2
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    ring_pts = np.column_stack([
        center[0] + ring_radius * np.cos(angles),
        center[1] + ring_radius * np.sin(angles)
    ])
    # Add 4 more points?
    # This is getting complicated. Random + Gradient should be enough.
    
    # Bounds for L-BFGS-B
    # x in [0, 1], y in [0, H]
    # But triangle is not a box.
    # We will project after each iteration? 
    # L-BFGS-B doesn't support custom callbacks for projection easily in all versions, 
    # but we can use a wrapper or just rely on the fact that the gradient pushes back.
    # However, if we use bounds [0,1]x[0,H], points might be outside the triangle corners.
    # The objective function will penalize them? No, the objective is just min area.
    # Points outside the triangle can have large areas, but the constraint is violated.
    # We MUST enforce the constraint.
    
    # Strategy: Use L-BFGS-B with bounds, but wrap the objective to project points?
    # No, that breaks the gradient.
    # Better: Use a penalty in the objective if outside?
    # Or, use the fact that the optimal solution is inside. 
    # If we start inside, and use small steps, we might stay inside.
    # But L-BFGS-B can take large steps.
    
    # Alternative: Use SLSQP which supports general constraints?
    # SLSQP is slower.
    
    # Let's try L-BFGS-B with a strong penalty for being outside, 
    # but keep the gradient of the area part.
    
    def objective_with_penalty(pts_flat):
        pts = pts_flat.reshape(N, 2)
        
        # Check bounds
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Triangle inequalities:
        # y >= 0
        # y <= sqrt(3) x
        # y <= -sqrt(3) (x - 1)
        
        # Violations
        v1 = np.minimum(0, y)
        v2 = np.minimum(0, SQRT3 * x - y)
        v3 = np.minimum(0, -SQRT3 * x - y + SQRT3)
        
        # Quadratic penalty
        pen = np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2)
        
        # Area objective
        val, grad_area = objective_and_grad(pts_flat)
        
        # Gradient of penalty
        # d/dy v1 = -1 if y < 0 else 0
        # d/dx v2 = sqrt(3) if v2 < 0 else 0
        # d/dy v2 = -1 if v2 < 0 else 0
        # d/dx v3 = -sqrt(3) if v3 < 0 else 0
        # d/dy v3 = -1 if v3 < 0 else 0
        
        grad_pen = np.zeros_like(pts_flat)
        
        # v1: y < 0
        mask1 = y < 0
        grad_pen[mask1, 1] += 2 * v1[mask1] * (-1) # d(v1^2)/dy = 2*v1 * dv1/dy = 2*v1 * (-1)
        
        # v2: sqrt(3)x - y < 0
        mask2 = (SQRT3 * x - y) < 0
        # dv2/dx = sqrt(3), dv2/dy = -1
        grad_pen[mask2, 0] += 2 * v2[mask2] * SQRT3
        grad_pen[mask2, 1] += 2 * v2[mask2] * (-1)
        
        # v3: -sqrt(3)x - y + sqrt(3) < 0
        mask3 = (-SQRT3 * x - y + SQRT3) < 0
        # dv3/dx = -sqrt(3), dv3/dy = -1
        grad_pen[mask3, 0] += 2 * v3[mask3] * (-SQRT3)
        grad_pen[mask3, 1] += 2 * v3[mask3] * (-1)
        
        # Scale penalty
        pen_weight = 100.0
        total_val = val + pen_weight * pen
        total_grad = grad_area + pen_weight * grad_pen
        
        return total_val, total_grad

    # Run L-BFGS-B
    bounds = [(0, 1), (0, H)] * N
    
    for init in inits:
        if _is_time_up():
            break
            
        x0 = init.flatten()
        
        try:
            res = minimize(
                objective_with_penalty, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        if _is_time_up():
            break
            
        res_pts = res.x.reshape(N, 2)
        
        # Project to be safe
        res_pts = _project_to_triangle(res_pts)
        
        area = _min_triangle_area(res_pts)
        
        if _is_inside_triangle(res_pts):
            if area > best_area:
                best_area = area
                best_pts = res_pts.copy()
        
        # Check initial
        init_area = _min_triangle_area(init)
        if _is_inside_triangle(init) and init_area > best_area:
            best_area = init_area
            best_pts = init.copy()

    # Final polish with higher beta (sharper min approximation)
    if best_pts is not None and not _is_time_up():
        # Redefine objective with higher beta?
        # We can't easily change beta inside the closure without redefining.
        # Let's just run another L-BFGS-B on the best point found.
        
        def objective_polish(pts_flat):
            if _is_time_up():
                return 1e9, np.zeros_like(pts_flat)
            pts = pts_flat.reshape(N, 2)
            # Use exact min for polish? No, gradient is undefined.
            # Use high beta.
            # We need to redefine the logic with beta=100
            # For simplicity, just run L-BFGS-B again with the same function.
            # The previous run should have converged.
            return objective_with_penalty(pts_flat)

        x0 = best_pts.flatten()
        try:
            res = minimize(
                objective_polish, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            res_pts = res.x.reshape(N, 2)
            res_pts = _project_to_triangle(res_pts)
            area = _min_triangle_area(res_pts)
            if _is_inside_triangle(res_pts):
                if area > best_area:
                    best_area = area
                    best_pts = res_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback
        bary = np.ones((N, 3)) / 3.0
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = H * bary[:, 2]
        best_pts = np.column_stack([x, y])
        
    return best_pts

