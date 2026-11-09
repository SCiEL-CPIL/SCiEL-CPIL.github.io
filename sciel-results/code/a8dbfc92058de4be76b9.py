"""Direct geometric approach: optimize 130 points on unit sphere to minimize covering radius."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _facet_circumcenters(points):
    """Compute circumcenter directions and cos values for all hull facets.
    Returns (cos_vals, normals, valid_mask) arrays of length n_facets.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None
    facets = hull.simplices
    n_facets = len(facets)
    cos_vals = np.zeros(n_facets)
    normals = np.zeros((n_facets, 3))
    valid = np.zeros(n_facets, dtype=bool)
    for i, facet in enumerate(facets):
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            continue
        p = p / p_norm
        if p @ a < 0:
            p = -p
        cos_vals[i] = p @ a
        normals[i] = p
        valid[i] = True
    return cos_vals, normals, valid, facets

def _covering_radius(points):
    """Compute covering radius in radians via convex hull facets."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return np.pi
    result = _facet_circumcenters(points)
    if result is None:
        return np.pi
    cos_vals, normals, valid, facets = result
    if not np.any(valid):
        return np.pi
    min_cos = np.min(cos_vals[valid])
    return np.arccos(np.clip(min_cos, -1.0, 1.0))

def _soft_objective_and_grad(points, beta=50.0):
    """
    Compute smooth surrogate objective and its gradient w.r.t. points.
    Objective: soft_max of angular distances = soft_min of cos_vals.
    Uses proper derivative of circumcenter w.r.t. facet vertices.
    """
    points = np.asarray(points, dtype=np.float64)
    result = _facet_circumcenters(points)
    if result is None:
        return np.pi, np.zeros_like(points)
    cos_vals, normals, valid, facets = result
    n_valid = np.sum(valid)
    if n_valid == 0:
        return np.pi, np.zeros_like(points)
    
    # Work with -cos_vals: we want to minimize max angular distance = minimize max(-cos)
    # soft_max(-cos) = (1/beta)*log(sum(exp(-beta*cos_i)))
    neg_cos = -cos_vals[valid]
    neg_cos_clipped = np.clip(neg_cos, -50.0, 50.0)
    max_neg = np.max(neg_cos_clipped)
    exp_vals = np.exp(beta * (neg_cos_clipped - max_neg))
    sum_exp = np.sum(exp_vals)
    soft_val = (1.0 / beta) * (max_neg + np.log(sum_exp))
    
    # Weight for each valid facet: w_i = exp(-beta*cos_i)/sum
    weights = np.zeros(len(facets))
    weights[valid] = exp_vals / sum_exp
    
    # Gradient of soft_val w.r.t. cos_vals: d(soft)/d(cos_i) = -w_i (for valid i)
    # So d(soft)/d(cos_i) = -weights[i] for valid facets
    
    # Now compute gradient w.r.t. points using chain rule.
    # For each facet with vertices a,b,c:
    #   p = normalize(cross(a-b, b-c))
    #   cos_val = p · a
    # The circumcenter direction p depends on all three vertices.
    # 
    # Let u = a - b, v = b - c, w = cross(u, v)
    # p = w / |w|
    # 
    # dp/da = (I - p p^T) · dw/da / |w|  where dw/da = cross(1, v) = -cross(v, 1)... 
    # Actually: w = cross(a-b, b-c)
    # dw/da = cross(1, b-c) = cross(1, v)  [treating 1 as identity]
    # More precisely: d(cross(u,v))/du = cross(v, ·) applied to delta_u
    # d(cross(u,v))/dv = -cross(u, ·) applied to delta_v
    # 
    # So: dw/da = cross(v, delta_a)  (since u = a-b, du = delta_a)
    #     dw/db = -cross(v, delta_b) - cross(u, delta_b)  (since du = -delta_b, dv = delta_b)
    #     dw/dc = -cross(u, -delta_c) = cross(u, delta_c)  (since dv = b-c, dv = -delta_c)
    #
    # Let me be more careful:
    # u = a - b, v = b - c
    # w = cross(u, v)
    # 
    # d w / d a = cross(du/da, v) + cross(u, dv/da) = cross(1, v) + 0 = cross(1, v)
    #   where cross(1, v) means: for any perturbation delta_a, the change is cross(delta_a, v)
    #   So dw/da as a linear map: delta_a -> cross(delta_a, v) = -cross(v, delta_a)
    #
    # d w / d b: du/db = -1, dv/db = 1
    #   delta_b -> cross(-delta_b, v) + cross(u, delta_b) = -cross(delta_b, v) + cross(u, delta_b)
    #
    # d w / d c: du/dc = 0, dv/dc = -1
    #   delta_c -> cross(u, -delta_c) = -cross(u, delta_c)
    #
    # Now p = w/|w|, so dp = (I - pp^T) dw / |w|
    # 
    # cos_val = p · a
    # d(cos_val)/da = dp · a + p · delta_a
    # d(cos_val)/db = dp · a
    # d(cos_val)/dc = dp · a
    #
    # This is getting complex. Let me use a simpler numerical approach for the gradient
    # of cos_val w.r.t. each point, then chain through the soft max.
    
    # Alternative: compute gradient numerically per facet contribution
    # But that's expensive. Let's use an analytic shortcut.
    
    # For the circumcenter of triangle (a,b,c) on unit sphere:
    # The circumcenter direction is proportional to:
    #   a * |b-c|^2 * sin(angle_a) ... no, let me use the cross product formula.
    #
    # Actually, there's a cleaner formula. The circumcenter (direction) of triangle abc
    # on the unit sphere can be computed as:
    #   p ∝ a*(b·c - 1) + b*(c·a - 1) + c*(a·b - 1)  ... no that's not right either.
    #
    # Let me just use the cross product approach with proper derivatives.
    # For efficiency, I'll compute the gradient contribution per facet.
    
    grad_points = np.zeros_like(points)
    
    for i in range(len(facets)):
        if not valid[i]:
            continue
        w_i = weights[i]
        if w_i < 1e-15:
            continue
        
        facet = facets[i]
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        u = a - b
        v = b - c
        w_vec = np.cross(u, v)
        w_norm = np.linalg.norm(w_vec)
        if w_norm < 1e-12:
            continue
        p = w_vec / w_norm
        if p @ a < 0:
            p = -p
            w_vec = -w_vec  # flip for consistency
        
        # Gradient of cos_val w.r.t. each vertex
        # cos_val = p · a
        # dp/da: for perturbation da, dp = (I - pp^T) * cross(da, v) / w_norm
        #   (since dw/da: delta -> cross(delta, v))
        #   Wait, I need to be careful about the sign flip.
        
        # Let me recompute without the flip and handle it separately.
        # Actually, let me just use the fact that:
        # d(cos_val)/d(point_j) for j in {a,b,c}
        
        # Using the formula: p = normalize(cross(a-b, b-c))
        # Let's define the unnormalized normal: n = cross(a-b, b-c)
        # p = n/|n| (with possible sign flip)
        
        # For the gradient, I'll use:
        # d(p·a)/da = (I - pp^T)/|n| * cross(da, v) · a + p · da
        #           = (I - pp^T)/|n| * cross(v, a) · da + p · da  [using cross(da,v)·a = cross(v,a)·da... let me verify]
        # Actually: cross(da, v) · a = da · (v × a) ... no.
        # cross(da, v) · a = da · (v × a)? Let's use the identity: (x×y)·z = x·(y×z)
        # So cross(da, v) · a = da · (v × a) = da · (-a × v)
        
        # This is getting too complex analytically. Let me use a simpler but correct approach:
        # Compute the gradient of cos_val w.r.t. each of the three points numerically
        # using small perturbations, but only for the weighted facets.
        
        # Actually, for the soft objective, the gradient is:
        # d(soft)/d(point_j) = sum_i w_i * d(cos_i)/d(point_j) * (-1)  [since soft uses -cos]
        # Wait: soft = (1/beta) log sum exp(-beta * cos_i)
        # d(soft)/d(cos_i) = -exp(-beta*cos_i)/sum = -w_i
        # d(soft)/d(point_j) = sum_i (-w_i) * d(cos_i)/d(point_j)
        
        # For efficiency, let me compute d(cos_i)/d(point_j) analytically.
        # For a facet with vertices at indices ia, ib, ic:
        # cos = p · a where p = normalize(cross(a-b, b-c))
        
        # Let me use a different parametrization. The circumcenter direction of
        # triangle (a,b,c) on unit sphere satisfies p·a = p·b = p·c = cos(theta).
        # So p is in the null space of [a-b; b-c] and is unit.
        
        # Gradient approach: I'll compute it via the formula
        # dp = (I - pp^T) dn / |n|
        # where n = cross(a-b, b-c)
        
        # dn/da: for perturbation δa, δn = cross(δa, b-c) = cross(δa, v)
        # dn/db: δn = cross(-δb, v) + cross(a-b, δb) = -cross(δb, v) + cross(u, δb)
        # dn/dc: δn = cross(u, -δc) = -cross(u, δc)
        
        # So:
        # d(cos)/da = dp·a + p·δa = [(I-pp^T)cross(δa,v)/|n|]·a + p·δa
        #          = [(I-pp^T)cross(v,a)/|n| + p] · δa  [using cross(δa,v)·a = δa·(v×a)]
        
        # Wait: cross(δa, v) · a. Using (x×y)·z = x·(y×z):
        # cross(δa, v) · a = δa · (v × a)
        
        # So d(cos)/da as a vector (coefficient of δa):
        # = (I-pp^T)/|n| * (v × a) + p
        
        # Similarly:
        # d(cos)/db: δn = -cross(δb, v) + cross(u, δb)
        #   = cross(δb, -v + u) ... no.
        #   -cross(δb, v) = δb · (v × ·) ... 
        #   cross(δb, v) · a = δb · (v × a)
        #   cross(u, δb) · a = u · (δb × a) = δb · (a × u)
        #   So d(cos)/db = [(I-pp^T)/|n| * (-(v×a) + (a×u))] + 0  [no p·δb term since cos=p·a]
        
        # Wait, cos = p · a, so d(cos)/db = dp · a (no direct δb in p·a)
        # dp·a = [(I-pp^T)dn/|n|]·a
        # dn/db = -cross(δb, v) + cross(u, δb)
        # dn/db · a = -cross(δb,v)·a + cross(u,δb)·a = -δb·(v×a) + δb·(a×u) = δb·(-(v×a)+(a×u))
        
        # d(cos)/dc: dn/dc = -cross(u, δc)
        # dn/dc · a = -cross(u,δc)·a = -δc·(a×u) ... 
        # cross(u, δc) · a = u · (δc × a) = δc · (a × u)
        # So dn/dc · a = -δc·(a×u)
        # d(cos)/dc = (I-pp^T)/|n| * (-(a×u))
        
        # Let me define:
        va_cross_a = np.cross(v, a)  # v × a
        a_cross_u = np.cross(a, u)   # a × u
        
        grad_a = (np.eye(3) - np.outer(p, p)) @ va_cross_a / w_norm + p
        grad_b = (np.eye(3) - np.outer(p, p)) @ (-va_cross_a + a_cross_u) / w_norm
        grad_c = (np.eye(3) - np.outer(p, p)) @ (-a_cross_u) / w_norm
        
        # But wait - if we flipped p (because p@a < 0), we need to account for that.
        # If the original p had p@a < 0, we flipped it. The gradient should be negated.
        # Actually, the circumcenter is uniquely determined (the one on the same side as the origin's antipode... 
        # no, it's the one such that p·a > 0 for the covering). Let me just check.
        
        # The issue is that cross(a-b, b-c) might point inward. We flip to make p·a > 0.
        # The gradient computation above assumes p points outward (p·a > 0).
        # If we had to flip, the derivative of the flip operation is -1, so we negate.
        
        # Let me recompute without the flip and determine the sign.
        n_orig = np.cross(a - b, b - c)
        n_norm_orig = np.linalg.norm(n_orig)
        if n_norm_orig < 1e-12:
            continue
        p_orig = n_orig / n_norm_orig
        sign = 1.0 if (p_orig @ a) > 0 else -1.0
        p_final = sign * p_orig
        
        # The gradient formulas above work for p_final = p_orig (sign=1)
        # If sign = -1, all gradients are negated.
        
        grad_a = sign * ((np.eye(3) - np.outer(p_final, p_final)) @ np.cross(v, a) / n_norm_orig + p_final)
        grad_b = sign * ((np.eye(3) - np.outer(p_final, p_final)) @ (-np.cross(v, a) + np.cross(a, u)) / n_norm_orig)
        grad_c = sign * ((np.eye(3) - np.outer(p_final, p_final)) @ (-np.cross(a, u)) / n_norm_orig)
        
        # d(soft)/d(point) = -w_i * d(cos_i)/d(point)
        grad_points[facet[0]] += -w_i * grad_a
        grad_points[facet[1]] += -w_i * grad_b
        grad_points[facet[2]] += -w_i * grad_c
    
    return soft_val, grad_points

def _lbfgs_smooth_optimize(points, deadline, beta_start=20.0, beta_end=80.0, max_iters=300):
    """Use L-BFGS on the smooth surrogate to refine the configuration."""
    points = points.copy()
    
    def objective(x):
        pts = x.reshape(N, 3)
        # Normalize to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        val, _ = _soft_objective_and_grad(pts, beta=beta_start)
        return val
    
    def gradient(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        _, grad = _soft_objective_and_grad(pts, beta=beta_start)
        # Project gradient to tangent space (remove radial component)
        grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        return grad.flatten()
    
    try:
        result = minimize(
            objective,
            points.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.success or result.fun < objective(points.flatten()):
            new_points = result.x.reshape(N, 3)
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            new_points = new_points / norms
            return new_points
    except Exception:
        pass
    
    return points

def _weighted_worst_facets(points, k=20, beta=50.0):
    """
    Return weighted worst facets using soft-max weighting.
    Returns list of (indices, circumcenter, weight) for top k facets.
    """
    points = np.asarray(points, dtype=np.float64)
    result = _facet_circumcenters(points)
    if result is None:
        return []
    cos_vals, normals, valid, facets = result
    
    if not np.any(valid):
        return []
    
    neg_cos = -cos_vals[valid]
    neg_cos_clipped = np.clip(neg_cos, -50.0, 50.0)
    max_neg = np.max(neg_cos_clipped)
    exp_vals = np.exp(beta * (neg_cos_clipped - max_neg))
    sum_exp = np.sum(exp_vals)
    weights = np.zeros(len(facets))
    weights[valid] = exp_vals / sum_exp
    
    valid_indices = np.where(valid)[0]
    sorted_idx = valid_indices[np.argsort(weights[valid_indices])[::-1]]
    
    result_list = []
    for idx in sorted_idx[:k]:
        if weights[idx] < 1e-8:
            break
        result_list.append((facets[idx].copy(), normals[idx].copy(), weights[idx]))
    
    return result_list


def _worst_facets(points, k=5):
    """Return the k worst facets (by circumradius) as list of (indices, circumcenter)."""
    points = np.asarray(points, dtype=np.float64)
    result = _facet_circumcenters(points)
    if result is None:
        return []
    cos_vals, normals, valid, facets = result
    
    if not np.any(valid):
        return []
    
    valid_indices = np.where(valid)[0]
    sorted_idx = valid_indices[np.argsort(cos_vals[valid_indices])]
    
    return [(facets[idx].copy(), normals[idx].copy()) for idx in sorted_idx[:k]]


def _gradient_descent(points, deadline, max_iters=3000, base_step=0.05):
    """Optimize points using gradient descent toward circumcenters of worst facets with soft weighting."""
    points = points.copy()
    best_radius = _covering_radius(points)
    step = base_step
    beta = 30.0
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            step = min(step * 1.05, 0.1)
        else:
            step *= 0.85
        
        if step < 1e-10:
            break
        
        progress = it / max_iters
        beta = 30.0 + 70.0 * progress
        
        worst = _weighted_worst_facets(points, k=15, beta=beta)
        if not worst:
            break
        
        displacement = np.zeros_like(points)
        for facet, circ, weight in worst:
            for idx in facet:
                pt = points[idx]
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    tangent = direction - np.dot(direction, pt) * pt
                    norm_t = np.linalg.norm(tangent)
                    if norm_t > 1e-10:
                        tangent = tangent / norm_t
                        displacement[idx] += weight * tangent
        
        for idx in range(N):
            if np.linalg.norm(displacement[idx]) > 1e-10:
                new_pt = points[idx] + step * displacement[idx]
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
    
    return points


def _random_restart_optimize(rng, deadline, max_iters=2000):
    """Generate random initial config and optimize."""
    points = rng.standard_normal((N, 3))
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return _gradient_descent(points, deadline, max_iters=max_iters)

def _polish(points, deadline, max_iters=500, base_step=0.005):
    """Final high-precision polish with very small steps."""
    points = points.copy()
    best_radius = _covering_radius(points)
    step = base_step
    
    for it in range(max_iters):
        if time.time() > deadline - 2.0:
            break
        
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            step = min(step * 1.02, 0.01)
        else:
            step *= 0.9
        
        if step < 1e-12:
            break
        
        worst = _weighted_worst_facets(points, k=3, beta=100.0)
        if not worst:
            break
        
        displacement = np.zeros_like(points)
        for facet, circ, weight in worst:
            for idx in facet:
                pt = points[idx]
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    tangent = direction - np.dot(direction, pt) * pt
                    norm_t = np.linalg.norm(tangent)
                    if norm_t > 1e-10:
                        tangent = tangent / norm_t
                        displacement[idx] += weight * tangent
        
        for idx in range(N):
            if np.linalg.norm(displacement[idx]) > 1e-10:
                new_pt = points[idx] + step * displacement[idx]
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
    
    return points


def construct():
    t_start = time.time()
    deadline = t_start + 140.0
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Start with Fibonacci spiral as baseline
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_points = fib_points / np.linalg.norm(fib_points, axis=1, keepdims=True)
    
    best_points = _gradient_descent(fib_points, deadline, max_iters=2000, base_step=0.04)
    best_radius = _covering_radius(best_points)
    
    # Multiple random restarts
    num_restarts = 6
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        
        if best_points is not None and restart % 2 == 0:
            perturbation = 0.15 * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            points = rng.standard_normal((N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        max_iters = min(2000, int(remaining * 20))
        optimized = _gradient_descent(points, deadline, max_iters=max_iters, base_step=0.06)
        
        radius = _covering_radius(optimized)
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # L-BFGS smooth optimization on the best configuration
    if best_points is not None and time.time() < deadline - 15.0:
        lbfgs_points = _lbfgs_smooth_optimize(best_points, deadline, beta_start=30.0, beta_end=80.0, max_iters=200)
        lbfgs_radius = _covering_radius(lbfgs_points)
        if lbfgs_radius < best_radius:
            best_radius = lbfgs_radius
            best_points = lbfgs_points
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 3.0:
        best_points = _polish(best_points, deadline, max_iters=800, base_step=0.003)
        best_radius = _covering_radius(best_points)
    
    # Second L-BFGS pass after polish
    if best_points is not None and time.time() < deadline - 3.0:
        lbfgs_points = _lbfgs_smooth_optimize(best_points, deadline, beta_start=50.0, beta_end=100.0, max_iters=150)
        lbfgs_radius = _covering_radius(lbfgs_points)
        if lbfgs_radius < best_radius:
            best_radius = lbfgs_radius
            best_points = lbfgs_points
    
    # Ensure valid output
    if best_points is None:
        best_points = fib_points
    
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
