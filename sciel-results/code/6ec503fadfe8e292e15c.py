import numpy as np
import time
from scipy.optimize import linprog, minimize
from itertools import combinations

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute all triples
_TRIPLES = list(combinations(range(N), 3))
_N_TRIPLES = len(_TRIPLES)

def _min_area(points):
    """Compute minimum triangle area for N points."""
    min_a = np.inf
    for i, j, k in _TRIPLES:
        pi, pj, pk = points[i], points[j], points[k]
        cross = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
        a = 0.5 * abs(cross)
        if a < min_a:
            min_a = a
    return min_a

def _bary_to_cart(bary):
    """Convert barycentric coords (N,3) to cartesian (N,2)."""
    return bary[:, 0] * V0 + bary[:, 1] * V1 + bary[:, 2] * V2

def _cart_to_bary(cart):
    """Convert cartesian (N,2) to barycentric coords (N,3)."""
    bary = np.zeros((N, 3))
    for i in range(N):
        x, y = cart[i]
        # l2 = y / (sqrt(3)/2)
        l2 = 2.0 * y / np.sqrt(3)
        # l1 = x - 0.5*l2
        l1 = x - 0.5 * l2
        # l0 = 1 - l1 - l2
        l0 = 1.0 - l1 - l2
        bary[i] = [l0, l1, l2]
    return bary

def _project_bary(bary):
    """Project barycentric coords to valid simplex (each row sums to 1, all >= 0)."""
    for i in range(N):
        b = bary[i].copy()
        if np.any(b < 0):
            # Simple projection: set negatives to 0, renormalize
            b = np.maximum(b, 0)
            s = b.sum()
            if s > 0:
                b /= s
            else:
                b = np.array([1.0, 0.0, 0.0])
            bary[i] = b
    return bary

def _solve_lp(bary, t0):
    """
    Solve LP to maximize t subject to:
    - t <= 0.5 * |det(V_i, V_j, V_k)| for all triples (using current orientation signs)
    - barycentric constraints: b_i >= 0, sum(b_i) = 1 for each point
    
    Variables: [t, b_10, b_11, b_12, ..., b_N0, b_N1, b_N2]
    Total: 1 + 3*N = 34 variables
    
    Constraints:
    - For each triple (i,j,k): sign * (det) - t >= 0  =>  -sign*det + t <= 0
    - For each point i: sum of bary = 1 (equality)
    - For each point i: each bary >= 0 (handled by bounds)
    """
    n_vars = 1 + 3 * N
    
    # Objective: maximize t => minimize -t
    c = np.zeros(n_vars)
    c[0] = -1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For each triple, we need: sign * det(b_i, b_j, b_k) >= t
    # det in barycentric coords is linear in each point's barycentric coords
    # Actually, the determinant of three points in cartesian is:
    # det = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    # In barycentric coords, x = b1 + 0.5*b2, y = (sqrt(3)/2)*b2
    # This is getting complex. Let me use cartesian directly.
    
    # Let me reformulate: use cartesian coordinates as variables
    # Variables: [t, x_1, y_1, x_2, y_2, ..., x_N, y_N]
    # Total: 1 + 2*N = 23 variables
    
    n_vars2 = 1 + 2 * N
    c2 = np.zeros(n_vars2)
    c2[0] = -1.0  # maximize t
    
    # Constraint: t >= 0 for t >= 0 (handled by bounds)
    # For each triple (i,j,k):
    #   sign_{ijk} * (x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)) - 2t >= 0
    #   => -sign_{ijk} * (x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)) + 2t <= 0
    # This is bilinear in the variables! Not linear.
    
    # The problem is that area is bilinear in coordinates. LP won't work directly.
    # 
    # Alternative: use the fact that for a FIXED orientation (sign pattern),
    # we can write the constraint as:
    #   sign * det >= 2t
    # where det is linear in each coordinate... wait, no.
    # det = x_i*y_j - x_i*y_k - x_j*y_i + x_j*y_k + x_k*y_i - x_k*y_j
    # This is bilinear, not linear.
    
    # So LP approach doesn't directly work because area is bilinear.
    # 
    # Let me try a different approach: use a smooth approximation with better optimization.
    
    return None  # LP approach is not viable due to bilinearity

def _smooth_min_area(points, beta=100.0):
    """
    Smooth approximation of min area using log-sum-exp.
    Returns (smooth_min, gradient).
    """
    areas = np.zeros(_N_TRIPLES)
    for idx, (i, j, k) in enumerate(_TRIPLES):
        pi, pj, pk = points[i], points[j], points[k]
        cross = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
        areas[idx] = 0.5 * abs(cross)
    
    # Log-sum-exp for min: min(a) ≈ -1/beta * log(sum(exp(-beta*a)))
    max_a = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_a))
    smooth_min = max_a - (1.0/beta) * np.log(np.sum(exp_terms))
    
    # Gradient
    weights = exp_terms / np.sum(exp_terms)  # softmax-like weights
    grad = np.zeros((N, 2))
    
    for idx, (i, j, k) in enumerate(_TRIPLES):
        w = weights[idx] / beta
        pi, pj, pk = points[i], points[j], points[k]
        cross = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
        s = np.sign(cross)
        if s == 0:
            s = 1.0
        
        # d(area)/d(points) where area = 0.5*|cross|
        # For cross > 0: d(area)/d(x_i) = -0.5*(y_j - y_k), etc.
        # For cross < 0: d(area)/d(x_i) = 0.5*(y_j - y_k), etc.
        
        dxi_x = -0.5 * s * (pj[1] - pk[1])
        dxi_y = -0.5 * s * (pk[0] - pj[0])
        dxj_x = -0.5 * s * (pk[1] - pi[1])
        dxj_y = 0.5 * s * (pk[0] - pi[0])
        dxk_x = 0.5 * s * (pj[1] - pi[1])
        dxk_y = -0.5 * s * (pj[0] - pi[0])
        
        grad[i, 0] += w * dxi_x
        grad[i, 1] += w * dxi_y
        grad[j, 0] += w * dxj_x
        grad[j, 1] += w * dxj_y
        grad[k, 0] += w * dxk_x
        grad[k, 1] += w * dxk_y
    
    return smooth_min, grad

def _objective_smooth(params, beta=100.0):
    """Objective using smooth min area."""
    points = _param_to_points(params)
    val, grad = _smooth_min_area(points, beta)
    return -val, grad

def _param_to_points(params):
    """Map parameters to points inside the triangle using barycentric coords."""
    # params: (3*N,) barycentric coordinates
    bary = params.reshape(N, 3)
    bary = np.maximum(bary, 0)
    for i in range(N):
        s = bary[i].sum()
        if s > 0:
            bary[i] /= s
        else:
            bary[i] = np.array([1.0, 0.0, 0.0])
    return _bary_to_cart(bary)

def _points_to_param(points):
    """Map points to barycentric parameter vector."""
    bary = _cart_to_bary(points)
    return bary.flatten()

def _generate_initializations(rng, n_inits=20):
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Grid-based initializations
    for grid_n in [3, 4, 5]:
        points = []
        for i in range(grid_n):
            for j in range(grid_n - i):
                l0 = 1.0 - (i + j) / (grid_n - 1)
                l1 = i / (grid_n - 1)
                l2 = j / (grid_n - 1)
                pt = l0 * V0 + l1 * V1 + l2 * V2
                points.append(pt)
        points = np.array(points)
        if len(points) >= N:
            idx = rng.choice(len(points), N, replace=False)
            inits.append(points[idx])
        else:
            while len(points) < N:
                r1, r2 = rng.uniform(0, 1), rng.uniform(0, 1)
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                pt = r1 * V1 + r2 * V2
                points = np.vstack([points, pt])
            inits.append(points)
    
    # 2. FPS-based initializations
    for _ in range(10):
        pts = np.zeros((N, 2))
        pts[0] = (V0 + V1 + V2) / 3.0
        for k in range(1, N):
            r1 = rng.uniform(0, 1, 500)
            r2 = rng.uniform(0, 1, 500)
            mask = r1 + r2 > 1
            r1[mask] = 1 - r1[mask]
            r2[mask] = 1 - r2[mask]
            candidates = np.column_stack([
                r1 * V1[0] + r2 * V2[0],
                r1 * V1[1] + r2 * V2[1]
            ])
            diffs = candidates[:, np.newaxis, :] - pts[np.newaxis, :k, :]
            dists = np.linalg.norm(diffs, axis=2)
            min_dists = np.min(dists, axis=1)
            best_idx = np.argmax(min_dists)
            pts[k] = candidates[best_idx]
        inits.append(pts)
    
    # 3. Random initializations
    for _ in range(5):
        r1 = rng.uniform(0, 1, N)
        r2 = rng.uniform(0, 1, N)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        pts = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        inits.append(pts)
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    rng = np.random.RandomState(42)
    
    best_params = None
    best_score = -np.inf
    
    # Generate initializations
    inits = _generate_initializations(rng)
    
    # Optimize each initialization
    for init_pts in inits:
        if time_left() < 30.0:
            break
        
        params = _points_to_param(init_pts)
        
        # Multi-stage smooth optimization
        for beta in [50.0, 100.0, 200.0, 500.0]:
            if time_left() < 25.0:
                break
            
            try:
                res = minimize(
                    _objective_smooth, params, args=(beta,),
                    method='L-BFGS-B', jac=True,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res.success or res.fun < -best_score:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_params = res.x.copy()
                    params = res.x
            except Exception:
                pass
        
        # Final polish with Nelder-Mead
        if time_left() > 20.0:
            try:
                def _obj_nm(p):
                    pts = _param_to_points(p)
                    return -_min_area(pts)
                
                res = minimize(
                    _obj_nm, params, method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Iterative refinement
    if best_params is not None:
        for scale in [0.05, 0.02, 0.01, 0.005]:
            if time_left() < 20.0:
                break
            for _ in range(8):
                if time_left() < 15.0:
                    break
                try:
                    noise = rng.normal(0, scale, size=3*N)
                    perturbed = best_params + noise
                    # Ensure valid barycentric
                    perturbed = np.maximum(perturbed, 0)
                    
                    for beta in [200.0, 500.0]:
                        res = minimize(
                            _objective_smooth, perturbed, args=(beta,),
                            method='L-BFGS-B', jac=True,
                            options={'maxiter': 150, 'ftol': 1e-12}
                        )
                        if res.fun < -best_score:
                            score = -res.fun
                            if score > best_score:
                                best_score = score
                                best_params = res.x.copy()
                            perturbed = res.x
                except Exception:
                    pass
    
    if best_params is None:
        # Fallback: centroid-based
        bary = np.zeros((N, 3))
        bary[:, 0] = 1.0 / N
        bary[:, 1] = 1.0 / N
        bary[:, 2] = 1.0 / N
        best_params = bary.flatten()
    
    points = _param_to_points(best_params)
    
    # Final safety: clip to triangle
    def _clip_to_triangle(pts):
        for i in range(len(pts)):
            x, y = pts[i]
            l2 = 2 * y / np.sqrt(3)
            l1 = x - y / np.sqrt(3)
            l0 = 1 - l1 - l2
            if l0 < 0 or l1 < 0 or l2 < 0:
                l0 = max(0, l0)
                l1 = max(0, l1)
                l2 = max(0, l2)
                s = l0 + l1 + l2
                if s > 0:
                    l0 /= s
                    l1 /= s
                    l2 /= s
                else:
                    l0 = 1
                    l1 = 0
                    l2 = 0
                pts[i] = l0 * V0 + l1 * V1 + l2 * V2
        return pts
    
    points = _clip_to_triangle(points)
    return np.array(points)
