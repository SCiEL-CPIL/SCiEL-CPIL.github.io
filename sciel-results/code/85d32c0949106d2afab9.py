"""Heilbronn-11: Successive Linear Programming (SLP) with symmetric initialization."""
import numpy as np
import time
from scipy.optimize import linprog
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3H]])

def _bary_to_cart(uv_flat):
    """Convert flat array of barycentric coords (u, v) for N points to Cartesian (N, 2)."""
    uv = uv_flat.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack([x, y])

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 2) (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    w = y / S3H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def _vectorized_areas_cart(points):
    """Calculate all triangle areas formed by the points in Cartesian."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _min_area_cart(points):
    """Calculate the minimum area of all triangles formed by the points in Cartesian."""
    areas = _vectorized_areas_cart(points)
    return np.min(areas)

def _symmetric_init():
    """
    Create a symmetric initial configuration for 11 points.
    We use a pattern that respects the 3-fold symmetry of the equilateral triangle.
    
    Structure:
    - 1 point at the center
    - 3 points forming a small triangle around the center
    - 3 points further out
    - 3 points near the vertices (but not at them)
    - 1 point... wait, 1+3+3+3 = 10. We need 11.
    
    Let's try:
    - 1 center
    - 3 on the medians, at 1/3 distance from center to vertex
    - 3 on the medians, at 2/3 distance from center to vertex
    - 3 near the midpoints of the sides? No, that breaks symmetry.
    
    Actually, a common good starting point for Heilbronn problems is a grid-like or hexagonal packing.
    For 11 points, a known good configuration often involves:
    - 1 center
    - 3 vertices (slightly inset)
    - 3 midpoints of sides (slightly inset)
    - 3 points on the medians
    
    Let's just use a perturbed symmetric hexagonal lattice subset.
    Or simply:
    - Center
    - 3 points at distance r1 from center, angles 0, 120, 240
    - 3 points at distance r2 from center, angles 60, 180, 300
    - 3 points at distance r3 from center, angles 0, 120, 240
    - 1 extra? No, 1+3+3+3=10.
    
    Let's try 1 center + 3 rings of 3? That's 10.
    Maybe 1 center + 2 rings of 3 + 1 vertex? No.
    
    Let's use a known configuration structure:
    The optimal 11-point configuration is believed to have 3-fold symmetry.
    Let's place:
    - 1 point at the centroid
    - 3 points forming an equilateral triangle centered at the centroid
    - 3 points forming a larger equilateral triangle
    - 3 points forming an even larger triangle
    - 1 point? No.
    
    Actually, 11 = 1 + 3 + 3 + 3 + 1? No.
    11 = 3 + 3 + 3 + 1 + 1?
    
    Let's just use a random symmetric initialization.
    We define 3 orbits under 120-degree rotation.
    Orbit 1: 1 point at center. (Size 1)
    Orbit 2: 3 points at distance r2, angle offset 0. (Size 3)
    Orbit 3: 3 points at distance r3, angle offset 60. (Size 3)
    Orbit 4: 3 points at distance r4, angle offset 0. (Size 3)
    Total: 1+3+3+3 = 10.
    We need 11.
    
    Maybe:
    Orbit 1: 1 center
    Orbit 2: 3 points
    Orbit 3: 3 points
    Orbit 4: 3 points
    Orbit 5: 1 point? This breaks symmetry.
    
    Perhaps the symmetry is not 3-fold for all points?
    Or maybe 11 points don't have full 3-fold symmetry?
    
    Let's just use a perturbed grid or random start, but project to symmetry if possible?
    No, let's just use a good random start and rely on SLP to find the structure.
    However, a symmetric start often helps convergence.
    
    Let's try:
    - 1 center
    - 3 points near vertices
    - 3 points near side midpoints
    - 3 points on medians
    - 1 point?
    
    Actually, let's just use a random start but ensure it's well-distributed.
    """
    # Random start
    uv = np.random.random((N, 2))
    # Project to simplex
    uv = np.maximum(0, uv)
    s = uv.sum(axis=1, keepdims=True)
    uv = uv / s
    # This doesn't ensure u+v<=1 strictly if we just normalize, but it does.
    # Wait, standard simplex projection:
    u = uv[:, 0]
    v = uv[:, 1]
    # If u+v > 1, scale
    mask = u + v > 1
    if np.any(mask):
        scale = 1.0 / (u[mask] + v[mask])
        u[mask] *= scale
        v[mask] *= scale
    uv = np.column_stack([u, v])
    
    cart = _bary_to_cart(uv.flatten())
    return cart

def _slp_iteration(points, t, max_iter=50, tol=1e-6):
    """
    Perform one SLP iteration: fix the triangle areas at current points,
    then solve the LP to maximize t subject to new area constraints.
    """
    n_vars = N * 2 + 1  # x1, y1, ..., xN, yN, t
    # Variables: p[0]=x1, p[1]=y1, ..., p[2N-2]=xN, p[2N-1]=yN, p[2N]=t
    
    # Objective: maximize t -> minimize -t
    c = np.zeros(n_vars)
    c[-1] = -1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 1. Triangle constraints (points inside the big triangle)
    # Big triangle: (0,0), (1,0), (0.5, S3H)
    # y >= 0  =>  -y <= 0
    # y <= S3 * x  =>  S3*x - y >= 0  =>  -S3*x + y <= 0
    # S3*x + y <= S3  =>  S3*x + y - S3 <= 0
    
    A_ub = []
    b_ub = []
    
    # y >= 0 for all points
    for i in range(N):
        row = np.zeros(n_vars)
        row[2*i + 1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        
    # y <= S3 * x
    for i in range(N):
        row = np.zeros(n_vars)
        row[2*i] = -S3
        row[2*i + 1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
        
    # S3 * x + y <= S3
    for i in range(N):
        row = np.zeros(n_vars)
        row[2*i] = S3
        row[2*i + 1] = 1.0
        A_ub.append(row)
        b_ub.append(S3)
        
    # 2. Area constraints: area(i,j,k) >= t
    # area = 0.5 * | (xj-xi)(yk-yi) - (yj-yi)(xk-xi) |
    # This is non-linear. Linearization:
    # At the current point, the area is A0. The gradient of the area function w.r.t. points gives the linearization.
    # Let f(x) be the signed area (without 0.5 and abs).
    # f = 0.5 * det([1, xj, yj], [1, xi, yi], [1, xk, yk]) ... no, standard cross product.
    # Let's use the signed area formula:
    # 2*Area = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
    # Let's linearize the absolute value.
    # If the current signed area is positive, we use the linearization of the positive part.
    # If negative, we use the negative part.
    
    # Precompute gradients for each triplet at the current point
    for idx in range(NUM_TRIP):
        i = _T_I[idx]
        j = _T_J[idx]
        k = _T_K[idx]
        
        xi, yi = points[i]
        xj, yj = points[j]
        xk, yk = points[k]
        
        # Signed double area
        dA = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
        
        # Gradient w.r.t. each coordinate
        # d(dA)/dxi = -(yk - yi) = yi - yk
        # d(dA)/dyi = -(xk - xi) = xi - xk
        # d(dA)/dxj = (yk - yi)
        # d(dA)/dyj = -(xk - xi)
        # d(dA)/dxk = -(yj - yi) = yi - yj
        # d(dA)/dyk = (xj - xi)
        
        g = np.zeros(n_vars)
        g[2*i]   = yi - yk
        g[2*i+1] = xi - xk
        g[2*j]   = yk - yi
        g[2*j+1] = xi - xk
        g[2*k]   = yi - yj
        g[2*k+1] = xj - xi
        
        # We want 0.5 * |dA| >= t
        # If dA > 0, then 0.5 * dA >= t  =>  dA - 2t >= 0  =>  -dA + 2t <= 0
        # If dA < 0, then -0.5 * dA >= t => -dA - 2t >= 0 => dA + 2t <= 0
        
        # Linearize dA around current point:
        # dA_new ≈ dA_old + g @ delta_x
        # So dA_new - 2t_new >= 0 (if dA_old > 0)
        # (dA_old + g @ delta_x) - 2(t_old + delta_t) >= 0
        # g @ delta_x - 2 delta_t >= -dA_old + 2t_old
        # -g @ delta_x + 2 delta_t <= dA_old - 2t_old
        
        # In our variable vector, x = [x1, y1, ..., t]
        # delta_x is the variable itself (since we are solving for the new point directly, not delta).
        # Wait, the linearization is:
        # dA(x) ≈ dA(x0) + g(x0) @ (x - x0)
        # We want dA(x) >= 2t (assuming dA>0)
        # dA(x0) + g @ (x - x0) - 2t >= 0
        # g @ x - g @ x0 + dA(x0) - 2t >= 0
        # -g @ x + 2t <= dA(x0) - g @ x0
        
        # If dA < 0, we want -dA(x) >= 2t
        # -(dA(x0) + g @ (x - x0)) - 2t >= 0
        # -dA(x0) - g @ x + g @ x0 - 2t >= 0
        # g @ x + 2t <= -dA(x0) + g @ x0
        
        row = np.zeros(n_vars)
        if dA >= 0:
            # -g @ x + 2t <= dA(x0) - g @ x0
            row[:2*N] = -g[:2*N]
            row[-1] = 2.0
            b_val = dA - np.dot(g[:2*N], points.reshape(-1))
        else:
            # g @ x + 2t <= -dA(x0) + g @ x0
            row[:2*N] = g[:2*N]
            row[-1] = 2.0
            b_val = -dA + np.dot(g[:2*N], points.reshape(-1))
            
        A_ub.append(row)
        b_ub.append(b_val)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds
    # x, y in [0, 1] (loose bounds, triangle constraints are tighter)
    # t >= 0
    bounds = [(0, 1)] * (2*N) + [(0, None)]
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        return points, t, False
        
    new_points = res.x[:2*N].reshape(N, 2)
    new_t = res.x[-1]
    
    return new_points, new_t, True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    # Initial points
    points = _symmetric_init()
    
    # Calculate initial min area
    min_area = _min_area_cart(points)
    
    # SLP iterations
    # We iterate SLP until convergence or time limit
    # Each SLP iteration solves an LP with ~165 + 33 constraints and 23 variables. This is fast.
    
    best_points = points.copy()
    best_min_area = min_area
    
    # We can also try multiple random starts if time permits
    num_starts = 5
    for start_idx in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        if start_idx == 0:
            points = _symmetric_init()
        else:
            # Perturb the best found so far
            points = best_points + np.random.normal(0, 0.1, (N, 2))
            # Project to triangle
            # Simple clamp
            x = points[:, 0]
            y = points[:, 1]
            y = np.maximum(0, y)
            y = np.minimum(y, S3 * x)
            y = np.minimum(y, S3 - S3 * x)
            x = np.maximum(0, x)
            x = np.minimum(1, x)
            points = np.column_stack([x, y])
        
        t = _min_area_cart(points)
        
        # SLP loop
        for it in range(100):
            if time.time() > deadline - 5:
                break
                
            new_points, new_t, success = _slp_iteration(points, t)
            
            if not success:
                break
                
            # Check convergence
            if abs(new_t - t) < 1e-8:
                points = new_points
                t = new_t
                break
                
            # Update
            points = new_points
            t = new_t
            
            # Evaluate true min area
            true_min = _min_area_cart(points)
            if true_min > best_min_area:
                best_min_area = true_min
                best_points = points.copy()
                
            # If the LP solution is not feasible for the true problem (due to linearization error),
            # the true min area might be less than t.
            # If true_min < t, we should adjust t down to true_min to stay feasible?
            # Actually, SLP guarantees that the new point satisfies the linearized constraints,
            # which are lower bounds on the true area (if we linearize the convex function area? No, area is not convex).
            # Area is convex in each pair of points? No.
            # However, the linearization of |dA| is a lower bound if we use the subgradient at the current point?
            # The function f(x) = |g(x)| is convex if g is linear.
            # The tangent plane to a convex function is a lower bound.
            # So 0.5 * |dA(x)| >= 0.5 * (|dA(x0)| + g(x0) @ (x - x0))
            # So the linearized constraint ensures the true area is >= the linearized value.
            # Thus, if the LP says t, the true area is >= t.
            # So true_min should be >= t.
            # If true_min < t, it means numerical issues or my linearization is wrong.
            
            # If true_min < t, we set t = true_min to be safe.
            if true_min < t:
                t = true_min
                
    # Final refinement: local search to fix any small violations
    # Use Nelder-Mead on the min area
    if time.time() < deadline - 5:
        from scipy.optimize import minimize
        
        def objective(flat_cart):
            cart = flat_cart.reshape(N, 2)
            min_val = _min_area_cart(cart)
            # Penalty for being outside
            x = cart[:, 0]
            y = cart[:, 1]
            pen1 = np.sum(np.maximum(0, -y)**2)
            pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
            pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
            penalty = 100.0 * (pen1 + pen2 + pen3)
            return -min_val + penalty
            
        res = minimize(objective, best_points.flatten(), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        new_points = res.x.reshape(N, 2)
        new_min = _min_area_cart(new_points)
        
        # Check validity
        x = new_points[:, 0]
        y = new_points[:, 1]
        valid = np.all(y >= -1e-9) and np.all(y <= S3 * x + 1e-9) and np.all(S3 * x + y <= S3 + 1e-9)
        
        if valid and new_min > best_min_area:
            best_points = new_points
            best_min_area = new_min
            
    # Final clamp
    x = best_points[:, 0]
    y = best_points[:, 1]
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)
    x = np.maximum(0, x)
    x = np.minimum(1, x)
    best_points = np.column_stack([x, y])
    
    return best_points
