"""Packing 200 equal circles in a unit square using cutting-plane + L-BFGS-B polish."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, linprog
from scipy.spatial import cKDTree, Delaunay


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern, searching over multiple spacings."""
    best_c = None
    best_r = 0.0
    
    for d in np.linspace(0.055, 0.075, 30):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        if len(points) >= N:
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        elif len(points) > 0 and best_c is None:
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _jittered_grid_init(seed=0):
    """Create a jittered grid initialization."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(np.sqrt(N)))
    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.45, 0.45, m),
        np.linspace(-0.45, 0.45, m)
    )
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    rng.shuffle(grid)
    c = grid[:N].copy()
    c += rng.normal(0, 0.01, size=(N, 2))
    c = np.clip(c, -0.4999, 0.4999)
    return c


def _soft_min_energy(params, temperature=0.015):
    """Smooth energy function using soft-min for the radius."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    dists = pdist(c, metric='euclidean')
    
    # Use power-sum for soft minimum of distances
    p = 30
    mask = dists < 0.15
    if np.any(mask):
        soft_min_dist = np.sum(dists[mask] ** p) ** (-1.0 / p)
    else:
        soft_min_dist = np.min(dists)
    
    p_b = 30
    soft_boundary = np.sum(boundary_dists ** p_b) ** (-1.0 / p_b)
    
    r_pair = soft_min_dist / 2.0
    r_bound = soft_boundary
    
    T = temperature
    soft_min_combined = -T * np.logaddexp(-r_pair / T, -r_bound / T)
    
    return -soft_min_combined


def _lbfgs_optimize(c0, max_time=20.0, temperatures=None):
    """Local optimization using L-BFGS-B with soft-min energy."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    if temperatures is None:
        temperatures = [0.01, 0.015, 0.02, 0.03]
    
    for T in temperatures:
        if time.time() > start_time + max_time - 2:
            break
        
        def obj(params):
            return _soft_min_energy(params, temperature=T)
        
        try:
            result = minimize(
                obj,
                c0.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new
                c0 = c_new.copy()
        except Exception:
            continue
    
    return best_c, best_r


def _delaunay_gradient_descent(c0, max_time=30.0, lr=0.001, max_iter=2000):
    """
    Delaunay-based gradient ascent: maximize minimum circumradius of Delaunay triangles
    and distance to boundary. Uses exact geometric gradients.
    """
    start_time = time.time()
    c = c0.copy()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Precompute pair indices
    n = N
    n_pairs = n * (n - 1) // 2
    pair_i = np.zeros(n_pairs, dtype=int)
    pair_j = np.zeros(n_pairs, dtype=int)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_i[idx] = i
            pair_j[idx] = j
            idx += 1
    
    for it in range(max_iter):
        if time.time() > start_time + max_time - 2:
            break
        
        # Compute current radius
        r_current = _compute_radius(c)
        if r_current > best_r:
            best_r = r_current
            best_c = c.copy()
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(c)
        except Exception:
            break
        
        # Compute circumradius and gradients for each triangle
        # For triangle with vertices A, B, C, circumradius R = abc/(4*Area)
        # Gradient with respect to each vertex points away from circumcenter
        
        grad = np.zeros_like(c)
        
        for simplex in tri.simplices:
            i, j, k = simplex
            A = c[i]
            B = c[j]
            C = c[k]
            
            # Compute circumcenter
            ax, ay = A
            bx, by = B
            cx, cy = C
            
            d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
            if abs(d) < 1e-14:
                continue
            
            ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
            uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
            
            # Circumcenter
            ccx, ccy = ux, uy
            
            # Circumradius
            R = np.sqrt((ax - ccx)**2 + (ay - ccy)**2)
            
            # The minimum distance constraint is satisfied if R >= r_current for all triangles
            # We want to maximize the minimum R, so we push vertices away from circumcenter
            # when R < target (but we use the actual minimum as target)
            
            # Gradient: for vertex A, the direction to increase R is away from circumcenter
            # dR/dA = (A - CC) / R (normalized)
            
            # Actually, we want to maximize min(R, boundary_dist)
            # Use a soft approach: push each vertex away from circumcenter proportional to how much R constrains
            # But simpler: just push all vertices away from their local circumcenters with small step
            
            # For each vertex in this triangle, add gradient pointing away from circumcenter
            for idx_v in [i, j, k]:
                vx, vy = c[idx_v]
                dx = vx - ccx
                dy = vy - ccy
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 1e-14:
                    # Gradient of R w.r.t. vertex = (vertex - cc) / R
                    # This increases R when we move in this direction
                    grad[idx_v] += (dx / dist, dy / dist)
        
        # Add boundary gradient: push vertices away from boundary
        for i in range(n):
            x, y = c[i]
            # Distance to each boundary
            d_left = x + 0.5
            d_right = 0.5 - x
            d_bottom = y + 0.5
            d_top = 0.5 - y
            
            min_d = min(d_left, d_right, d_bottom, d_top)
            
            # Gradient of min distance: push away from nearest boundary
            if min_d == d_left:
                grad[i] += (1.0, 0.0)
            elif min_d == d_right:
                grad[i] += (-1.0, 0.0)
            elif min_d == d_bottom:
                grad[i] += (0.0, 1.0)
            elif min_d == d_top:
                grad[i] += (0.0, -1.0)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-14:
            grad = grad / grad_norm
        
        # Update with adaptive learning rate
        c_new = c + lr * grad
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        # Check if this improved
        r_new = _compute_radius(c_new)
        if r_new > r_current:
            c = c_new
            # Slightly increase lr if improving
            lr = min(lr * 1.01, 0.005)
        else:
            # Decrease lr
            lr *= 0.95
            if lr < 1e-6:
                break
    
    return best_c, best_r


def _cutting_plane_optimize(c0, max_time=30.0, max_cuts=500):
    """
    Cutting-plane method: iteratively build a set of active constraints
    (pairwise and boundary) and solve a QP subproblem to increase radius.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Precompute all pairs
    n = N
    n_pairs = n * (n - 1) // 2
    pair_i = np.zeros(n_pairs, dtype=int)
    pair_j = np.zeros(n_pairs, dtype=int)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_i[idx] = i
            pair_j[idx] = j
            idx += 1
    
    c_current = c0.copy()
    r_current = best_r
    
    # Track which constraints are active
    hot_pairs = set()
    hot_boundary = set()
    
    def _update_hot_sets(c, r, top_k=200):
        """Find the most violated constraints."""
        nonlocal hot_pairs, hot_boundary
        
        # Pairwise violations
        dists = pdist(c, metric='euclidean')
        violations = 2 * r - dists
        if len(violations) > top_k:
            top_indices = np.argpartition(violations, -top_k)[-top_k:]
        else:
            top_indices = np.arange(len(violations))
        
        new_hot_pairs = set()
        for k in top_indices:
            i = pair_i[k]
            j = pair_j[k]
            if violations[k] > -0.005:
                new_hot_pairs.add((i, j))
        
        # Boundary violations
        bnd_viol = r - (0.5 - np.abs(c))
        flat_bnd = bnd_viol.ravel()
        if len(flat_bnd) > top_k:
            top_bnd_indices = np.argpartition(flat_bnd, -top_k)[-top_k:]
        else:
            top_bnd_indices = np.arange(len(flat_bnd))
        
        new_hot_boundary = set()
        for k in top_bnd_indices:
            i = k // 2
            axis = k % 2
            if c[i, axis] < 0:
                sign = 0
            else:
                sign = 1
            if flat_bnd[k] > -0.005:
                new_hot_boundary.add((i, axis, sign))
        
        hot_pairs = new_hot_pairs
        hot_boundary = new_hot_boundary
    
    _update_hot_sets(c_current, r_current)
    
    iteration = 0
    while time.time() < start_time + max_time - 2 and iteration < max_cuts:
        iteration += 1
        
        n_vars = 2 * N + 1
        
        c_obj = np.zeros(n_vars)
        c_obj[-1] = -1.0
        
        A_rows = []
        b_vals = []
        
        for (i, j) in hot_pairs:
            diff = c_current[i] - c_current[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                d = np.array([1.0, 0.0])
            else:
                d = diff / dist
            
            row = np.zeros(n_vars)
            row[2 * i] = -d[0]
            row[2 * i + 1] = -d[1]
            row[2 * j] = d[0]
            row[2 * j + 1] = d[1]
            row[-1] = 2.0
            A_rows.append(row)
            b_vals.append(0.0)
        
        for (i, axis, sign) in hot_boundary:
            row = np.zeros(n_vars)
            if sign == 0:
                row[2 * i + axis] = -1.0
            else:
                row[2 * i + axis] = 1.0
            row[-1] = 1.0
            A_rows.append(row)
            b_vals.append(0.5)
        
        if not A_rows:
            break
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_vals)
        
        bounds = [(-0.4999, 0.4999)] * (2 * N) + [(0.0, 1.0)]
        
        try:
            result = linprog(
                c_obj,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method='highs',
                options={'presolve': True, 'time_limit': 2.0}
            )
            
            if result.status != 0:
                break
            
            x_sol = result.x
            c_new = x_sol[:2 * N].reshape(N, 2)
            r_new = x_sol[-1]
            
            c_new = np.clip(c_new, -0.4999, 0.4999)
            
            r_actual = _compute_radius(c_new)
            
            if r_actual > best_r:
                best_r = r_actual
                best_c = c_new.copy()
            
            c_current = c_new.copy()
            r_current = r_actual
            
            _update_hot_sets(c_current, r_current)
            
        except Exception:
            break
    
    return best_c, best_r


def _anneal_optimize(c0, max_time=20.0):
    """Simulated annealing with cluster moves."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    T = 0.05
    T_min = 0.001
    cooling = 0.98
    step_size = 0.02
    c_current = c0.copy()
    r_current = best_r
    
    while time.time() < start_time + max_time - 2:
        if np.random.random() < 0.4:
            idx = np.random.randint(N)
            tree = cKDTree(c_current)
            neighbors = tree.query_ball_point(c_current[idx], 0.06)
            c_trial = c_current.copy()
            displacement = np.random.normal(0, step_size * 0.5, size=2)
            for nb in neighbors:
                c_trial[nb] += displacement
        else:
            c_trial = c_current.copy()
            idx = np.random.randint(N)
            c_trial[idx] += np.random.normal(0, step_size, size=2)
        
        c_trial = np.clip(c_trial, -0.4999, 0.4999)
        r_trial = _compute_radius(c_trial)
        
        if r_trial >= r_current or np.random.random() < np.exp((r_trial - r_current) / T):
            c_current = c_trial
            r_current = r_trial
            if r_current > best_r:
                best_r = r_current
                best_c = c_current.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
            step_size *= 0.85
    
    return best_c, best_r


def _polish(c0, max_time=15.0):
    """Final high-precision polish with tight tolerances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    def obj(params):
        return _soft_min_energy(params, temperature=0.005)
    
    try:
        result = minimize(
            obj,
            best_c.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.4999, 0.4999)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
        )
        c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
    except Exception:
        pass
    
    if time.time() < start_time + max_time - 2:
        def obj2(params):
            return _soft_min_energy(params, temperature=0.002)
        
        try:
            result = minimize(
                obj2,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 50}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
        except Exception:
            pass
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: L-BFGS optimization from hex
    if time.time() < deadline - 40:
        max_opt_time = min(25.0, deadline - time.time() - 10)
        c_opt, r_opt = _lbfgs_optimize(c_hex, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Delaunay-based gradient ascent
    if time.time() < deadline - 35:
        remaining = deadline - time.time() - 10
        if remaining > 15:
            c_del, r_del = _delaunay_gradient_descent(best_c, max_time=min(20.0, remaining))
            if r_del > best_radius:
                best_radius = r_del
                best_c = c_del.copy()
    
    # Step 4: Cutting-plane optimization
    if time.time() < deadline - 25:
        remaining = deadline - time.time() - 10
        if remaining > 10:
            c_cp, r_cp = _cutting_plane_optimize(best_c, max_time=min(20.0, remaining))
            if r_cp > best_radius:
                best_radius = r_cp
                best_c = c_cp.copy()
    
    # Step 5: Multiple random restarts with L-BFGS
    n_restarts = 3
    for trial in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        c_init = _jittered_grid_init(seed=42 + trial)
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        remaining = deadline - time.time() - 10
        if remaining > 5:
            max_opt_time = min(12.0, remaining)
            c_opt, r_opt = _lbfgs_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 6: Annealing for refinement
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 8
        if remaining > 8:
            c_ann, r_ann = _anneal_optimize(best_c, max_time=min(12.0, remaining))
            if r_ann > best_radius:
                best_radius = r_ann
                best_c = c_ann.copy()
    
    # Step 7: Final polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 5:
            c_pol, r_pol = _polish(best_c, max_time=min(10.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
