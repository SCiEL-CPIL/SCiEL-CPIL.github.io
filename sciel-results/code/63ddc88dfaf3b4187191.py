"""Packing 200 equal circles in a unit square using cutting-plane (piecewise-linear) approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from scipy.optimize import minimize, LinearConstraint
import sys


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
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0.0
    
    for d in np.linspace(0.055, 0.075, 25):
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
    
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _get_active_pairs(c, r_target, margin=0.05):
    """Get pairs of points that are close to the minimum distance constraint."""
    tree = cKDTree(c)
    threshold = 2.0 * r_target + margin
    pairs = tree.query_pairs(r=threshold, output_type='ndarray')
    if len(pairs) == 0:
        pairs = np.array([], dtype=int).reshape(0, 2)
    return pairs


def _cutting_plane_optimize(c0, r_target, max_time=20.0, max_iter=50):
    """
    Cutting-plane method: solve a QP subproblem that maximizes r subject to
    linearized constraints from active pairs and boundaries.
    """
    start_time = time.time()
    n = c0.shape[0]
    m = n * 2
    
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Build the constraint set incrementally
    # Constraints: for each active pair (i,j): |c_i - c_j| >= 2r
    # Linearized: (c_i - c_j)·(c_i - c_j)/|c_i - c_j| >= 2r  (at the active point)
    # This is a linear constraint in c if we fix the direction
    
    # We'll maintain a list of active constraints
    # Each constraint is a linear inequality: a^T c >= b
    
    def solve_qp(c_current, r_current, constraints):
        """Solve the QP: maximize r subject to linear constraints and bounds."""
        # Variables: x = [c_0, c_1, ..., c_{n-1}, r]  (2n + 1 variables)
        # Objective: maximize r  =>  minimize -r
        
        # Build A matrix for linear constraints
        # Each constraint: a^T x >= b  =>  -a^T x <= -b
        
        if len(constraints) == 0:
            # No constraints, just maximize r within bounds
            # r is bounded by min boundary distance
            boundary_dists = 0.5 - np.abs(c_current)
            r_max = np.min(boundary_dists)
            return c_current, r_max
        
        # Convert constraints to standard form for scipy
        # scipy LinearConstraint: lb <= A @ x <= ub
        # We have a^T x >= b, so A = -a, lb = -b, ub = inf
        
        A_list = []
        lb_list = []
        ub_list = []
        
        for a, b in constraints:
            A_list.append(-a)
            lb_list.append(-b)
            ub_list.append(np.inf)
        
        if len(A_list) == 0:
            return c_current, _compute_radius(c_current)
        
        A = np.array(A_list)
        lb = np.array(lb_list)
        ub = np.array(ub_list)
        
        # Bounds on variables
        # c_i in [-0.5 + r, 0.5 - r] but r is also a variable
        # For simplicity, use loose bounds and let constraints handle it
        bounds_list = [(-0.5, 0.5)] * (2 * n)
        bounds_list.append((0.0, 0.5))  # r >= 0
        bounds = bounds_list
        
        # Objective: minimize -r
        def objective(x):
            return -x[-1]
        
        def gradient(x):
            grad = np.zeros(2 * n + 1)
            grad[-1] = -1.0
            return grad
        
        # Initial guess
        x0 = np.concatenate([c_current.flatten(), [r_current]])
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='SLSQP',
                bounds=bounds,
                constraints=[LinearConstraint(A, lb, ub)],
                options={'maxiter': 30, 'ftol': 1e-10, 'disp': False}
            )
            
            if result.success or result.fun < -r_current + 1e-8:
                c_new = result.x[:2*n].reshape(n, 2)
                r_new = result.x[-1]
                # Clip to valid range
                c_new = np.clip(c_new, -0.5 + r_new, 0.5 - r_new)
                r_new = _compute_radius(c_new)
                return c_new, r_new
            else:
                return c_current, r_current
        except Exception:
            return c_current, r_current
    
    # Iterative cutting-plane
    for iteration in range(max_iter):
        if time.time() > start_time + max_time - 2:
            break
        
        c_current = best_c
        r_current = _compute_radius(c_current)
        
        # Get active pairs
        pairs = _get_active_pairs(c_current, r_current, margin=0.02)
        
        # Build constraints
        constraints = []
        
        # Pairwise constraints
        for i, j in pairs:
            diff = c_current[i] - c_current[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            direction = diff / dist
            # Linearized: (c_i - c_j)·direction >= 2r
            # => c_i·direction - c_j·direction >= 2r
            a = np.zeros(2 * n + 1)
            a[i*2] = direction[0]
            a[i*2+1] = direction[1]
            a[j*2] = -direction[0]
            a[j*2+1] = -direction[1]
            a[-1] = -2.0  # coefficient of r
            b = 0.0
            constraints.append((a, b))
        
        # Boundary constraints
        for i in range(n):
            # x_i + 0.5 >= r  =>  x_i - r >= -0.5
            a = np.zeros(2 * n + 1)
            a[i*2] = 1.0
            a[-1] = -1.0
            constraints.append((a, -0.5))
            
            # 0.5 - x_i >= r  =>  -x_i - r >= -0.5
            a = np.zeros(2 * n + 1)
            a[i*2] = -1.0
            a[-1] = -1.0
            constraints.append((a, -0.5))
            
            # y_i + 0.5 >= r
            a = np.zeros(2 * n + 1)
            a[i*2+1] = 1.0
            a[-1] = -1.0
            constraints.append((a, -0.5))
            
            # 0.5 - y_i >= r
            a = np.zeros(2 * n + 1)
            a[i*2+1] = -1.0
            a[-1] = -1.0
            constraints.append((a, -0.5))
        
        # Solve QP
        c_new, r_new = solve_qp(c_current, r_current, constraints)
        
        if r_new > r_current + 1e-10:
            best_c = c_new
            best_r = r_new
        elif iteration > 5 and r_new <= r_current - 1e-8:
            # Stagnation, break
            break
    
    return best_c, best_r


def _soft_min_energy(c, temperature=0.01):
    """Soft-min energy function for L-BFGS-B optimization."""
    c = np.clip(c, -0.4999, 0.4999)
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    T = max(temperature, 1e-8)
    
    # Soft min of pairwise distances
    threshold = min_dist * 1.2
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    
    # Soft min of boundary distances
    b_min = np.min(boundary_dists)
    b_threshold = b_min * 1.2
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = min_boundary
    
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    
    return -soft_min_combined


def _local_optimize_soft(c0, max_time=15.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with soft-min energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    if temperature_schedule is None:
        configs = [0.02, 0.01, 0.005, 0.002, 0.001]
    else:
        configs = list(temperature_schedule)
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 2:
            break
        
        def obj(params, _T=T):
            c = params.reshape(-1, 2)
            return _soft_min_energy(c, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 80}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
            current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r


def _cluster_anneal(c0, max_time=12.0, n_iter=3000):
    """
    Simulated annealing with cluster moves.
    """
    start_time = time.time()
    current_c = c0.copy()
    current_r = _compute_radius(current_c)
    best_c = current_c.copy()
    best_r = current_r
    
    T_start = 0.004
    T_end = 0.0001
    
    for step in range(n_iter):
        if time.time() > start_time + max_time - 1:
            break
        
        T = T_start * (T_end / T_start) ** (step / n_iter)
        
        # Pick a random cluster
        idx = np.random.randint(0, N)
        tree = cKDTree(current_c)
        min_dist = 2.0 * current_r
        
        # Find neighbors within 1.5 * min_dist
        neighbors = tree.query_ball_point(current_c[idx], r=min_dist * 1.5)
        cluster = neighbors[:min(10, len(neighbors))]
        
        # Generate random perturbation
        delta = np.random.normal(0, T, size=(len(cluster), 2))
        
        new_c = current_c.copy()
        for k, idx_k in enumerate(cluster):
            new_c[idx_k] += delta[k]
        
        new_c = np.clip(new_c, -0.4999, 0.4999)
        
        new_r = _compute_radius(new_c)
        
        if new_r > current_r:
            current_c = new_c
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_c = new_c.copy()
        else:
            delta_r = new_r - current_r
            if delta_r > -1e-10:
                current_c = new_c
                current_r = new_r
            else:
                prob = np.exp(delta_r / T)
                if np.random.random() < prob:
                    current_c = new_c
                    current_r = new_r
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization with soft-min
    if time.time() < deadline - 50:
        remaining = deadline - time.time() - 20
        max_opt_time = min(12.0, remaining)
        c_opt, r_opt = _local_optimize_soft(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Cutting-plane optimization
    if time.time() < deadline - 35:
        remaining = deadline - time.time() - 15
        max_cp_time = min(15.0, remaining)
        c_cp, r_cp = _cutting_plane_optimize(best_c, best_radius, max_time=max_cp_time)
        if r_cp > best_radius:
            best_radius = r_cp
            best_c = c_cp.copy()
    
    # Step 4: Cluster annealing to escape local minima
    if time.time() < deadline - 25:
        remaining = deadline - time.time() - 10
        max_anneal_time = min(10.0, remaining)
        c_anneal, r_anneal = _cluster_anneal(best_c, max_time=max_anneal_time)
        if r_anneal > best_radius:
            best_radius = r_anneal
            best_c = c_anneal.copy()
        
        # Local optimize after annealing
        if time.time() < deadline - 15:
            remaining = deadline - time.time() - 5
            max_opt_time = min(8.0, remaining)
            c_opt, r_opt = _local_optimize_soft(best_c, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 5: Another round of cutting-plane
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 3
        max_cp_time = min(8.0, remaining)
        c_cp, r_cp = _cutting_plane_optimize(best_c, best_radius, max_time=max_cp_time)
        if r_cp > best_radius:
            best_radius = r_cp
            best_c = c_cp.copy()
    
    # Step 6: Final soft-min polish with very low temperature
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 1
        if remaining > 2:
            c_pol, r_pol = _local_optimize_soft(best_c, max_time=min(4.0, remaining),
                                                temperature_schedule=[0.0005, 0.0002, 0.0001])
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
