"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay
from scipy.optimize import NonlinearConstraint


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


def _get_delaunay_edges(tri):
    """Get unique edges from Delaunay triangulation."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a = simplex[i]
            b = simplex[(i + 1) % 3]
            if a > b:
                a, b = b, a
            edges.add((a, b))
    return np.array(list(edges))


def _soft_min_objective_and_grad(params, beta, tri_edges):
    """
    Compute soft-min objective and gradient for Delaunay edges and boundary constraints.
    We maximize the min of:
    1. Half the length of Delaunay edges
    2. Distance to boundary for all points
    
    We use a soft-min approximation: -log-sum-exp(-beta * x) / beta
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    # Collect all constraint values
    vals = []
    
    # 1. Boundary constraints (4 sides per point)
    b_left = c[:, 0] + 0.5
    b_right = 0.5 - c[:, 0]
    b_bottom = c[:, 1] + 0.5
    b_top = 0.5 - c[:, 1]
    
    boundary_vals = np.concatenate([b_left, b_right, b_bottom, b_top])
    vals.append(boundary_vals)
    
    # 2. Delaunay Edge constraints
    if tri_edges is not None and len(tri_edges) > 0:
        i_idx = tri_edges[:, 0]
        j_idx = tri_edges[:, 1]
        
        diffs = c[i_idx] - c[j_idx]
        dists = np.sqrt(np.sum(diffs**2, axis=1) + 1e-16)
        edge_vals = dists / 2.0
        vals.append(edge_vals)
        
        unit_vecs = diffs / dists[:, np.newaxis]
    else:
        i_idx = np.array([], dtype=int)
        j_idx = np.array([], dtype=int)
        unit_vecs = np.array([])
    
    all_vals = np.concatenate(vals)
    
    # Soft-min calculation
    min_val = np.min(all_vals)
    shifted = beta * (all_vals - min_val)
    shifted = np.clip(shifted, -700, 700)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp < 1e-300:
        sum_exp = 1e-300
    
    lse = np.log(sum_exp) - beta * min_val
    soft_min_val = min_val + lse / beta
    
    obj = -soft_min_val
    
    # Weights for gradient
    weights = exp_shifted / sum_exp
    
    # Accumulate gradient
    grad_c = np.zeros_like(c)
    
    # 1. Boundary gradients
    w_b = weights[:4*n]
    grad_c[:, 0] += w_b[:n] - w_b[n:2*n]
    grad_c[:, 1] += w_b[2*n:3*n] - w_b[3*n:4*n]
    
    # 2. Edge gradients
    if len(tri_edges) > 0:
        w_e = weights[4*n:]
        factor = 0.5 * w_e[:, np.newaxis] * unit_vecs
        np.add.at(grad_c, i_idx, factor)
        np.add.at(grad_c, j_idx, -factor)
        
    return obj, grad_c.flatten()


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 20):
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


def _local_optimize(c0, max_time=10.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    if temperature_schedule is None:
        configs = [0.02, 0.01, 0.005, 0.002]
    else:
        configs = list(temperature_schedule)
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 1:
            break
        
        def obj(params, _T=T):
            c = params.reshape(-1, 2)
            c = np.clip(c, -0.4999, 0.4999)
            
            boundary_dists = 0.5 - np.abs(c)
            min_boundary = np.min(boundary_dists)
            
            if N < 2:
                return -min_boundary
            
            dists = pdist(c, metric='euclidean')
            min_dist = np.min(dists)
            
            _T = max(_T, 1e-8)
            
            threshold = min_dist * 1.15 
            mask = dists < threshold
            if np.any(mask):
                d_sub = dists[mask]
                d_min_sub = np.min(d_sub)
                soft_min_pair = d_min_sub - _T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / _T)))
            else:
                soft_min_pair = min_dist
            
            b_min = np.min(boundary_dists)
            b_threshold = b_min * 1.15
            b_mask = boundary_dists < b_threshold
            if np.any(b_mask):
                b_sub = boundary_dists[b_mask]
                b_min_sub = np.min(b_sub)
                soft_min_bound = b_min_sub - _T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / _T)))
            else:
                soft_min_bound = min_boundary
            
            r_pair = soft_min_pair / 2.0
            r_bound = soft_min_bound
            
            r_vals = np.array([r_pair, r_bound])
            m = np.min(r_vals)
            soft_min_combined = m - _T * np.log(np.sum(np.exp(-(r_vals - m) / _T)))
            
            return -soft_min_combined
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 100}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
            else:
                current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r


def _delaunay_optimize(c0, max_time=30.0):
    """
    Optimize using Delaunay-based soft-min with gradient.
    This is the core improvement from the inspiration.
    """
    start_time = time.time()
    c = c0.copy()
    n = c.shape[0]
    
    # Initial triangulation
    try:
        tri = Delaunay(c)
        edges = _get_delaunay_edges(tri)
    except Exception:
        c += np.random.normal(0, 1e-6, c.shape)
        c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
        try:
            tri = Delaunay(c)
            edges = _get_delaunay_edges(tri)
        except Exception:
            return c0, _compute_radius(c0)
        
    beta = 50.0
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Optimization loop with annealing beta
    # We re-compute Delaunay every few steps to keep edges up to date
    max_iter_per_chunk = 50
    retri_freq = 10
    chunk_count = 0
    
    while time.time() < start_time + max_time - 5:
        # Define functions that capture current 'edges' and 'beta'
        def _f(p):
            return _soft_min_objective_and_grad(p, beta, edges)[0]
            
        def _g(p):
            return _soft_min_objective_and_grad(p, beta, edges)[1]
            
        try:
            result = minimize(
                _f,
                c.flatten(),
                jac=_g,
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
                options={'maxiter': max_iter_per_chunk, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                c = c_new
            else:
                c = c_new
            
            chunk_count += 1
            
            # Re-triangulate periodically
            if chunk_count % retri_freq == 0:
                try:
                    tri = Delaunay(c)
                    edges = _get_delaunay_edges(tri)
                except Exception:
                    pass
                
                # Anneal beta
                beta = max(20.0, beta * 0.8)
                
        except Exception:
            break
    
    return best_c, best_r


def _slsqp_critical_polish(c0, r_target, max_time=10.0, num_critical_pairs=50):
    """
    Polish using SLSQP on a subset of critical constraints.
    """
    start_time = time.time()
    n = c0.shape[0]
    m = n * 2
    
    dists = squareform(pdist(c0, metric='euclidean'))
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    boundary_dists = 0.5 - np.abs(c0)
    min_boundary_dist = np.min(boundary_dists)
    
    sorted_indices = np.argsort(pair_dists)
    critical_pair_indices = sorted_indices[:num_critical_pairs]
    
    ci_idx = i_idx[critical_pair_indices]
    cj_idx = j_idx[critical_pair_indices]
    
    b_thresh = min_boundary_dist * 1.1 + 1e-6
    critical_boundary_points = np.where(np.min(boundary_dists, axis=1) < b_thresh)[0]
    
    n_crit_pairs = len(ci_idx)
    n_crit_bounds = len(critical_boundary_points)
    
    if n_crit_pairs == 0 and n_crit_bounds == 0:
        return c0, r_target

    def pairwise_constraints(params):
        c = params.reshape(n, 2)
        ci = c[ci_idx]
        cj = c[cj_idx]
        diff = ci - cj
        dists_sub = np.sqrt(np.sum(diff**2, axis=1))
        return dists_sub - 2.0 * r_target
    
    def grad_pairwise_constraints(params):
        c = params.reshape(n, 2)
        ci = c[ci_idx]
        cj = c[cj_idx]
        diff = ci - cj
        dists_sub = np.sqrt(np.sum(diff**2, axis=1))
        dists_safe = np.where(dists_sub > 1e-12, dists_sub, 1e-12)
        
        J = np.zeros((n_crit_pairs, m))
        
        for k in range(n_crit_pairs):
            i = ci_idx[k]
            j = cj_idx[k]
            d = dists_sub[k]
            if d < 1e-12: continue
            inv_dk = 1.0/d
            dx = (ci[k, 0] - cj[k, 0]) * inv_dk
            dy = (ci[k, 1] - cj[k, 1]) * inv_dk
            
            J[k, 2*i] += dx
            J[k, 2*i+1] += dy
            J[k, 2*j] -= dx
            J[k, 2*j+1] -= dy
        return J

    def boundary_constraints(params):
        c = params.reshape(n, 2)
        x = c[:, 0]
        y = c[:, 1]
        
        res = []
        for idx in critical_boundary_points:
            res.append(x[idx] + 0.5 - r_target)
            res.append(0.5 - r_target - x[idx])
            res.append(y[idx] + 0.5 - r_target)
            res.append(0.5 - r_target - y[idx])
        return np.array(res)
    
    def grad_boundary_constraints(params):
        J = np.zeros((4 * n_crit_bounds, m))
        for k, idx in enumerate(critical_boundary_points):
            J[4*k, 2*idx] = 1.0
            J[4*k+1, 2*idx] = -1.0
            J[4*k+2, 2*idx+1] = 1.0
            J[4*k+3, 2*idx+1] = -1.0
        return J

    constraints = []
    if n_crit_pairs > 0:
        constraints.append(NonlinearConstraint(pairwise_constraints, 0.0, np.inf, jac=grad_pairwise_constraints))
    if n_crit_bounds > 0:
        constraints.append(NonlinearConstraint(boundary_constraints, 0.0, np.inf, jac=grad_boundary_constraints))
    
    def objective(params):
        return 0.0
    def grad_objective(params):
        return np.zeros(m)
    
    bounds = [(-0.5 + r_target - 0.01, 0.5 - r_target + 0.01)] * m
    
    try:
        result = minimize(
            objective,
            c0.flatten(),
            method='SLSQP',
            jac=grad_objective,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        c_new = np.clip(result.x.reshape(n, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(c_new)
        
        if r_new > r_target:
            return c_new, r_new
        else:
            return c0, r_target
    except Exception:
        return c0, r_target


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 5s before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Initial local optimization
    if time.time() < deadline - 30:
        max_opt_time = min(10.0, deadline - time.time() - 20)
        c_opt, r_opt = _local_optimize(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Delaunay-based optimization (main improvement)
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 10
        if remaining > 10:
            c_del, r_del = _delaunay_optimize(best_c, max_time=min(30.0, remaining))
            if r_del > best_radius:
                best_radius = r_del
                best_c = c_del.copy()
    
    # Step 4: A few random restarts with Delaunay optimization
    np.random.seed(42)
    n_restarts = 2
    
    for trial in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        # Jittered grid initialization
        m = int(np.ceil(np.sqrt(N)))
        grid_x, grid_y = np.meshgrid(
            np.linspace(-0.45, 0.45, m),
            np.linspace(-0.45, 0.45, m)
        )
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        np.random.shuffle(grid)
        c_init = grid[:N].copy()
        c_init += np.random.normal(0, 0.01, size=(N, 2))
        c_init = np.clip(c_init, -0.4999, 0.4999)
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Delaunay optimize from this restart
        remaining = deadline - time.time() - 15
        if remaining > 10:
            c_opt, r_opt = _delaunay_optimize(c_init, max_time=min(15.0, remaining))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 5: SLSQP critical pair polish
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        
        c_pol, r_pol = _slsqp_critical_polish(best_c, best_radius, max_time=5.0, num_critical_pairs=100)
        
        if r_pol > best_radius:
            best_radius = r_pol
            best_c = c_pol.copy()
        else:
            break
    
    # Step 6: Final Delaunay polish with lower beta
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 5:
            # Run another round of Delaunay optimization with lower beta for refinement
            c_final, r_final = _delaunay_optimize(best_c, max_time=remaining)
            if r_final > best_radius:
                best_radius = r_final
                best_c = c_final.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
