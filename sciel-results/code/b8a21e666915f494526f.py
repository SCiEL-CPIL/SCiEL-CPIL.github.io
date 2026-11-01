"""Rewritten program for packing 200 equal circles in a unit square using Delaunay-based optimization."""
import numpy as np
import time
from scipy.spatial import Delaunay, distance
from scipy.optimize import minimize

N = 200
SQRT3 = np.sqrt(3.0)

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

def _compute_radius(c):
    """Compute the exact packing radius for a configuration."""
    n = c.shape[0]
    # Boundary distances
    # dist to boundary is min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5)
    # Since c is in [-0.5, 0.5], dist = 0.5 - max(|x|, |y|) is wrong.
    # dist to boundary = min(0.5 - c[:,0], c[:,0] + 0.5, 0.5 - c[:,1], c[:,1] + 0.5)
    # Actually, just 0.5 - np.max(np.abs(c), axis=1) is NOT correct for square.
    # dist to boundary for point (x,y) in [-0.5, 0.5]^2 is min(0.5-x, x+0.5, 0.5-y, y+0.5)
    # Which simplifies to 0.5 - max(|x|, |y|)? No.
    # If x=0.4, y=0.1: dist = min(0.1, 0.9, 0.4, 0.6) = 0.1.
    # 0.5 - max(0.4, 0.1) = 0.1. Correct.
    # If x=0.1, y=0.4: dist = 0.1. 0.5 - 0.4 = 0.1. Correct.
    # So dist to boundary = 0.5 - np.max(np.abs(c), axis=1).
    
    boundary_dists = 0.5 - np.max(np.abs(c), axis=1)
    
    if n < 2:
        return np.min(boundary_dists)
    
    # Pairwise distances
    # For N=200, pdist is fast enough for exact check
    dists = distance.pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, np.min(boundary_dists))

def _soft_min_objective_and_grad(params, beta, tri_edges, boundary_active_mask=None):
    """
    Compute soft-min objective and gradient for Delaunay edges and boundary constraints.
    We maximize the min of:
    1. Half the length of Delaunay edges
    2. Distance to boundary for all points (or active ones)
    
    We use a soft-min approximation: -log-sum-exp(-beta * x) / beta
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    # Collect all constraint values
    vals = []
    grads = np.zeros_like(c)
    
    # 1. Boundary constraints
    # v = 0.5 - max(|x|, |y|) is non-smooth at |x|=|y|.
    # Better to use 4 separate boundary distances? Or just use the 4 sides.
    # dist to left: x + 0.5
    # dist to right: 0.5 - x
    # dist to bottom: y + 0.5
    # dist to top: 0.5 - y
    # The min of these 4 is the dist to boundary.
    # To keep it smooth, we can soft-min these 4 per point, or just include all 4N values in the global soft-min.
    # Including all 4N values is safer and simpler.
    
    b_left = c[:, 0] + 0.5
    b_right = 0.5 - c[:, 0]
    b_bottom = c[:, 1] + 0.5
    b_top = 0.5 - c[:, 1]
    
    boundary_vals = np.concatenate([b_left, b_right, b_bottom, b_top])
    vals.append(boundary_vals)
    
    # Gradients for boundary parts
    # d(b_left)/dx = 1, d(b_right)/dx = -1, etc.
    grad_boundary = np.zeros((4 * n, 2))
    grad_boundary[:n, 0] = 1.0
    grad_boundary[n:2*n, 0] = -1.0
    grad_boundary[2*n:3*n, 1] = 1.0
    grad_boundary[3*n:4*n, 1] = -1.0
    
    # 2. Delaunay Edge constraints
    if tri_edges is not None and len(tri_edges) > 0:
        i_idx = tri_edges[:, 0]
        j_idx = tri_edges[:, 1]
        
        diffs = c[i_idx] - c[j_idx]
        dists = np.sqrt(np.sum(diffs**2, axis=1) + 1e-16)
        edge_vals = dists / 2.0
        vals.append(edge_vals)
        
        # Gradients for edge parts
        # v = 0.5 * ||c_i - c_j||
        # d(v)/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # d(v)/dc_j = -0.5 * (c_i - c_j) / ||c_i - c_j||
        unit_vecs = diffs / dists[:, np.newaxis]
        
        grad_edges = np.zeros((len(tri_edges), 2))
        # We need to accumulate into the full gradient array later, 
        # but here we just compute the local gradient for each edge value.
        # The soft-min gradient will be a weighted sum of these local gradients.
        
    all_vals = np.concatenate(vals)
    
    # Soft-min calculation
    min_val = np.min(all_vals)
    shifted = beta * (all_vals - min_val)
    shifted = np.clip(shifted, -700, 700)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    lse = np.log(sum_exp) - beta * min_val
    soft_min_val = min_val + lse / beta
    
    obj = -soft_min_val
    
    # Weights for gradient
    weights = exp_shifted / sum_exp
    
    # Accumulate gradient
    grad_c = np.zeros_like(c)
    
    # 1. Boundary gradients
    # weights[0 : 4*n] correspond to boundary vals
    w_b = weights[:4*n]
    # w_b[0:n] for left, w_b[n:2n] for right, etc.
    # grad_c[:, 0] += w_b[0:n] * 1.0 + w_b[n:2*n] * (-1.0)
    # grad_c[:, 1] += w_b[2*n:3*n] * 1.0 + w_b[3*n:4*n] * (-1.0)
    grad_c[:, 0] += w_b[:n] - w_b[n:2*n]
    grad_c[:, 1] += w_b[2*n:3*n] - w_b[3*n:4*n]
    
    # 2. Edge gradients
    if tri_edges is not None and len(tri_edges) > 0:
        w_e = weights[4*n:]
        # unit_vecs is (M, 2)
        # For each edge k, grad_c[i] += w_e[k] * 0.5 * unit_vecs[k]
        # grad_c[j] -= w_e[k] * 0.5 * unit_vecs[k]
        
        # Vectorized accumulation
        # We can use np.add.at
        # factor = 0.5 * w_e[:, np.newaxis] * unit_vecs
        factor = 0.5 * w_e[:, np.newaxis] * unit_vecs
        
        np.add.at(grad_c, i_idx, factor)
        np.add.at(grad_c, j_idx, -factor)
        
    return obj, grad_c.flatten()

def _optimize_delaunay(c0, max_time=100.0):
    """
    Optimize using Delaunay-based soft-min.
    """
    start_time = time.time()
    c = c0.copy()
    n = c.shape[0]
    
    # Initial triangulation
    try:
        tri = Delaunay(c)
        edges = _get_delaunay_edges(tri)
    except Exception:
        # Fallback if triangulation fails (e.g., collinear points)
        # Just use all pairs? No, that's expensive.
        # Add small noise and retry
        c += np.random.normal(0, 1e-6, c.shape)
        tri = Delaunay(c)
        edges = _get_delaunay_edges(tri)
        
    beta = 50.0
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Optimization loop with annealing beta
    # We re-compute Delaunay every few steps to keep edges up to date
    step_count = 0
    retri_freq = 10
    
    def _obj_and_jac(params):
        return _soft_min_objective_and_grad(params, beta, edges)
        
    # We use L-BFGS-B
    # Note: We need to update 'edges' inside the closure or pass it via a mutable container
    # Since edges change, we can't easily use a static closure for L-BFGS-B if we want to update them during the run.
    # Strategy: Run L-BFGS-B for a short duration, then re-triangulate, then continue.
    
    max_iter_per_chunk = 50
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
                # If no improvement, accept anyway to move out of local minima?
                # Or keep best? Usually keep best for stability.
                # Let's move to c_new to explore, but track best.
                c = c_new
                
        except Exception as e:
            # If optimization fails, break or continue
            break
        
        # Increase beta to sharpen the soft-min
        beta *= 1.5
        if beta > 500:
            beta = 500
            
        # Re-triangulate periodically
        step_count += 1
        if step_count % retri_freq == 0:
            try:
                # Add tiny noise to avoid degenerate triangulations
                c_perturbed = c + np.random.normal(0, 1e-8, c.shape)
                c_perturbed = np.clip(c_perturbed, -0.5 + 1e-12, 0.5 - 1e-12)
                tri = Delaunay(c_perturbed)
                edges = _get_delaunay_edges(tri)
            except Exception:
                pass
                
    return best_c, best_r

def _high_precision_polish(c0, max_time=10.0):
    """
    Final polish using only active constraints (tightest edges and boundaries).
    """
    start_time = time.time()
    c = c0.copy()
    n = c.shape[0]
    r = _compute_radius(c)
    
    # Identify active constraints
    # 1. Boundary: points where dist < r + 1e-6
    # 2. Edges: Delaunay edges where dist/2 < r + 1e-6
    
    tri = Delaunay(c)
    edges = _get_delaunay_edges(tri)
    
    # Compute all constraint values
    b_left = c[:, 0] + 0.5
    b_right = 0.5 - c[:, 0]
    b_bottom = c[:, 1] + 0.5
    b_top = 0.5 - c[:, 1]
    boundary_vals = np.concatenate([b_left, b_right, b_bottom, b_top])
    
    diffs = c[edges[:, 0]] - c[edges[:, 1]]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    edge_vals = dists / 2.0
    
    all_vals = np.concatenate([boundary_vals, edge_vals])
    min_val = np.min(all_vals)
    
    # Active mask: within 1e-5 of min_val
    active_mask = all_vals < (min_val + 1e-5)
    
    # Extract active indices
    n_b = len(boundary_vals)
    active_b_idx = np.where(active_mask[:n_b])[0]
    active_e_idx = np.where(active_mask[n_b:])[0]
    
    if len(active_b_idx) == 0 and len(active_e_idx) == 0:
        return c, r
        
    # Define objective for active set only
    # This is a much smaller problem
    
    def _active_obj_and_grad(params):
        c_local = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        vals = []
        grads = np.zeros_like(c_local)
        
        # Boundary
        if len(active_b_idx) > 0:
            # Map global boundary index to local point and side
            # 0:n left, n:2n right, 2n:3n bottom, 3n:4n top
            bi = active_b_idx
            b_vals = []
            b_grads = []
            
            for idx in bi:
                if idx < n:
                    b_vals.append(c_local[idx, 0] + 0.5)
                    b_grads.append((idx, 0, 1.0))
                elif idx < 2*n:
                    b_vals.append(0.5 - c_local[idx, 0])
                    b_grads.append((idx, 0, -1.0))
                elif idx < 3*n:
                    b_vals.append(c_local[idx, 1] + 0.5)
                    b_grads.append((idx, 1, 1.0))
                else:
                    b_vals.append(0.5 - c_local[idx, 1])
                    b_grads.append((idx, 1, -1.0))
            
            b_vals = np.array(b_vals)
            vals.append(b_vals)
            
        # Edges
        if len(active_e_idx) > 0:
            e_idx = active_e_idx
            i_idx = edges[e_idx, 0]
            j_idx = edges[e_idx, 1]
            diffs = c_local[i_idx] - c_local[j_idx]
            dists = np.sqrt(np.sum(diffs**2, axis=1) + 1e-16)
            e_vals = dists / 2.0
            vals.append(e_vals)
            
        if not vals:
            return 0.0, np.zeros_like(params)
            
        all_v = np.concatenate(vals)
        min_v = np.min(all_v)
        beta = 1000.0 # High beta for precision
        shifted = beta * (all_v - min_v)
        shifted = np.clip(shifted, -700, 700)
        exp_s = np.exp(shifted)
        sum_s = np.sum(exp_s)
        lse = np.log(sum_s) - beta * min_v
        soft_min = min_v + lse / beta
        obj = -soft_min
        
        w = exp_s / sum_s
        
        grad_c = np.zeros_like(c_local)
        
        # Apply boundary grads
        if len(active_b_idx) > 0:
            w_b = w[:len(b_vals)]
            for k, (idx, dim, sign) in enumerate(b_grads):
                grad_c[idx, dim] += w_b[k] * sign
                
        # Apply edge grads
        if len(active_e_idx) > 0:
            w_e = w[len(b_vals):]
            unit_vecs = diffs / dists[:, np.newaxis]
            factor = 0.5 * w_e[:, np.newaxis] * unit_vecs
            np.add.at(grad_c, i_idx, factor)
            np.add.at(grad_c, j_idx, -factor)
            
        return obj, grad_c.flatten()
        
    try:
        result = minimize(
            lambda p: _active_obj_and_grad(p)[0],
            c.flatten(),
            jac=lambda p: _active_obj_and_grad(p)[1],
            method='L-BFGS-B',
            bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
        )
        c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(c_new)
        if r_new > r:
            c = c_new
            r = r_new
    except Exception:
        pass
        
    return c, r

def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    # Try a few spacing parameters
    for d in np.linspace(0.06, 0.08, 15):
        points = []
        row_spacing = d * SQRT3 / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_c = _initialize_hexagonal()
    best_r = _compute_radius(best_c)
    
    # Phase 1: Delaunay-based optimization
    if time.time() < deadline - 20:
        c_opt, r_opt = _optimize_delaunay(best_c, max_time=min(100.0, deadline - time.time() - 15))
        if r_opt > best_r:
            best_c = c_opt
            best_r = r_opt
            
    # Phase 2: High precision polish
    if time.time() < deadline - 5:
        c_polish, r_polish = _high_precision_polish(best_c, max_time=min(15.0, deadline - time.time() - 2))
        if r_polish > best_r:
            best_c = c_polish
            best_r = r_polish
            
    # Ensure valid output
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
