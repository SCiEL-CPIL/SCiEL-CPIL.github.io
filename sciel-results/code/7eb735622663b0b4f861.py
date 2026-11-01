"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
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


def _energy(params, temperature=0.02):
    """Smooth energy function to maximize minimum distance."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    # Use pdist for all pairwise distances
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    T = max(temperature, 1e-8)
    
    # Soft min of pairwise distances - only consider close pairs for speed
    threshold = min_dist * 1.15 
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        # Numerically stable soft min
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    
    # For boundary distances: soft min
    b_min = np.min(boundary_dists)
    b_threshold = b_min * 1.15
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
    
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    
    return -soft_min_combined


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
        configs = [0.02, 0.01, 0.005, 0.002, 0.001]
    else:
        configs = list(temperature_schedule)
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 1:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
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


def _delaunay_gradient_polish(c, max_time=10.0, tol=1e-12):
    """
    Polish using Delaunay triangulation to identify critical pairs.
    The minimum distance in a 2D packing is achieved by an edge in the
    Delaunay triangulation (or by a boundary constraint). We use exact
    geometric gradients on the critical edges.
    """
    start_time = time.time()
    best_c = c.copy()
    best_r = _compute_radius(c)
    c = c.copy()
    
    for iteration in range(500):
        if time.time() > start_time + max_time - 0.1:
            break
        
        # Compute current radius
        r = _compute_radius(c)
        
        # Build Delaunay triangulation to find critical pairs
        try:
            tri = Delaunay(c)
            simplices = tri.simplices  # shape (M, 3)
            
            # Get all edges from triangles (unique)
            edges = set()
            for simplex in simplices:
                for i in range(3):
                    j = (i + 1) % 3
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
            edges = list(edges)
            
            if len(edges) == 0:
                break
            
            # Compute distances for all Delaunay edges
            edge_dists = np.zeros(len(edges))
            for k, (i, j) in enumerate(edges):
                edge_dists[k] = np.linalg.norm(c[i] - c[j])
            
            min_edge_dist = np.min(edge_dists)
            
            # Also check boundary distances
            boundary_dists = 0.5 - np.abs(c)
            min_boundary = np.min(boundary_dists)
            
            current_r = min(min_edge_dist / 2.0, min_boundary)
            
            if current_r > best_r + 1e-15:
                best_r = current_r
                best_c = c.copy()
            
            # Find critical pairs (within tolerance of minimum)
            tol_dist = 1e-8
            critical_edges = [edges[k] for k in range(len(edges)) 
                            if edge_dists[k] < min_edge_dist + tol_dist]
            
            # Find critical boundary points
            critical_boundary = []
            for i in range(N):
                for axis in range(2):
                    if boundary_dists[i, axis] < min_boundary + tol_dist:
                        critical_boundary.append((i, axis))
            
            if len(critical_edges) == 0 and len(critical_boundary) == 0:
                break
            
            # Compute gradient: we want to maximize r
            # For each critical edge (i,j), the constraint is |c_i - c_j| >= 2r
            # Gradient of |c_i - c_j| w.r.t. c_i is (c_i - c_j)/|c_i - c_j|
            # Gradient w.r.t. c_j is (c_j - c_i)/|c_j - c_i|
            
            grad = np.zeros_like(c)
            step_size = 1e-4
            
            # Edge contributions
            for (i, j) in critical_edges:
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    continue
                unit = diff / dist
                # Moving i away from j increases distance
                grad[i] += unit
                grad[j] -= unit
            
            # Boundary contributions
            for (i, axis) in critical_boundary:
                # If c[i, axis] > 0, boundary is at +0.5, gradient is -e_axis
                # If c[i, axis] < 0, boundary is at -0.5, gradient is +e_axis
                if c[i, axis] >= 0:
                    grad[i, axis] -= 1.0
                else:
                    grad[i, axis] += 1.0
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            grad /= grad_norm
            
            # Take a step
            c_new = c + step_size * grad
            c_new = np.clip(c_new, -0.4999, 0.4999)
            
            r_new = _compute_radius(c_new)
            if r_new > r:
                c = c_new
                # Reduce step size slightly for convergence
                step_size *= 0.95
            else:
                # Try smaller step
                for scale in [0.5, 0.1, 0.01]:
                    c_try = c + step_size * scale * grad
                    c_try = np.clip(c_try, -0.4999, 0.4999)
                    r_try = _compute_radius(c_try)
                    if r_try > r:
                        c = c_try
                        step_size *= 0.5
                        break
                else:
                    # No improvement, break
                    break
            
            # Check for convergence
            if r - best_r < tol:
                if r > best_r:
                    best_r = r
                    best_c = c.copy()
                # Check if we can still improve
                if np.linalg.norm(c - best_c) < 1e-10:
                    break
        
        except Exception:
            break
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization from best initial
    if time.time() < deadline - 20:
        max_opt_time = min(15.0, deadline - time.time() - 5)
        c_opt, r_opt = _local_optimize(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Delaunay-based polish
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            c_dela, r_dela = _delaunay_gradient_polish(best_c, max_time=min(10.0, remaining))
            if r_dela > best_radius:
                best_radius = r_dela
                best_c = c_dela.copy()
    
    # Step 4: A few random restarts
    np.random.seed(42)
    n_restarts = 3
    
    for trial in range(n_restarts):
        if time.time() > deadline - 15:
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
        
        # Add random perturbation
        c_init += np.random.normal(0, 0.01, size=(N, 2))
        c_init = np.clip(c_init, -0.4999, 0.4999)
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Local optimize
        remaining = deadline - time.time() - 5
        if remaining > 5:
            max_opt_time = min(8.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 5: Final high-precision polish with Delaunay + L-BFGS
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 1
        if remaining > 3:
            # First do Delaunay polish
            c_pol1, r_pol1 = _delaunay_gradient_polish(best_c, max_time=min(5.0, remaining * 0.5))
            if r_pol1 > best_radius:
                best_radius = r_pol1
                best_c = c_pol1.copy()
            
            # Then final L-BFGS with very tight tolerances
            remaining2 = deadline - time.time() - 1
            if remaining2 > 2:
                c_pol2, r_pol2 = _local_optimize(best_c, max_time=min(5.0, remaining2), 
                                                 temperature_schedule=[0.0005, 0.0002])
                if r_pol2 > best_radius:
                    best_radius = r_pol2
                    best_c = c_pol2.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
