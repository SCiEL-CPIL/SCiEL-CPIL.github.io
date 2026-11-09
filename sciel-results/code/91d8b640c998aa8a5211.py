"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 26

# EVOLVE-BLOCK-START

def _validity_check(x, y, r, tol=1e-9):
    """Returns True if configuration is valid, False otherwise"""
    n = len(r)
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
        return False
    # Vectorized overlap check
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx*dx + dy*dy)
    if np.any(dist < r[i_idx] + r[j_idx] - tol):
        return False
    return True

def _make_valid(x, y, r, tol=1e-9):
    """Ensure configuration is valid by shrinking radii if needed"""
    r = r.copy()
    # First ensure bounds
    for i in range(len(r)):
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
        if r[i] > max_r:
            r[i] = max(0, max_r)
    
    i_idx, j_idx = np.triu_indices(len(r), k=1)
    
    # Iteratively fix overlaps
    for _ in range(500):
        # Vectorized overlap calculation
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        overlaps = r[i_idx] + r[j_idx] - dist
        
        max_overlap = np.max(overlaps)
        if max_overlap <= tol:
            break
        
        # Find the pair with max overlap
        max_pair_idx = np.argmax(overlaps)
        max_i = i_idx[max_pair_idx]
        max_j = j_idx[max_pair_idx]
        
        # Shrink the larger circle by half the overlap plus a small buffer
        shrink = max_overlap / 2 + 1e-7
        if r[max_i] >= r[max_j]:
            r[max_i] = max(0, r[max_i] - shrink)
        else:
            r[max_j] = max(0, r[max_j] - shrink)
    
    # Final safety: if still invalid, scale down all radii
    if not _validity_check(x, y, r, tol):
        r *= 0.95
        for _ in range(50):
            if _validity_check(x, y, r, tol):
                break
            r *= 0.9
        
    return x, y, r

def _maximize_radii_vectorized(x, y, r, iterations=100, tol=1e-10):
    """Try to increase each radius to its maximum possible value - vectorized"""
    r = r.copy()
    n = len(r)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for _ in range(iterations):
        changed = False
        for i in range(n):
            # Max radius from boundary
            r_bound = min(x[i], 1 - x[i], y[i], 1 - y[i])
            
            # Vectorized: compute max radius from all neighbors
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            dx = x[i] - x[mask]
            dy = y[i] - y[mask]
            dist = np.sqrt(dx*dx + dy*dy)
            r_neighbor = np.min(np.maximum(0, dist - r[mask] - tol))
            
            new_r = min(r_bound, r_neighbor)
            if new_r > r[i] + 1e-12:
                r[i] = new_r
                changed = True
        
        if not changed:
            break
    
    # Ensure validity after maximization
    if not _validity_check(x, y, r, 1e-9):
        x, y, r = _make_valid(x, y, r)
    
    return x, y, r

def _identify_contact_graph(x, y, r, threshold_factor=1.01):
    """
    Identify contact graph: edges between circles that are nearly tangent
    or between circle and boundary.
    Returns adjacency matrix and boundary contacts.
    """
    n = len(r)
    adj = np.zeros((n, n), dtype=bool)
    boundary = np.zeros((n, 4), dtype=bool)  # left, right, bottom, top
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx*dx + dy*dy)
    threshold = r[i_idx] + r[j_idx]
    
    # Contact if distance is close to sum of radii
    for k in range(len(i_idx)):
        if dist[k] <= threshold[k] * threshold_factor:
            i, j = i_idx[k], j_idx[k]
            adj[i, j] = True
            adj[j, i] = True
    
    # Boundary contacts
    for i in range(n):
        if x[i] <= r[i] * threshold_factor:
            boundary[i, 0] = True  # left
        if x[i] >= 1 - r[i] * threshold_factor:
            boundary[i, 1] = True  # right
        if y[i] <= r[i] * threshold_factor:
            boundary[i, 2] = True  # bottom
        if y[i] >= 1 - r[i] * threshold_factor:
            boundary[i, 3] = True  # top
    
    return adj, boundary

def _solve_socp_for_contact_graph(x0, y0, r0, adj, boundary, max_iter=20, tol=1e-12):
    """
    Solve SOCP for a fixed contact graph to maximize sum of radii.
    
    The SOCP formulation:
    - Variables: x_i, y_i, r_i for each circle
    - Objective: maximize sum(r_i)
    - Constraints:
      * Boundary: x_i >= r_i, 1-x_i >= r_i, y_i >= r_i, 1-y_i >= r_i
      * For each edge (i,j) in contact graph: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
        This is a second-order cone constraint.
    
    We use scipy's minimize with SLSQP and carefully constructed constraints,
    or we can use a custom SOCP solver. For now, we'll use an iterative
    approach that alternates between fixing positions and optimizing radii.
    """
    n = len(r0)
    x = x0.copy()
    y = y0.copy()
    r = r0.copy()
    
    # Extract edge list from adjacency matrix
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append((i, j))
    
    # Boundary contact lists
    left_circles = [i for i in range(n) if boundary[i, 0]]
    right_circles = [i for i in range(n) if boundary[i, 1]]
    bottom_circles = [i for i in range(n) if boundary[i, 2]]
    top_circles = [i for i in range(n) if boundary[i, 3]]
    
    def socp_objective(params):
        """
        Objective: minimize -sum(r)
        params: [x0..xn-1, y0..yn-1, r0..rn-1]
        """
        x_p = params[:n]
        y_p = params[n:2*n]
        r_p = params[2*n:3*n]
        
        # Penalty for violating contact constraints
        pen = 0.0
        for i, j in edges:
            dx = x_p[i] - x_p[j]
            dy = y_p[i] - y_p[j]
            dist = np.sqrt(dx*dx + dy*dy)
            # Constraint: dist >= r_i + r_j
            violation = np.maximum(0, r_p[i] + r_p[j] - dist)
            pen += violation**2
        
        # Boundary penalties
        for i in left_circles:
            pen += np.maximum(0, r_p[i] - x_p[i])**2
        for i in right_circles:
            pen += np.maximum(0, x_p[i] - (1 - r_p[i]))**2
        for i in bottom_circles:
            pen += np.maximum(0, r_p[i] - y_p[i])**2
        for i in top_circles:
            pen += np.maximum(0, y_p[i] - (1 - r_p[i]))**2
        
        # General boundary constraints (always active)
        for i in range(n):
            pen += np.maximum(0, r_p[i] - x_p[i])**2
            pen += np.maximum(0, x_p[i] - (1 - r_p[i]))**2
            pen += np.maximum(0, r_p[i] - y_p[i])**2
            pen += np.maximum(0, y_p[i] - (1 - r_p[i]))**2
        
        # Overlap penalties for non-contact edges (should be zero)
        i_idx, j_idx = np.triu_indices(n, k=1)
        for k in range(len(i_idx)):
            i, j = i_idx[k], j_idx[k]
            if not adj[i, j]:
                dx = x_p[i] - x_p[j]
                dy = y_p[i] - y_p[j]
                dist = np.sqrt(dx*dx + dy*dy)
                violation = np.maximum(0, r_p[i] + r_p[j] - dist)
                pen += 100 * violation**2
        
        return -np.sum(r_p) + 1000 * pen
    
    def socp_constraints(params):
        """
        Inequality constraints: g(params) >= 0
        """
        x_p = params[:n]
        y_p = params[n:2*n]
        r_p = params[2*n:3*n]
        
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            constraints.append(x_p[i] - r_p[i])  # x >= r
            constraints.append((1 - x_p[i]) - r_p[i])  # 1-x >= r
            constraints.append(y_p[i] - r_p[i])  # y >= r
            constraints.append((1 - y_p[i]) - r_p[i])  # 1-y >= r
        
        # Contact constraints (for edges in contact graph)
        for i, j in edges:
            dx = x_p[i] - x_p[j]
            dy = y_p[i] - y_p[j]
            dist = np.sqrt(dx*dx + dy*dy)
            constraints.append(dist - (r_p[i] + r_p[j]))
        
        # Non-contact separation constraints
        i_idx, j_idx = np.triu_indices(n, k=1)
        for k in range(len(i_idx)):
            i, j = i_idx[k], j_idx[k]
            if not adj[i, j]:
                dx = x_p[i] - x_p[j]
                dy = y_p[i] - y_p[j]
                dist = np.sqrt(dx*dx + dy*dy)
                constraints.append(dist - (r_p[i] + r_p[j]))
        
        return constraints
    
    # Initial params
    params0 = np.concatenate([x, y, r])
    
    # Bounds
    bounds = []
    for _ in range(n):
        bounds.append((0, 1))  # x
    for _ in range(n):
        bounds.append((0, 1))  # y
    for _ in range(n):
        bounds.append((1e-6, 0.5))  # r
    
    # Try SLSQP
    try:
        res = minimize(socp_objective, params0, method='SLSQP', 
                      bounds=bounds, options={'maxiter': 300, 'ftol': 1e-14})
        
        x_opt = res.x[:n]
        y_opt = res.x[n:2*n]
        r_opt = res.x[2*n:3*n]
        
        # Validate and repair
        x_opt, y_opt, r_opt = _make_valid(x_opt, y_opt, r_opt)
        x_opt, y_opt, r_opt = _maximize_radii_vectorized(x_opt, y_opt, r_opt)
        
        return x_opt, y_opt, r_opt, np.sum(r_opt)
        
    except Exception:
        return x, y, r, np.sum(r)

def _hexagonal_lattice(n, margin=0.05):
    """
    Generate n points on a hexagonal lattice inside [margin, 1-margin]^2.
    Returns array of shape (n, 2).
    """
    points = []
    # For a hex lattice, we want circles of radius r to be tangent.
    # Distance between centers in hex lattice = 2r (for tangent circles)
    # Horizontal spacing = 2r, vertical spacing = 2r * sqrt(3)/2 = r*sqrt(3)
    # Area per circle in hex packing = 2r^2 * sqrt(3)
    # Total area = (1-2*margin)^2
    # So 2r^2 * sqrt(3) * n <= (1-2*margin)^2
    # r <= sqrt((1-2*margin)^2 / (2 * sqrt(3) * n))
    
    side = 1.0 - 2.0 * margin
    s = np.sqrt(2.0 * side * side / (n * np.sqrt(3)))
    
    row_spacing = s * np.sqrt(3) / 2
    col_spacing = s
    
    # Determine number of rows and columns
    n_rows = int(np.floor(side / row_spacing)) + 1
    n_cols = int(np.floor(side / col_spacing)) + 1
    
    # Generate points
    for i in range(n_rows):
        y = margin + i * row_spacing
        offset = (i % 2) * (col_spacing / 2.0)
        for j in range(n_cols):
            x = margin + j * col_spacing + offset
            if margin <= x <= 1.0 - margin and margin <= y <= 1.0 - margin:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than n points, randomly sample n
    if len(points) > n:
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
    elif len(points) < n:
        # Pad with random points
        while len(points) < n:
            points = np.vstack([points, np.array([np.random.uniform(margin, 1-margin), 
                                                   np.random.uniform(margin, 1-margin)])])
    
    return points[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_result = None
    best_score = -np.inf
    
    # Precompute indices for vectorized overlap calculation
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def objective(params):
        """
        Objective function for L-BFGS-B.
        We want to maximize sum(r), so we minimize -sum(r).
        We add penalties for boundary violations and circle overlaps.
        Vectorized for speed.
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Ensure radii are positive
        r_pos = np.maximum(r, 1e-6)
        
        # 1. Boundary Penalties
        pen_x_left = np.maximum(0, r_pos - x)**2
        pen_x_right = np.maximum(0, x - (1 - r_pos))**2
        pen_y_bottom = np.maximum(0, r_pos - y)**2
        pen_y_top = np.maximum(0, y - (1 - r_pos))**2
        boundary_pen = np.sum(pen_x_left + pen_x_right + pen_y_bottom + pen_y_top)
        
        # 2. Overlap Penalties (Vectorized)
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        d = np.sqrt(dx*dx + dy*dy)
        overlaps = np.maximum(0, r_pos[i_idx] + r_pos[j_idx] - d)
        overlap_pen = np.sum(overlaps**2)
        
        # Weights: High weight to ensure validity is prioritized
        w_pen = 100.0
        
        return -np.sum(r_pos) + w_pen * (boundary_pen + overlap_pen)
    
    def optimize_start(x0, y0, r0):
        """
        Run L-BFGS-B optimization from a given start.
        Returns the best valid configuration found.
        """
        params0 = np.concatenate([x0, y0, r0])
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
        for i in range(N):
            bounds.append((0, 1))  # y
        for i in range(N):
            bounds.append((0, 0.5))  # r
            
        try:
            res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            # Try to repair by making valid
            x_opt, y_opt, r_opt = _make_valid(x_opt, y_opt, r_opt)
            
            # Try to maximize radii further
            x_opt, y_opt, r_opt = _maximize_radii_vectorized(x_opt, y_opt, r_opt)
            
            score = np.sum(r_opt)
            
            return score, np.column_stack((x_opt, y_opt, r_opt))
            
        except Exception:
            return -np.inf, None

    def polish_with_socp(x, y, r):
        """
        Polish using SOCP with contact graph identification.
        """
        # Identify contact graph
        adj, boundary = _identify_contact_graph(x, y, r, threshold_factor=1.02)
        
        # Solve SOCP
        x_opt, y_opt, r_opt, score = _solve_socp_for_contact_graph(x, y, r, adj, boundary)
        
        # Final validation and maximization
        x_opt, y_opt, r_opt = _make_valid(x_opt, y_opt, r_opt)
        x_opt, y_opt, r_opt = _maximize_radii_vectorized(x_opt, y_opt, r_opt)
        
        return np.sum(r_opt), np.column_stack((x_opt, y_opt, r_opt))
    
    def polish(x, y, r):
        """
        High-precision polish pass.
        Runs L-BFGS-B with tighter tolerances and more iterations.
        """
        params0 = np.concatenate([x, y, r])
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
        for i in range(N):
            bounds.append((0, 1))  # y
        for i in range(N):
            bounds.append((0, 0.5))  # r
            
        try:
            res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            # Repair and maximize
            x_opt, y_opt, r_opt = _make_valid(x_opt, y_opt, r_opt)
            x_opt, y_opt, r_opt = _maximize_radii_vectorized(x_opt, y_opt, r_opt)
            
            score = np.sum(r_opt)
            return score, np.column_stack((x_opt, y_opt, r_opt))
            
        except Exception:
            return np.sum(r), np.column_stack((x, y, r))
    
    def local_search(x, y, r, num_iter=20):
        """
        Simple local search: perturb positions/radii, re-optimize.
        """
        best_score = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        for _ in range(num_iter):
            if time.time() > deadline - 5:
                break
            
            # Perturb
            perturb_scale = 0.05
            x_new = x + np.random.normal(0, perturb_scale, N)
            y_new = y + np.random.normal(0, perturb_scale, N)
            r_new = r + np.random.normal(0, 0.01, N)
            
            # Keep within bounds
            x_new = np.clip(x_new, 0, 1)
            y_new = np.clip(y_new, 0, 1)
            r_new = np.clip(r_new, 0, 0.5)
            
            # Optimize from perturbed state
            score, config = optimize_start(x_new, y_new, r_new)
            
            if config is not None and score > best_score:
                best_score = score
                best_x = config[:, 0]
                best_y = config[:, 1]
                best_r = config[:, 2]
        
        return best_score, np.column_stack((best_x, best_y, best_r))
    
    # Try multiple random restarts with different initializations
    n_restarts = 16
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with hexagonal lattice + random perturbation
        x0 = _hexagonal_lattice(N, margin=0.05)
        # Add small random perturbation
        x0 += rng.uniform(-0.02, 0.02, size=x0.shape)
        
        # Initial radii: small
        r0 = np.full(N, 0.03)
        
        # Optimize
        score, config = optimize_start(x0[:, 0], x0[:, 1], r0)
        
        if config is not None and score > best_score:
            best_score = score
            best_result = config
        
        # If we have time, do local search and polish
        if time.time() < deadline - 20.0 and best_result is not None:
            x_cur = best_result[:, 0]
            y_cur = best_result[:, 1]
            r_cur = best_result[:, 2]
            
            # Local search
            ls_score, ls_config = local_search(x_cur, y_cur, r_cur, num_iter=15)
            
            if ls_score > best_score:
                best_score = ls_score
                best_result = ls_config
            
            # Polish with SOCP
            if time.time() < deadline - 15.0:
                socp_score, socp_config = polish_with_socp(best_result[:, 0], best_result[:, 1], best_result[:, 2])
                
                if socp_score > best_score:
                    best_score = socp_score
                    best_result = socp_config
            
            # Also try regular polish
            if time.time() < deadline - 12.0:
                pol_score, pol_config = polish(best_result[:, 0], best_result[:, 1], best_result[:, 2])
                
                if pol_score > best_score:
                    best_score = pol_score
                    best_result = pol_config
    
    # If no valid result found, return a trivial valid packing
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
