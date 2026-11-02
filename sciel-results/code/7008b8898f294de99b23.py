"""Circle packing in unit square using Delaunay-based constrained optimization."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

def _get_delaunay_edges(points):
    """
    Get Delaunay edges for the given points.
    Returns a list of (i, j) tuples.
    """
    if len(points) < 3:
        return []
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a = simplex[i]
                b = simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback: all pairs if Delaunay fails (e.g., collinear points)
        return [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]

def _vectorized_constraints(x, N, edges):
    """
    Calculate all inequality constraints.
    Returns an array where each element must be >= 0.
    Constraints:
    1. Boundary: x_i - r_i >= 0, y_i - r_i >= 0, 1 - x_i - r_i >= 0, 1 - y_i - r_i >= 0
    2. Non-overlap: dist(i,j) - r_i - r_j >= 0 for all edges (i,j)
    """
    pts = x.reshape(N, 3)
    radii = pts[:, 2]
    centers = pts[:, :2]
    
    # Boundary constraints (4 per circle)
    # Left: x_i - r_i
    c_left = centers[:, 0] - radii
    # Bottom: y_i - r_i
    c_bottom = centers[:, 1] - radii
    # Right: 1 - x_i - r_i
    c_right = 1.0 - centers[:, 0] - radii
    # Top: 1 - y_i - r_i
    c_top = 1.0 - centers[:, 1] - radii
    
    cons = np.concatenate([c_left, c_bottom, c_right, c_top])
    
    # Non-overlap constraints for edges
    if edges:
        edge_i = np.array([e[0] for e in edges])
        edge_j = np.array([e[1] for e in edges])
        
        ci = centers[edge_i]
        cj = centers[edge_j]
        ri = radii[edge_i]
        rj = radii[edge_j]
        
        dists = np.linalg.norm(ci - cj, axis=1)
        c_nonoverlap = dists - ri - rj
        
        cons = np.concatenate([cons, c_nonoverlap])
        
    return cons

def _objective(x, N):
    """Minimize negative sum of radii."""
    pts = x.reshape(N, 3)
    return -np.sum(pts[:, 2])

def _is_valid_solution(x, N, tol=1e-6):
    """Check if the solution is valid (all constraints satisfied)."""
    pts = x.reshape(N, 3)
    radii = pts[:, 2]
    centers = pts[:, :2]
    
    # Check boundaries
    if np.any(centers[:, 0] - radii < -tol): return False
    if np.any(centers[:, 1] - radii < -tol): return False
    if np.any(1.0 - centers[:, 0] - radii < -tol): return False
    if np.any(1.0 - centers[:, 1] - radii < -tol): return False
    
    # Check all pairwise distances (not just Delaunay, to be safe)
    # This is O(N^2) but N=26 is small.
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist + tol:
                return False
    return True

def _make_initial_guess(N, strategy='grid'):
    """
    Create an initial guess.
    """
    pts = np.zeros((N, 3))
    
    if strategy == 'grid':
        side = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.8
            pts[i] = [x, y, r]
            
    elif strategy == 'hex':
        # Approximate hexagonal packing
        # Rows of circles
        r = 0.1
        dy = r * np.sqrt(3)
        dx = 2 * r
        row = 0
        col = 0
        for i in range(N):
            y = r + row * dy
            x_offset = (row % 2) * r
            x = r + x_offset + col * dx
            
            # Wrap around if x > 1 - r
            if x > 1 - r:
                col = 0
                row += 1
                y = r + row * dy
                x_offset = (row % 2) * r
                x = r + x_offset + col * dx
            
            pts[i] = [x, y, r]
            col += 1
            
    elif strategy == 'random':
        # Random points with small radii
        pts[:, 0] = np.random.uniform(0.1, 0.9, N)
        pts[:, 1] = np.random.uniform(0.1, 0.9, N)
        pts[:, 2] = 0.05
        
    return pts

def construct():
    start_time = time.time()
    time_limit = 110  # leave margin for final checks
    
    best_x = None
    best_obj = -np.inf  # We maximize sum of radii, so minimize -sum
    
    # Strategies to try
    strategies = ['grid', 'hex', 'random', 'random', 'grid']
    
    # We will iterate through strategies and potentially multiple random restarts
    # within the time limit.
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        strategy = strategies[(attempts - 1) % len(strategies)]
        if strategy == 'random':
             # Slightly vary random seed or just rely on np.random
             pass
             
        pts0 = _make_initial_guess(N, strategy)
        x0 = pts0.flatten()
        
        # Get initial edges
        edges = _get_delaunay_edges(pts0[:, :2])
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((1e-6, 0.5))
            
        # Define constraint function that updates edges dynamically?
        # SLSQP doesn't easily support dynamic constraint structures mid-optimization
        # unless we restart. We will run SLSQP with current edges, then check validity.
        # If invalid, we might need to add more constraints or restart.
        
        def obj_func(x):
            return _objective(x, N)
            
        def cons_func(x):
            return _vectorized_constraints(x, N, edges)
            
        constraints = {'type': 'ineq', 'fun': cons_func}
        
        try:
            result = minimize(
                obj_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
            )
            
            if result.success:
                x_sol = result.x
                # Check if valid against ALL pairs
                if _is_valid_solution(x_sol, N, tol=1e-5):
                    obj_val = -result.fun
                    if obj_val > best_obj:
                        best_obj = obj_val
                        best_x = x_sol.copy()
                else:
                    # Try to repair by shrinking radii slightly and re-optimizing?
                    # Or just discard.
                    pass
            else:
                # Sometimes SLSQP fails but provides a good point
                x_sol = result.x
                if _is_valid_solution(x_sol, N, tol=1e-5):
                    obj_val = -result.fun
                    if obj_val > best_obj:
                        best_obj = obj_val
                        best_x = x_sol.copy()
                        
        except Exception:
            pass
            
        # If we found a good solution, try to refine it with updated Delaunay edges
        if best_x is not None:
            # Refinement loop
            for ref_iter in range(5):
                if time.time() - start_time > time_limit:
                    break
                centers_cur = best_x.reshape(N, 3)[:, :2]
                edges = _get_delaunay_edges(centers_cur)
                
                def obj_func_ref(x):
                    return _objective(x, N)
                    
                def cons_func_ref(x):
                    return _vectorized_constraints(x, N, edges)
                    
                constraints_ref = {'type': 'ineq', 'fun': cons_func_ref}
                
                try:
                    result = minimize(
                        obj_func_ref,
                        best_x,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints_ref,
                        options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
                    )
                    
                    if result.success or (not result.success and result.fun < best_obj):
                        x_sol = result.x
                        if _is_valid_solution(x_sol, N, tol=1e-5):
                            obj_val = -result.fun
                            if obj_val > best_obj:
                                best_obj = obj_val
                                best_x = x_sol.copy()
                except Exception:
                    break
                    
            # If we've done a few refinements, break out to try another strategy
            if attempts % 2 == 0:
                continue

    if best_x is None:
        # Fallback: return a valid grid packing
        pts = _make_initial_guess(N, 'grid')
        # Ensure valid
        for i in range(N):
            pts[i, 2] = min(pts[i, 2], pts[i, 0], pts[i, 1], 1-pts[i, 0], 1-pts[i, 1])
        return pts

    # Final safety check and cleanup
    pts_final = best_x.reshape(N, 3)
    # Clamp to ensure no negative radii or out of bounds
    for i in range(N):
        pts_final[i, 2] = max(pts_final[i, 2], 1e-6)
        pts_final[i, 0] = np.clip(pts_final[i, 0], 0, 1)
        pts_final[i, 1] = np.clip(pts_final[i, 1], 0, 1)
        
    # Final validity check, if invalid, shrink radii
    if not _is_valid_solution(pts_final.flatten(), N, tol=1e-6):
         # Simple shrink
         for i in range(N):
             pts_final[i, 2] *= 0.9
         # Re-check
         if not _is_valid_solution(pts_final.flatten(), N, tol=1e-6):
             pts_final = _make_initial_guess(N, 'grid')
             for i in range(N):
                 pts_final[i, 2] = min(pts_final[i, 2], pts_final[i, 0], pts_final[i, 1], 1-pts_final[i, 0], 1-pts_final[i, 1])

    return pts_final
# EVOLVE-BLOCK-END
