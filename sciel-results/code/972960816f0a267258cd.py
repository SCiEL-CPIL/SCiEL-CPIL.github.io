"""Circle packing in unit square using Delaunay-based constrained optimization."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

def _delaunay_constraints(points):
    """
    Given an array of points (n, 2), return a list of edge indices (i, j)
    from the Delaunay triangulation. These are the pairs that need to be
    checked for non-overlap (plus boundary).
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
        # Fallback: all pairs
        return [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]


def _validate_packing(pts, tol=1e-5):
    """Check if a packing is valid: all circles inside [0,1]^2 and non-overlapping."""
    N = len(pts)
    # Check boundaries
    for i in range(N):
        xi, yi, ri = pts[i]
        if ri - xi > tol or ri - yi > tol:
            return False
        if xi + ri - 1 > tol or yi + ri - 1 > tol:
            return False
    # Check all pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ri = pts[i]
            xj, yj, rj = pts[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if ri + rj - dist > tol:
                return False
    return True


def _make_initial_guess(N):
    """
    Create a good initial guess using a hexagonal-like pattern.
    """
    # Try different grid configurations and pick the best
    best_pts = None
    best_score = -np.inf
    
    # Configuration 1: 5x6 grid (last row has 1)
    configs = [
        (5, 6),  # 5 cols, 6 rows
        (6, 5),  # 6 cols, 5 rows
        (4, 7),  # 4 cols, 7 rows
        (7, 4),  # 7 cols, 4 rows
    ]
    
    for cols, rows in configs:
        if cols * rows < N:
            continue
        pts = np.zeros((N, 3))
        for i in range(N):
            row = i // cols
            col = i % cols
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            r = min(0.5 / cols, 0.5 / rows) * 0.85
            pts[i] = [x, y, r]
        
        if _validate_packing(pts):
            score = np.sum(pts[:, 2])
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        pts = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.9
            pts[i] = [x, y, r]
        best_pts = pts
    
    return best_pts


def _solve_with_edges(x0, edges, bounds, maxiter=500, ftol=1e-10):
    """
    Solve the constrained optimization problem with given Delaunay edges.
    """
    def obj_func(x):
        return -np.sum(x[2::3])  # negative sum of radii
    
    constraints = []
    
    # Boundary constraints: for each circle i
    # Left: x_i - r_i >= 0  =>  x_i - r_i >= 0
    # Bottom: y_i - r_i >= 0
    # Right: 1 - x_i - r_i >= 0
    # Top: 1 - y_i - r_i >= 0
    for i in range(N):
        xi_idx = 3 * i
        yi_idx = 3 * i + 1
        ri_idx = 3 * i + 2
        # Left boundary: x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, xi=xi_idx, ri=ri_idx: x[xi] - x[ri]})
        # Bottom boundary: y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, yi=yi_idx, ri=ri_idx: x[yi] - x[ri]})
        # Right boundary: 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, xi=xi_idx, ri=ri_idx: 1 - x[xi] - x[ri]})
        # Top boundary: 1 - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, yi=yi_idx, ri=ri_idx: 1 - x[yi] - x[ri]})
    
    # Non-overlap constraints for Delaunay edges
    for (i, j) in edges:
        def _nonoverlap(x, i=i, j=j):
            xi, yi = x[3*i], x[3*i+1]
            xj, yj = x[3*j], x[3*j+1]
            ri = x[3*i+2]
            rj = x[3*j+2]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            # Constraint: dist - ri - rj >= 0
            return dist - ri - rj
        constraints.append({'type': 'ineq', 'fun': _nonoverlap})
    
    try:
        result = minimize(
            obj_func,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
        )
        return result
    except Exception as e:
        return None


def construct():
    start_time = time.time()
    time_limit = 110  # leave some margin
    
    # Initial guess
    pts = _make_initial_guess(N)
    x0 = pts.flatten()
    
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-6, 0.5))  # r
    
    best_x = x0.copy()
    best_score = -np.sum(x0[2::3])
    
    # Iterative improvement: solve, update edges, repeat
    max_outer_iters = 15
    
    for outer_iter in range(max_outer_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Re-compute edges based on current best solution
        centers_cur = best_x.reshape(N, 3)[:, :2]
        edges = _delaunay_constraints(centers_cur)
        
        # Run optimization
        result = _solve_with_edges(best_x, edges, bounds, maxiter=400, ftol=1e-10)
        
        if result is not None and result.fun < best_score:
            x_sol = result.x
            pts_sol = x_sol.reshape(N, 3)
            
            if _validate_packing(pts_sol, tol=1e-5):
                score = np.sum(pts_sol[:, 2])
                if score > best_score:
                    best_score = score
                    best_x = x_sol.copy()
                else:
                    # Try to improve with a smaller tolerance
                    pass
            else:
                # Solution is invalid, try to repair
                # Project back to valid region
                pass
        
        # Also try from the original initial point with current edges
        if outer_iter == 0:
            result2 = _solve_with_edges(x0, edges, bounds, maxiter=400, ftol=1e-10)
            if result2 is not None:
                x_sol2 = result2.x
                pts_sol2 = x_sol2.reshape(N, 3)
                if _validate_packing(pts_sol2, tol=1e-5):
                    score2 = np.sum(pts_sol2[:, 2])
                    if score2 > best_score:
                        best_score = score2
                        best_x = x_sol2.copy()
    
    # Final solution
    pts_final = best_x.reshape(N, 3).copy()
    
    # Ensure validity by clamping radii if needed
    for i in range(N):
        xi, yi, ri = pts_final[i]
        # Clamp radius to fit within boundaries
        ri = min(ri, xi, yi, 1 - xi, 1 - yi)
        ri = max(ri, 1e-6)
        pts_final[i, 2] = ri
    
    # Final pairwise check and shrink if overlapping
    for iteration in range(10):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ri = pts_final[i]
                xj, yj, rj = pts_final[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if ri + rj > dist - 1e-8:
                    # Shrink both radii slightly
                    overlap = ri + rj - dist
                    shrink = overlap / 2 + 1e-6
                    ri = max(ri - shrink, 1e-6)
                    rj = max(rj - shrink, 1e-6)
                    pts_final[i, 2] = ri
                    pts_final[j, 2] = rj
                    changed = True
        if not changed:
            break
    
    # Final boundary check
    for i in range(N):
        xi, yi, ri = pts_final[i]
        ri = min(ri, xi - 1e-6, yi - 1e-6, 1 - xi - 1e-6, 1 - yi - 1e-6)
        ri = max(ri, 1e-6)
        pts_final[i, 2] = ri
    
    return pts_final
# EVOLVE-BLOCK-END
