"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate(arr):
    """Check that circles are inside [0,1]^2 and non-overlapping."""
    if arr is None or len(arr) != N:
        return False
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    if np.any(r < 0) or np.any(x - r < -1e-9) or np.any(y - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    # Check pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True

def _initial_hex(n):
    """Generate initial positions in a hexagonal packing pattern scaled to unit square."""
    pts = []
    rows = 0
    # For n=26, use a hexagonal arrangement
    # 5 rows: 5, 5, 5, 5, 6 = 26 or 6,5,5,5,5=26
    # Let's use 5 rows of 5 and one row of 6, or better:
    # Hex packing: rows alternate offset
    # For 26 circles, use 5 full rows of 5 (25) + 1, or 6 rows
    # Let's do 5 rows: 5,5,5,5,6
    row_counts = [5, 5, 5, 5, 6]
    total = sum(row_counts)
    if total != n:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(n)))
        for i in range(side):
            for j in range(side):
                if len(pts) < n:
                    pts.append(((i + 0.5) / side, (j + 0.5) / side))
        return np.array(pts[:n])
    
    # Calculate hex spacing
    # In hex packing, horizontal spacing = d, vertical spacing = d * sqrt(3)/2
    # For 6 columns max, width = 5*d (for 6 cols, centers from d/2 to 5.5d)
    # Actually for 6 circles in a row: centers at 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 times d
    # Width = 5.5d - 0.5d = 5d, so d = 1/5 = 0.2 for 6 cols
    # For 5 cols: width = 4d, d = 0.25
    # Let's use d = 0.2 for consistency
    d = 0.18  # Slightly smaller to fit
    vert_spacing = d * np.sqrt(3) / 2
    
    row_idx = 0
    for row in row_counts:
        offset = 0 if row_idx % 2 == 0 else d / 2
        # Center the row
        total_width = (row - 1) * d
        start_x = 0.5 - total_width / 2 + offset
        for col in range(row):
            x = start_x + col * d
            y = 0.5 + (row_idx - 2) * vert_spacing  # Center vertically around 0.5
            pts.append((x, y))
        row_idx += 1
    
    return np.array(pts[:n])

def _delaunay_edges(points):
    """Get Delaunay edges from points."""
    if len(points) < 3:
        return list(zip(range(len(points)), range(1, len(points))))
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback to all pairs
        return [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]

def _optimize_from_start(start_xy, start_r, deadline):
    """Run SLSQP optimization from a given starting point."""
    n = N
    x0 = np.concatenate([start_xy.flatten(), start_r])
    
    def objective(z):
        xy = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        return -np.sum(r)
    
    # Constraints:
    # 1. x - r >= 0, y - r >= 0, x + r <= 1, y + r <= 1
    # 2. For Delaunay edges: dist(i,j) >= r_i + r_j
    
    def constraints_factory(xy_init, r_init):
        edges = _delaunay_edges(xy_init)
        cons = []
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2 * i] - z[2 * i + 2 * n]})  # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2 * i + 1] - z[2 * i + 2 * n]})  # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[2 * i] - z[2 * i + 2 * n]})  # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[2 * i + 1] - z[2 * i + 2 * n]})  # 1 - y - r >= 0
        for (i, j) in edges:
            def edge_constraint(z, i=i, j=j):
                xi, yi = z[2 * i], z[2 * i + 1]
                xj, yj = z[2 * j], z[2 * j + 1]
                ri, rj = z[2 * i + 2 * n], z[2 * j + 2 * n]
                return np.sqrt((xi - xj)**2 + (yi - yj)**2) - ri - rj
            cons.append({'type': 'ineq', 'fun': edge_constraint})
        return cons
    
    cons = constraints_factory(start_xy, start_r)
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0.01, 0.5))  # r
    bounds = [(0, 1)] * (2 * n) + [(0.01, 0.5)] * n
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                          constraints=cons, options={'maxiter': 200, 'ftol': 1e-12})
        if result.success:
            z = result.x
            xy = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            arr = np.column_stack([xy, r])
            if _validate(arr):
                return arr, -result.fun
    except Exception:
        pass
    
    return None, None

def construct():
    t_start = time.time()
    deadline = t_start + 140  # Safety margin
    
    # Generate multiple initial configurations
    best_arr = None
    best_score = -np.inf
    
    # Initial hexagonal packing
    xy_init = _initial_hex(N)
    # Estimate initial radii
    r_init = np.full(N, 0.08)
    
    # Try optimization
    arr, score = _optimize_from_start(xy_init, r_init, deadline)
    if arr is not None and score > best_score:
        best_score = score
        best_arr = arr
    
    # If time allows, try a few more random restarts
    t_now = time.time()
    if t_now < deadline - 20:
        np.random.seed(42)
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            # Random jittered hex
            xy = xy_init + np.random.randn(N, 2) * 0.02
            xy = np.clip(xy, 0.1, 0.9)
            r = np.full(N, 0.08)
            arr, score = _optimize_from_start(xy, r, deadline)
            if arr is not None and score > best_score:
                best_score = score
                best_arr = arr
    
    # Fallback to seed if optimization failed
    if best_arr is None or not _validate(best_arr):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
    
    return best_arr

# EVOLVE-BLOCK-END
