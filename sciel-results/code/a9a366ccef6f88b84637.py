# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

TOL = 1e-7
DEADLINE_MARGIN = 10.0  # seconds before 630s limit

def _is_valid(centers, sides, n=10):
    """Check if a packing is valid."""
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Check inside [-tol, 1+tol]^2
    for i in range(n):
        x0 = centers[i, 0] - sides[i] / 2
        x1 = centers[i, 0] + sides[i] / 2
        y0 = centers[i, 1] - sides[i] / 2
        y1 = centers[i, 1] + sides[i] / 2
        if x0 < -TOL - 1e-10 or x1 > 1 + TOL + 1e-10:
            return False
        if y0 < -TOL - 1e-10 or y1 > 1 + TOL + 1e-10:
            return False
    # Check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _solve_lp_edge_based(n, sep_x, sep_y):
    """
    Solve LP using edge coordinates.
    Variables: s[0..n-1], x[0..n-1], y[0..n-1]
    sep_x[i,j] = 1 if i is left of j, -1 if i is right of j, 0 if not separated in x.
    sep_y[i,j] = 1 if i is below j, -1 if i is above j, 0 if not separated in y.
    """
    # Total vars: 3n
    # s_i at i, x_i at n+i, y_i at 2n+i
    
    c = np.zeros(3 * n)
    for i in range(n):
        c[i] = -1.0  # Maximize sum(s_i) -> Minimize -sum(s_i)
        
    A_ub = []
    b_ub = []
    
    # Boundary constraints:
    # x_i >= 0 (handled by bounds)
    # x_i + s_i <= 1  =>  s_i + x_i <= 1
    # y_i >= 0 (handled by bounds)
    # y_i + s_i <= 1  =>  s_i + y_i <= 1
    
    for i in range(n):
        # s_i + x_i <= 1
        row = np.zeros(3 * n)
        row[i] = 1.0
        row[n + i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
        
        # s_i + y_i <= 1
        row = np.zeros(3 * n)
        row[i] = 1.0
        row[2 * n + i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
        
    # Separation constraints:
    # If sep_x[i,j] == 1: x_i + s_i <= x_j  =>  s_i + x_i - x_j <= 0
    # If sep_x[i,j] == -1: x_j + s_j <= x_i  =>  s_j + x_j - x_i <= 0
    # Similarly for y.
    
    for i in range(n):
        for j in range(i + 1, n):
            if sep_x[i, j] == 1:
                row = np.zeros(3 * n)
                row[i] = 1.0
                row[n + i] = 1.0
                row[n + j] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)
            elif sep_x[i, j] == -1:
                row = np.zeros(3 * n)
                row[j] = 1.0
                row[n + j] = 1.0
                row[n + i] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)
            
            if sep_y[i, j] == 1:
                row = np.zeros(3 * n)
                row[i] = 1.0
                row[2 * n + i] = 1.0
                row[2 * n + j] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)
            elif sep_y[i, j] == -1:
                row = np.zeros(3 * n)
                row[j] = 1.0
                row[2 * n + j] = 1.0
                row[2 * n + i] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)
                
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = None
        b_ub = None
        
    # Bounds: s_i >= 0, x_i >= 0, y_i >= 0, s_i <= 1, x_i <= 1, y_i <= 1
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # s_i
        bounds.append((0, 1))  # x_i
        bounds.append((0, 1))  # y_i
        
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            s = result.x[:n]
            x = result.x[n:2*n]
            y = result.x[2*n:3*n]
            centers = np.stack([x + s/2, y + s/2], axis=1)
            return centers, s, -result.fun
    except Exception:
        pass
    return None

def _local_polish(centers, sides, n, max_iter=50, deadline=None):
    """
    Local improvement: try to increase each side slightly while maintaining validity.
    """
    best_val = np.sum(sides)
    centers = centers.copy()
    sides = sides.copy()
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            # Try to increase side i
            for delta in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                new_sides = sides.copy()
                new_sides[i] += delta
                if new_sides[i] <= 0 or new_sides[i] > 1:
                    continue
                # Check if still valid
                if _is_valid(centers, new_sides, n):
                    sides = new_sides
                    best_val = np.sum(sides)
                    improved = True
                    break
        if not improved:
            break
            
    return centers, sides, best_val

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 630 - DEADLINE_MARGIN
    
    # Baseline grid packing
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
    best_sides = np.full(n, side, dtype=np.float64)
    best_val = np.sum(best_sides)
    
    # Generate separation patterns
    # We will try a set of structured patterns and random patterns.
    # Structured patterns help find good solutions quickly.
    
    patterns = []
    
    # 1. Grid patterns (rows x cols)
    for rows in range(1, n + 1):
        for cols in range(1, n + 1):
            if rows * cols < n:
                continue
            sep_x = np.zeros((n, n), dtype=int)
            sep_y = np.zeros((n, n), dtype=int)
            for i in range(n):
                ri, ci = divmod(i, cols)
            for i in range(n):
                for j in range(i + 1, n):
                    ri_i, ci_i = divmod(i, cols)
                    ri_j, ci_j = divmod(j, cols)
                    if ri_i == ri_j:
                        # Same row, separate in x
                        if ci_i < ci_j:
                            sep_x[i, j] = 1
                            sep_x[j, i] = -1
                        else:
                            sep_x[i, j] = -1
                            sep_x[j, i] = 1
                    else:
                        # Different row, separate in y
                        if ri_i < ri_j:
                            sep_y[i, j] = 1
                            sep_y[j, i] = -1
                        else:
                            sep_y[i, j] = -1
                            sep_y[j, i] = 1
            patterns.append((sep_x, sep_y))
            
    # 2. Random patterns
    rng = np.random.RandomState(42)
    num_random = 200
    for _ in range(num_random):
        sep_x = np.zeros((n, n), dtype=int)
        sep_y = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    # Separate in x
                    if rng.random() < 0.5:
                        sep_x[i, j] = 1
                        sep_x[j, i] = -1
                    else:
                        sep_x[i, j] = -1
                        sep_x[j, i] = 1
                else:
                    # Separate in y
                    if rng.random() < 0.5:
                        sep_y[i, j] = 1
                        sep_y[j, i] = -1
                    else:
                        sep_y[i, j] = -1
                        sep_y[j, i] = 1
        patterns.append((sep_x, sep_y))
        
    # Solve LP for each pattern
    for sep_x, sep_y in patterns:
        if time.time() > deadline:
            break
            
        result = _solve_lp_edge_based(n, sep_x, sep_y)
        if result is not None:
            centers, s, val = result
            if _is_valid(centers, s, n):
                if val > best_val:
                    best_val = val
                    best_centers = centers
                    best_sides = s
    
    # Local polish on the best solution found
    if time.time() < deadline:
        centers_polished, sides_polished, val_polished = _local_polish(
            best_centers, best_sides, n, max_iter=50, deadline=deadline
        )
        if val_polished > best_val:
            best_val = val_polished
            best_centers = centers_polished
            best_sides = sides_polished
            
    return best_centers, best_sides

# EVOLVE-BLOCK-END
