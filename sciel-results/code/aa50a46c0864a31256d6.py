# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product, combinations
from scipy.optimize import linprog

TOL = 1e-7
DEADLINE_MARGIN = 15.0

def _is_valid(centers, sides, n=10):
    """Check if a packing is valid."""
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        x0 = centers[i, 0] - sides[i] / 2
        x1 = centers[i, 0] + sides[i] / 2
        y0 = centers[i, 1] - sides[i] / 2
        y1 = centers[i, 1] + sides[i] / 2
        if x0 < -TOL - 1e-10 or x1 > 1 + TOL + 1e-10:
            return False
        if y0 < -TOL - 1e-10 or y1 > 1 + TOL + 1e-10:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _solve_grid_lp(n, rows, cols, cell_assign):
    """
    Solve LP for a grid packing.
    rows: number of row bands, cols: number of column bands
    cell_assign: list of length n, each entry is (row_idx, col_idx) telling which cell square i is in.
    
    The grid divides [0,1] x [0,1] into row bands and column bands.
    Variables: for each row band r: y0_r, y1_r (bottom/top of band)
               for each col band c: x0_c, x1_c (left/right of band)
               for each square i: s_i (side length)
    
    Constraints:
    - Bands partition [0,1]: x0_0=0, x1_{cols-1}=1, x1_c = x0_{c+1}, y1_r = y0_{r+1}
    - Square i in cell (r,c): s_i <= x1_c - x0_c, s_i <= y1_r - y0_r
    - Square i centered in cell: cx_i = (x0_c + x1_c)/2, cy_i = (y0_r + y1_r)/2
    - s_i > 0
    
    Objective: maximize sum(s_i)
    """
    # Variables:
    # x0_c for c in 0..cols-1: cols vars
    # x1_c for c in 0..cols-1: cols vars  
    # y0_r for r in 0..rows-1: rows vars
    # y1_r for r in 0..rows-1: rows vars
    # s_i for i in 0..n-1: n vars
    # Total: 2*cols + 2*rows + n
    
    nx0 = cols
    nx1 = cols
    ny0 = rows
    ny1 = rows
    ns = n
    
    total_vars = nx0 + nx1 + ny0 + ny1 + ns
    
    def var_x0(c): return c
    def var_x1(c): return nx0 + c
    def var_y0(r): return nx0 + nx1 + r
    def var_y1(r): return nx0 + nx1 + ny0 + r
    def var_s(i): return nx0 + nx1 + ny0 + ny1 + i
    
    # Objective: maximize sum(s_i) -> minimize -sum(s_i)
    c_obj = np.zeros(total_vars)
    for i in range(n):
        c_obj[var_s(i)] = -1.0
    
    A_ub = []
    b_ub = []
    
    # Boundary: x0_0 = 0 -> x0_0 <= 0 and -x0_0 <= 0
    row = np.zeros(total_vars); row[var_x0(0)] = 1.0
    A_ub.append(row); b_ub.append(0.0)
    row = np.zeros(total_vars); row[var_x0(0)] = -1.0
    A_ub.append(row); b_ub.append(0.0)
    
    # x1_{cols-1} = 1 -> x1_{cols-1} <= 1 and -x1_{cols-1} <= -1
    row = np.zeros(total_vars); row[var_x1(cols-1)] = 1.0
    A_ub.append(row); b_ub.append(1.0)
    row = np.zeros(total_vars); row[var_x1(cols-1)] = -1.0
    A_ub.append(row); b_ub.append(-1.0)
    
    # y0_0 = 0
    row = np.zeros(total_vars); row[var_y0(0)] = 1.0
    A_ub.append(row); b_ub.append(0.0)
    row = np.zeros(total_vars); row[var_y0(0)] = -1.0
    A_ub.append(row); b_ub.append(0.0)
    
    # y1_{rows-1} = 1
    row = np.zeros(total_vars); row[var_y1(rows-1)] = 1.0
    A_ub.append(row); b_ub.append(1.0)
    row = np.zeros(total_vars); row[var_y1(rows-1)] = -1.0
    A_ub.append(row); b_ub.append(-1.0)
    
    # Column continuity: x1_c = x0_{c+1} for c in 0..cols-2
    for c in range(cols - 1):
        # x1_c - x0_{c+1} <= 0
        row = np.zeros(total_vars)
        row[var_x1(c)] = 1.0
        row[var_x0(c+1)] = -1.0
        A_ub.append(row); b_ub.append(0.0)
        # x0_{c+1} - x1_c <= 0
        row = np.zeros(total_vars)
        row[var_x0(c+1)] = 1.0
        row[var_x1(c)] = -1.0
        A_ub.append(row); b_ub.append(0.0)
    
    # Row continuity: y1_r = y0_{r+1} for r in 0..rows-2
    for r in range(rows - 1):
        # y1_r - y0_{r+1} <= 0
        row = np.zeros(total_vars)
        row[var_y1(r)] = 1.0
        row[var_y0(r+1)] = -1.0
        A_ub.append(row); b_ub.append(0.0)
        # y0_{r+1} - y1_r <= 0
        row = np.zeros(total_vars)
        row[var_y0(r+1)] = 1.0
        row[var_y1(r)] = -1.0
        A_ub.append(row); b_ub.append(0.0)
    
    # Square i in cell (r_i, c_i): s_i <= x1_{c_i} - x0_{c_i} and s_i <= y1_{r_i} - y0_{r_i}
    for i in range(n):
        r_i, c_i = cell_assign[i]
        # s_i - x1_{c_i} + x0_{c_i} <= 0
        row = np.zeros(total_vars)
        row[var_s(i)] = 1.0
        row[var_x1(c_i)] = -1.0
        row[var_x0(c_i)] = 1.0
        A_ub.append(row); b_ub.append(0.0)
        # s_i - y1_{r_i} + y0_{r_i} <= 0
        row = np.zeros(total_vars)
        row[var_s(i)] = 1.0
        row[var_y1(r_i)] = -1.0
        row[var_y0(r_i)] = 1.0
        A_ub.append(row); b_ub.append(0.0)
    
    # Bounds: x0, x1, y0, y1 in [0, 1], s_i > 0
    bounds = []
    for _ in range(nx0 + nx1 + ny0 + ny1):
        bounds.append((0.0, 1.0))
    for _ in range(ns):
        bounds.append((1e-12, None))
    
    if not A_ub:
        return None
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except:
        return None
    
    if not result.success:
        return None
    
    x = result.x
    x0 = x[:nx0]
    x1 = x[nx0:nx0+nx1]
    y0 = x[nx0+nx1:nx0+nx1+ny0]
    y1 = x[nx0+nx1+ny0:nx0+nx1+ny0+ny1]
    s = x[nx0+nx1+ny0+ny1:]
    
    centers = np.zeros((n, 2))
    for i in range(n):
        r_i, c_i = cell_assign[i]
        centers[i, 0] = (x0[c_i] + x1[c_i]) / 2.0
        centers[i, 1] = (y0[r_i] + y1[r_i]) / 2.0
    
    val = np.sum(s)
    return centers, s, val


def _enumerate_grid_configs(n, max_configs=50000):
    """
    Enumerate grid configurations for n squares.
    A grid config is defined by (rows, cols) and an assignment of each square to a cell.
    Multiple squares can be in the same cell (they'll be sized by the cell dimensions).
    We want to maximize sum of side lengths.
    
    Strategy: try various (rows, cols) combinations and different cell assignments.
    """
    results = []
    start_time = time.time()
    deadline = start_time + 630 - DEADLINE_MARGIN
    
    # Try various grid dimensions
    grid_dims = []
    for rows in range(1, n + 1):
        for cols in range(1, n + 1):
            if rows * cols >= n:  # enough cells
                grid_dims.append((rows, cols))
    
    # Sort by some heuristic: prefer more balanced grids
    grid_dims.sort(key=lambda rc: abs(rc[0] - rc[1]) + 1.0/rc[0] + 1.0/rc[1])
    
    # For each grid, try different cell assignments
    # Key insight: we want to assign squares to cells to maximize the sum
    # of min(cell_width, cell_height)
    # This is equivalent to: for each cell, the square gets side = min(w_cell, h_cell)
    # But the cell dimensions are also variables!
    
    # For a given (rows, cols), the LP will optimize the band widths/heights.
    # The cell assignment determines which squares share which band constraints.
    
    # Different assignment strategies:
    # 1. Fill cells row by row (deterministic)
    # 2. Try various permutations of which squares go where
    # 3. Group squares by size class
    
    configs_tried = 0
    
    for rows, cols in grid_dims:
        if configs_tried >= max_configs or time.time() > deadline:
            break
        
        # Strategy 1: Fill cells in a specific order
        # Assign squares to cells: cell (r,c) gets some squares
        # For simplicity, first try: fill cells row by row, left to right
        # Each cell gets at most ceil(n / (rows*cols)) squares
        
        # Generate cell indices
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        
        # Assignment 1: sequential fill
        assign1 = [cells[i % len(cells)] for i in range(n)]
        result = _solve_grid_lp(n, rows, cols, assign1)
        configs_tried += 1
        if result:
            centers, s, val = result
            if _is_valid(centers, s, n):
                results.append((val, centers, s))
        
        # Assignment 2: column-major fill
        cells_col = [(r, c) for c in range(cols) for r in range(rows)]
        assign2 = [cells_col[i % len(cells_col)] for i in range(n)]
        if assign2 != assign1:
            result = _solve_grid_lp(n, rows, cols, assign2)
            configs_tried += 1
            if result:
                centers, s, val = result
                if _is_valid(centers, s, n):
                    results.append((val, centers, s))
        
        # Assignment 3: try to put larger squares in larger cells
        # Sort cells by potential size and assign
        # For now, try a few random assignments
        np.random.seed(rows * 1000 + cols)
        for trial in range(20):
            if configs_tried >= max_configs or time.time() > deadline:
                break
            assign = [cells[np.random.randint(len(cells))] for _ in range(n)]
            # Check for duplicates that might cause issues (same cell is fine, just shared)
            result = _solve_grid_lp(n, rows, cols, assign)
            configs_tried += 1
            if result:
                centers, s, val = result
                if _is_valid(centers, s, n):
                    results.append((val, centers, s))
        
        # Special structured assignments
        if rows * cols == n:
            # Each cell gets exactly one square
            # Try all permutations? Too many. Try a few.
            pass
        elif rows * cols > n:
            # Some cells are empty. Try different patterns of which cells are used.
            # Pattern: use first n cells in row-major order (already done)
            # Pattern: use cells in a "compact" pattern
            pass
    
    return results


def _try_specific_packing(n):
    """
    Try known good packings for 10 squares.
    The optimal packing for 10 squares in a unit square has been studied.
    Known result: sum of sides ≈ 3.666... or similar.
    
    Let's try: 3 rows x 4 cols grid where the middle row is split differently.
    Actually, let's try: 2 rows of 5, or 3+4+3 arrangement, etc.
    """
    results = []
    
    # Packing 1: 3 rows, 4 cols, but with 10 squares
    # Row 0: 4 squares, Row 1: 3 squares, Row 2: 3 squares
    # Or: Row 0: 3, Row 1: 4, Row 2: 3
    # The LP will figure out the optimal band widths.
    
    # For 3x4 grid with 10 squares:
    # Option A: cells (0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)
    assign_a = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    result = _solve_grid_lp(n, 3, 4, assign_a)
    if result:
        centers, s, val = result
        if _is_valid(centers, s, n):
            results.append((val, centers, s))
    
    # Option B: cells (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2)
    assign_b = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2)]
    result = _solve_grid_lp(n, 3, 4, assign_b)
    if result:
        centers, s, val = result
        if _is_valid(centers, s, n):
            results.append((val, centers, s))
    
    # Option C: 4x3 grid
    assign_c = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),(3,0)]
    result = _solve_grid_lp(n, 4, 3, assign_c)
    if result:
        centers, s, val = result
        if _is_valid(centers, s, n):
            results.append((val, centers, s))
    
    # Option D: 2x5 grid
    assign_d = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(1,2),(1,3),(1,4)]
    result = _solve_grid_lp(n, 2, 5, assign_d)
    if result:
        centers, s, val = result
        if _is_valid(centers, s, n):
            results.append((val, centers, s))
    
    # Option E: 5x2 grid
    assign_e = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,0),(3,1),(4,0),(4,1)]
    result = _solve_grid_lp(n, 5, 2, assign_e)
    if result:
        centers, s, val = result
        if _is_valid(centers, s, n):
            results.append((val, centers, s))
    
    # Option F: 1x10 (degenerate)
    assign_f = [(0,i) for i in range(10)]
    result = _solve_grid_lp(n, 1, 10, assign_f)
    if result:
        centers, s, val = result
        if _is_valid(centers, s, n):
            results.append((val, centers, s))
    
    # Option G: 10x1 (degenerate)
    assign_g = [(i,0) for i in range(10)]
    result = _solve_grid_lp(n, 10, 1, assign_g)
    if result:
        centers, s, val = result
        if _is_valid(centers, s, n):
            results.append((val, centers, s))
    
    return results


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
    
    # Try specific known packings
    specific_results = _try_specific_packing(n)
    if specific_results:
        specific_results.sort(key=lambda x: -x[0])
        for val, centers, sides in specific_results:
            if val > best_val:
                best_val = val
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Check deadline
    if time.time() > deadline:
        return best_centers, best_sides
    
    # Enumerate grid configurations
    grid_results = _enumerate_grid_configs(n, max_configs=30000)
    if grid_results:
        grid_results.sort(key=lambda x: -x[0])
        for val, centers, sides in grid_results:
            if val > best_val:
                best_val = val
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Check deadline
    if time.time() > deadline:
        return best_centers, best_sides
    
    # If we have time, try more refined grid configs
    remaining = deadline - time.time()
    if remaining > 20:
        # Try more random assignments for the best-performing grid dims
        # Identify which grid dims gave best results
        # For now, just do more random sampling
        np.random.seed(12345)
        for rows in range(2, 6):
            for cols in range(2, 6):
                if rows * cols < n or rows * cols > 20:
                    continue
                cells = [(r, c) for r in range(rows) for c in range(cols)]
                for trial in range(50):
                    if time.time() > deadline:
                        break
                    assign = [cells[np.random.randint(len(cells))] for _ in range(n)]
                    result = _solve_grid_lp(n, rows, cols, assign)
                    if result:
                        centers, s, val = result
                        if _is_valid(centers, s, n) and val > best_val:
                            best_val = val
                            best_centers = centers.copy()
                            best_sides = s.copy()
    
    # Final validation
    if not _is_valid(best_centers, best_sides, n):
        # Fallback to baseline
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
