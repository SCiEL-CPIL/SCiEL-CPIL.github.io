# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def squares_in_square10() -> tuple:
    """
    Pack 10 axis-aligned squares in [0,1]^2 maximizing sum of side lengths.

    Strategy:
    1. For each possible assignment of separation axes (binary variables for each pair),
       formulate a linear program (LP) that maximizes the sum of side lengths.
    2. Solve each LP and keep the best feasible solution.
    3. Use a smarter enumeration: group pairs into "rows" (same y-range) and "columns".
    """
    t0 = time.time()
    deadline = t0 + 580.0  # safety margin before 630s kill

    n = 10
    tol = 1e-7
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pairs = len(pairs)

    def is_valid(centers, sides):
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # boundary
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
                return False
            if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
                return False
        # overlap
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    def solve_lp(separation_assignments):
        """
        separation_assignments: dict mapping (i,j) to 'x' or 'y'
        Returns (total_side_length, centers, sides) or None if infeasible.
        
        Variables:
        - x[i] = left edge of square i (n vars)
        - y[i] = bottom edge of square i (n vars)
        - s[i] = side length of square i (n vars)
        Total: 3n variables
        
        Objective: maximize sum(s[i])
        
        Constraints:
        - 0 <= x[i] <= 1, 0 <= y[i] <= 1, 0 <= s[i] <= 1
        - x[i] + s[i] <= 1, y[i] + s[i] <= 1
        - For each pair (i,j):
          If separation is 'x': x[j] >= x[i] + s[i]  OR  x[i] >= x[j] + s[j]
          If separation is 'y': y[j] >= y[i] + s[i]  OR  y[i] >= y[j] + s[j]
        """
        # Variable ordering: [x_0, ..., x_9, y_0, ..., y_9, s_0, ..., s_9]
        n_vars = 3 * n
        
        # Objective: maximize sum(s) -> minimize -sum(s)
        c = np.zeros(n_vars)
        for i in range(n):
            c[2 * n + i] = -1.0  # s[i] is at index 2n + i
        
        # Inequality constraints: A_ub @ var <= b_ub
        A_rows = []
        b_rows = []
        
        # Boundary: x[i] >= 0, y[i] >= 0, s[i] >= 0
        # x[i] + s[i] <= 1
        for i in range(n):
            row = np.zeros(n_vars)
            row[i] = 1.0  # x[i]
            row[2 * n + i] = 1.0  # s[i]
            A_rows.append(row)
            b_rows.append(1.0)
            
            row = np.zeros(n_vars)
            row[n + i] = 1.0  # y[i]
            row[2 * n + i] = 1.0  # s[i]
            A_rows.append(row)
            b_rows.append(1.0)
        
        # Separation constraints
        for (i, j), sep in separation_assignments.items():
            if sep == 'x':
                # Either x[j] >= x[i] + s[i]  =>  x[j] - x[i] - s[i] >= 0  =>  -x[j] + x[i] + s[i] <= 0
                # Or x[i] >= x[j] + s[j]  =>  x[i] - x[j] - s[j] >= 0  =>  -x[i] + x[j] + s[j] <= 0
                # Since we don't know which is left/right, we need both possibilities.
                # But wait - in an LP, we can't have OR directly.
                # We need to enumerate the ordering too, or use a different approach.
                # 
                # Actually, for a fixed separation axis, we need to decide the order.
                # Let's assume i is to the left of j: x[i] + s[i] <= x[j]
                # This gives: x[i] - x[j] + s[i] <= 0
                row = np.zeros(n_vars)
                row[i] = 1.0  # x[i]
                row[j] = -1.0  # -x[j]
                row[2 * n + i] = 1.0  # s[i]
                A_rows.append(row)
                b_rows.append(0.0)
            else:  # sep == 'y'
                # Assume i is below j: y[i] + s[i] <= y[j]
                row = np.zeros(n_vars)
                row[n + i] = 1.0  # y[i]
                row[n + j] = -1.0  # -y[j]
                row[2 * n + i] = 1.0  # s[i]
                A_rows.append(row)
                b_rows.append(0.0)
        
        if len(A_rows) > 0:
            A_ub = np.array(A_rows)
            b_ub = np.array(b_rows)
        else:
            A_ub = None
            b_ub = None
        
        # Bounds
        bounds = [(0, 1)] * n_vars  # x, y, s all in [0, 1]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0 and res.fun is not None:
                total = -res.fun
                x = res.x[:n]
                y = res.x[n:2*n]
                s = res.x[2*n:3*n]
                centers = np.column_stack([x + s/2, y + s/2])
                if is_valid(centers, s):
                    return total, centers, s
        except Exception:
            pass
        return None

    def solve_with_ordering(separation_assignments):
        """
        Try both orderings for each pair. This is exponential, so we limit it.
        For simplicity, let's just try the natural ordering (i before j in index).
        """
        return solve_lp(separation_assignments)

    # Smart enumeration of separation patterns
    # We'll try a structured approach:
    # 1. All pairs separated in x
    # 2. All pairs separated in y
    # 3. Grid-like patterns
    # 4. Random samples

    best_val = 0.0
    best_centers = None
    best_sides = None

    def try_solution(centers, sides):
        nonlocal best_val, best_centers, best_sides
        val = np.sum(sides)
        if val > best_val and is_valid(centers, sides):
            best_val = val
            best_centers = centers.copy()
            best_sides = sides.copy()

    # Pattern 1: All in x (one column)
    sep_all_x = {(i, j): 'x' for i, j in pairs}
    res = solve_with_ordering(sep_all_x)
    if res is not None:
        total, centers, sides = res
        try_solution(centers, sides)

    # Pattern 2: All in y (one row)
    sep_all_y = {(i, j): 'y' for i, j in pairs}
    res = solve_with_ordering(sep_all_y)
    if res is not None:
        total, centers, sides = res
        try_solution(centers, sides)

    # Pattern 3: Grid 2x5 (2 rows, 5 cols)
    # Pairs in same row: separated in x
    # Pairs in same col: separated in y
    # Pairs in different rows and cols: can go either way
    for rows in [2, 3, 4]:
        if time.time() > deadline - 30:
            break
        cols = -(-n // rows)  # ceil division
        sep = {}
        for i in range(n):
            for j in range(i + 1, n):
                ri, ci = i // cols, i % cols
                rj, cj = j // cols, j % cols
                if ri == rj:  # same row
                    sep[(i, j)] = 'x'
                elif ci == cj:  # same col
                    sep[(i, j)] = 'y'
                else:
                    # Different row and col: let's say separated in x if ri < rj, else y
                    sep[(i, j)] = 'x' if ri < rj else 'y'
        res = solve_with_ordering(sep)
        if res is not None:
            total, centers, sides = res
            try_solution(centers, sides)

    # Pattern 4: Random separation patterns
    np.random.seed(42)
    n_random = 100
    for k in range(n_random):
        if time.time() > deadline - 60:
            break
        sep = {}
        for i, j in pairs:
            sep[(i, j)] = 'x' if np.random.random() < 0.5 else 'y'
        res = solve_with_ordering(sep)
        if res is not None:
            total, centers, sides = res
            try_solution(centers, sides)

    # Fallback
    if best_centers is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
    else:
        centers = best_centers.astype(np.float64)
        sides = best_sides.astype(np.float64)

    return centers, sides

# EVOLVE-BLOCK-END
