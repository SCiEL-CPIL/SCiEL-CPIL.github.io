import numpy as np
import time
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import cdist

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    def validate(centers, sides):
        """Check if configuration is valid; return (valid, score)."""
        tol = 1e-7
        s = np.maximum(sides, 1e-8)
        for i in range(n):
            if centers[i, 0] - s[i]/2 < -tol or centers[i, 0] + s[i]/2 > 1 + tol:
                return False, 0.0
            if centers[i, 1] - s[i]/2 < -tol or centers[i, 1] + s[i]/2 > 1 + tol:
                return False, 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (s[i] + s[j])/2 - tol
                if dx < t and dy < t:
                    return False, 0.0
        return True, np.sum(s)

    def clamp_to_box(centers, sides):
        """Clamp centers so squares fit in [0,1]^2."""
        s = np.maximum(sides, 1e-8)
        cx = centers[:, 0].copy()
        cy = centers[:, 1].copy()
        for i in range(n):
            cx[i] = np.clip(cx[i], s[i]/2, 1 - s[i]/2)
            cy[i] = np.clip(cy[i], s[i]/2, 1 - s[i]/2)
        return np.stack([cx, cy], axis=1), s

    def solve_lp_for_pattern(rows, cols, row_sizes, col_sizes):
        """
        Solve LP for a given grid pattern.
        rows: list of row indices for each row group
        cols: list of col indices for each col group
        row_sizes: target relative sizes for rows
        col_sizes: target relative sizes for cols
        """
        # Variables: x_i, y_i, s_i for i=0..n-1
        # Maximize sum(s_i)
        # Subject to:
        #   x_i - s_i/2 >= 0
        #   1 - x_i - s_i/2 >= 0
        #   y_i - s_i/2 >= 0
        #   1 - y_i - s_i/2 >= 0
        #   s_i >= eps
        #   For each pair in same row: |x_i - x_j| >= (s_i+s_j)/2
        #   For each pair in same col: |y_i - y_j| >= (s_i+s_j)/2
        #   For pairs in different rows AND different cols: no constraint (they can overlap in x or y)
        
        # Actually, for a grid pattern, squares in the same row must not overlap in x,
        # and squares in the same column must not overlap in y.
        # But we need to be more careful: the pattern defines which squares share rows/cols.
        
        # Let's use a simpler approach: for each pattern, define the ordering and
        # solve a system of linear inequalities.
        
        # Pattern: rows is a list where rows[r] = list of square indices in row r
        # cols is a list where cols[c] = list of square indices in col c
        
        # We'll use scipy linprog to maximize sum(s_i)
        
        # Decision variables: x[0..n-1], y[n..2n-1], s[2n..3n-1]
        # Objective: maximize sum(s_i) => minimize -sum(s_i)
        
        c = np.zeros(3*n)
        for i in range(n):
            c[2*n + i] = -1.0  # minimize -s_i
        
        # Inequality constraints: A_ub @ x <= b_ub
        A_rows = []
        b_rows = []
        
        # Boundary constraints: x_i - s_i/2 >= 0 => -x_i + s_i/2 <= 0
        for i in range(n):
            row = np.zeros(3*n)
            row[i] = -1.0
            row[2*n+i] = 0.5
            A_rows.append(row)
            b_rows.append(0.0)
            
            # 1 - x_i - s_i/2 >= 0 => x_i + s_i/2 <= 1
            row = np.zeros(3*n)
            row[i] = 1.0
            row[2*n+i] = 0.5
            A_rows.append(row)
            b_rows.append(1.0)
            
            # y_i - s_i/2 >= 0 => -y_i + s_i/2 <= 0
            row = np.zeros(3*n)
            row[n+i] = -1.0
            row[2*n+i] = 0.5
            A_rows.append(row)
            b_rows.append(0.0)
            
            # 1 - y_i - s_i/2 >= 0 => y_i + s_i/2 <= 1
            row = np.zeros(3*n)
            row[n+i] = 1.0
            row[2*n+i] = 0.5
            A_rows.append(row)
            b_rows.append(1.0)
        
        # Same-row separation: for squares i,j in same row, x_i + s_i/2 <= x_j - s_j/2 (if i left of j)
        # We assume the ordering within each row/col is fixed by the pattern
        for r in range(len(rows)):
            row_squares = rows[r]
            for k in range(len(row_squares)-1):
                i = row_squares[k]
                j = row_squares[k+1]
                # x_i + s_i/2 <= x_j - s_j/2 => x_i - x_j + s_i/2 + s_j/2 <= 0
                row = np.zeros(3*n)
                row[i] = 1.0
                row[j] = -1.0
                row[2*n+i] = 0.5
                row[2*n+j] = 0.5
                A_rows.append(row)
                b_rows.append(0.0)
        
        # Same-col separation: for squares i,j in same col, y_i + s_i/2 <= y_j - s_j/2 (if i above j)
        for cidx in range(len(cols)):
            col_squares = cols[cidx]
            for k in range(len(col_squares)-1):
                i = col_squares[k]
                j = col_squares[k+1]
                # y_i + s_i/2 <= y_j - s_j/2 => y_i - y_j + s_i/2 + s_j/2 <= 0
                row = np.zeros(3*n)
                row[n+i] = 1.0
                row[n+j] = -1.0
                row[2*n+i] = 0.5
                row[2*n+j] = 0.5
                A_rows.append(row)
                b_rows.append(0.0)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        # Bounds: s_i >= 1e-8, x_i, y_i in [0, 1]
        bounds = [(0, 1)] * n + [(0, 1)] * n + [(1e-8, 1)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success:
                return None, None, 0.0
            
            x = res.x
            cx = x[0:n]
            cy = x[n:2*n]
            s = x[2*n:3*n]
            centers = np.stack([cx, cy], axis=1)
            score = np.sum(s)
            return centers, s, score
        except Exception:
            return None, None, 0.0

    def generate_grid_patterns():
        """Generate various grid patterns for 10 squares."""
        patterns = []
        
        # 2x5 grid
        rows = [[0,1,2,3,4],[5,6,7,8,9]]
        cols = [[0,5],[1,6],[2,7],[3,8],[4,9]]
        patterns.append((rows, cols))
        
        # 5x2 grid
        rows = [[0,1],[2,3],[4,5],[6,7],[8,9]]
        cols = [[0,2,4,6,8],[1,3,5,7,9]]
        patterns.append((rows, cols))
        
        # 1x10 grid (all in one row)
        rows = [[0,1,2,3,4,5,6,7,8,9]]
        cols = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
        patterns.append((rows, cols))
        
        # 10x1 grid (all in one column)
        rows = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
        cols = [[0,1,2,3,4,5,6,7,8,9]]
        patterns.append((rows, cols))
        
        # 3-row layout: 4,3,3
        rows = [[0,1,2,3],[4,5,6],[7,8,9]]
        cols = [[0,4,7],[1,5,8],[2,6,9],[3]]
        patterns.append((rows, cols))
        
        # 3-row layout: 3,4,3
        rows = [[0,1,2],[3,4,5,6],[7,8,9]]
        cols = [[0,3,7],[1,4,8],[2,5,9],[6]]
        patterns.append((rows, cols))
        
        # 3-row layout: 3,3,4
        rows = [[0,1,2],[3,4,5],[6,7,8,9]]
        cols = [[0,3,6],[1,4,7],[2,5,8],[9]]
        patterns.append((rows, cols))
        
        # 4-row layout: 3,3,2,2
        rows = [[0,1,2],[3,4,5],[6,7],[8,9]]
        cols = [[0,3,6,8],[1,4,7,9],[2,5]]
        patterns.append((rows, cols))
        
        # 4-row layout: 2,2,3,3
        rows = [[0,1],[2,3],[4,5,6],[7,8,9]]
        cols = [[0,2,4,7],[1,3,5,8],[6,9]]
        patterns.append((rows, cols))
        
        # 4-row layout: 3,2,3,2
        rows = [[0,1,2],[3,4],[5,6,7],[8,9]]
        cols = [[0,3,5,8],[1,4,6,9],[2,7]]
        patterns.append((rows, cols))
        
        # 4-row layout: 2,3,2,3
        rows = [[0,1],[2,3,4],[5,6],[7,8,9]]
        cols = [[0,2,5,7],[1,3,6,8],[4,9]]
        patterns.append((rows, cols))
        
        # 2-row layout with varying counts: 5,5 is already there
        # Try 4,6
        rows = [[0,1,2,3],[4,5,6,7,8,9]]
        cols = [[0,4],[1,5],[2,6],[3,7],[8],[9]]
        patterns.append((rows, cols))
        
        # 6,4
        rows = [[0,1,2,3,4,5],[6,7,8,9]]
        cols = [[0,6],[1,7],[2,8],[3,9],[4],[5]]
        patterns.append((rows, cols))
        
        # 3-row: 2,3,5
        rows = [[0,1],[2,3,4],[5,6,7,8,9]]
        cols = [[0,2,5],[1,3,6],[4,7],[8],[9]]
        patterns.append((rows, cols))
        
        # 3-row: 5,3,2
        rows = [[0,1,2,3,4],[5,6,7],[8,9]]
        cols = [[0,5,8],[1,6,9],[2,7],[3],[4]]
        patterns.append((rows, cols))
        
        # 3-row: 2,5,3
        rows = [[0,1],[2,3,4,5,6],[7,8,9]]
        cols = [[0,2,7],[1,3,8],[4,9],[5],[6]]
        patterns.append((rows, cols))
        
        # 3-row: 3,5,2
        rows = [[0,1,2],[3,4,5,6,7],[8,9]]
        cols = [[0,3,8],[1,4,9],[2,5],[6],[7]]
        patterns.append((rows, cols))
        
        return patterns

    def local_polish(centers, sides, max_iter=200):
        """Local polish using SLSQP to refine a valid configuration."""
        def to_flat(centers, sides):
            x = np.zeros(3 * n)
            for i in range(n):
                x[3*i] = centers[i, 0]
                x[3*i+1] = centers[i, 1]
                x[3*i+2] = sides[i]
            return x

        def from_flat(x):
            cx = x[0::3]
            cy = x[1::3]
            s = np.maximum(x[2::3], 1e-8)
            centers = np.stack([cx, cy], axis=1)
            return centers, s

        def objective(x):
            s = x[2::3]
            return -np.sum(np.maximum(s, 1e-10))

        def gradient_obj(x):
            s = x[2::3]
            grad = np.zeros(3*n)
            for i in range(n):
                grad[3*i+2] = -1.0 if s[i] > 1e-10 else 0.0
            return grad

        # Constraints
        cons = []
        for i in range(n):
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i: x[3*i] - x[3*i+2]/2,
                         'jac': lambda x, i=i: np.array([1.0, 0.0, -0.5])})
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]/2,
                         'jac': lambda x, i=i: np.array([-1.0, 0.0, -0.5])})
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]/2,
                         'jac': lambda x, i=i: np.array([0.0, 1.0, -0.5])})
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]/2,
                         'jac': lambda x, i=i: np.array([0.0, -1.0, -0.5])})
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i: x[3*i+2],
                         'jac': lambda x, i=i: np.array([0.0, 0.0, 1.0])})
        
        for i in range(n):
            for j in range(i+1, n):
                def overlap_con(x, i=i, j=j):
                    dx = abs(x[3*i] - x[3*j])
                    dy = abs(x[3*i+1] - x[3*j+1])
                    req = (x[3*i+2] + x[3*j+2]) / 2
                    return -(req - dx) * (req - dy)
                
                def overlap_jac(x, i=i, j=j):
                    dx = abs(x[3*i] - x[3*j])
                    dy = abs(x[3*i+1] - x[3*j+1])
                    req = (x[3*i+2] + x[3*j+2]) / 2
                    sign_dx = 1.0 if x[3*i] > x[3*j] else -1.0
                    sign_dy = 1.0 if x[3*i+1] > x[3*j+1] else -1.0
                    a = req - dx
                    b = req - dy
                    grad = np.zeros(3*n)
                    grad[3*i] = sign_dx * b
                    grad[3*i+1] = sign_dy * a
                    grad[3*i+2] = -0.5 * (a + b)
                    grad[3*j] = -sign_dx * b
                    grad[3*j+1] = -sign_dy * a
                    grad[3*j+2] = -0.5 * (a + b)
                    return grad
                
                cons.append({'type': 'ineq',
                             'fun': overlap_con,
                             'jac': overlap_jac})
        
        bounds = [(0.0, 1.0)] * (2*n) + [(1e-8, 1.0)] * n
        
        try:
            x0 = to_flat(centers, sides)
            res = minimize(objective, x0, jac=gradient_obj, method='SLSQP',
                          bounds=bounds, constraints=cons,
                          options={'maxiter': max_iter, 'ftol': 1e-10})
            if res.success:
                c_res, s_res = from_flat(res.x)
                c_res, s_res = clamp_to_box(c_res, s_res)
                valid, score = validate(c_res, s_res)
                if valid:
                    return c_res, s_res, score
        except Exception:
            pass
        
        return centers, sides, validate(centers, sides)[1]

    # Generate and solve all patterns
    patterns = generate_grid_patterns()
    
    best_centers = None
    best_sides = None
    best_score = 0.0

    for rows, cols in patterns:
        if time.time() > deadline - 30:
            break
        centers, sides, score = solve_lp_for_pattern(rows, cols, None, None)
        if centers is not None and score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_sides = sides.copy()

    # If no valid LP solution found, use a simple grid
    if best_centers is None:
        centers_init = np.array([
            [0.25, 0.25], [0.75, 0.25],
            [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
            [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875]
        ])
        sides_init = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
        best_centers, best_sides = clamp_to_box(centers_init, sides_init)
        valid, best_score = validate(best_centers, best_sides)
        if not valid:
            return centers_init, sides_init

    # Local polish on the best solution
    if time.time() < deadline - 20:
        polished_c, polished_s, polished_score = local_polish(best_centers, best_sides, max_iter=300)
        if polished_score > best_score:
            best_score = polished_score
            best_centers = polished_c
            best_sides = polished_s

    # Second polish pass with tighter tolerance
    if time.time() < deadline - 10:
        polished_c, polished_s, polished_score = local_polish(best_centers, best_sides, max_iter=200)
        if polished_score > best_score:
            best_score = polished_score
            best_centers = polished_c
            best_sides = polished_s

    # Final validation
    valid, final_score = validate(best_centers, best_sides)
    if not valid:
        # Fallback to initial grid
        centers_init = np.array([
            [0.25, 0.25], [0.75, 0.25],
            [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
            [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875]
        ])
        sides_init = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
        return centers_init, sides_init

    return best_centers, best_sides
