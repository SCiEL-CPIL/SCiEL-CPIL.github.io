# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def squares_in_square10() -> tuple:
    """
    Solve the "10 squares in a square" packing problem by enumerating all
    shelf layouts and solving the exact LP for each, then polishing.
    
    A shelf layout partitions the unit square into horizontal shelves.
    Squares in each shelf are placed side-by-side (touching), so the
    sum of side lengths in each shelf equals 1. The shelf heights
    determine how much vertical space is available, which constrains
    the maximum side length of squares in that shelf.
    
    For a given shelf layout (partition of 10 into shelf sizes), the LP
    maximizes sum of all side lengths subject to:
      - sum of sides in each shelf = 1 (they fill the shelf width)
      - each side <= shelf_height (the square must fit vertically)
      - shelf heights sum <= 1 (they stack vertically)
    
    Actually, the constraint is more nuanced: the shelf height must be
    at least the max side in that shelf. Let's formulate properly.
    
    For shelf i with m_i squares of sides s_{i,1}, ..., s_{i,m_i}:
      - sum_j s_{i,j} = 1 (fill width)
      - h_i >= max_j s_{i,j} (shelf height >= largest square)
      - sum_i h_i <= 1 (shelves stack)
    
    We maximize sum of all s_{i,j}.
    
    This is an LP. We enumerate all compositions of 10 into positive
    integers (shelf sizes), solve the LP for each, and take the best.
    """
    n = 10
    tol = 1e-7
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 600.0
    
    best_score = 0.0
    best_solution = None
    
    # Generate all compositions of n into positive integers
    # A composition of n is a sequence (a_1, ..., a_k) with a_i >= 1, sum = n
    # Number of compositions is 2^(n-1) = 512 for n=10
    
    def generate_compositions(n):
        """Generate all compositions of n into positive integers."""
        if n == 0:
            yield []
            return
        for first in range(1, n + 1):
            for rest in generate_compositions(n - first):
                yield [first] + rest
    
    compositions = list(generate_compositions(n))
    
    for comp in compositions:
        if time.time() > deadline - 10.0:
            break
        
        num_shelves = len(comp)
        m = comp  # m[i] = number of squares in shelf i
        
        # Variables:
        #   s[i][j] for i in range(num_shelves), j in range(m[i]) : side lengths
        #   h[i] for i in range(num_shelves) : shelf heights
        #
        # Let's flatten: first all s variables, then all h variables.
        # Total s variables: n = 10
        # Total h variables: num_shelves
        # Total variables: 10 + num_shelves
        
        # Index mapping
        s_idx = {}
        var_idx = 0
        for i in range(num_shelves):
            for j in range(m[i]):
                s_idx[(i, j)] = var_idx
                var_idx += 1
        h_idx = {}
        for i in range(num_shelves):
            h_idx[i] = var_idx
            var_idx += 1
        
        total_vars = var_idx
        
        # Objective: maximize sum of all s variables
        # linprog minimizes, so negate
        c = np.zeros(total_vars)
        for (i, j) in s_idx:
            c[s_idx[(i, j)]] = -1.0  # minimize negative sum = maximize sum
        
        # Constraints:
        # 1. For each shelf i: sum_j s[i][j] = 1
        A_eq = []
        b_eq = []
        for i in range(num_shelves):
            row = np.zeros(total_vars)
            for j in range(m[i]):
                row[s_idx[(i, j)]] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)
        
        # 2. For each shelf i: h[i] - max_j s[i][j] >= 0
        #    This is h[i] >= s[i][j] for all j, i.e., h[i] - s[i][j] >= 0
        #    linprog uses A_ub @ x <= b_ub, so: -h[i] + s[i][j] <= 0
        A_ub = []
        b_ub = []
        for i in range(num_shelves):
            for j in range(m[i]):
                row = np.zeros(total_vars)
                row[h_idx[i]] = -1.0
                row[s_idx[(i, j)]] = 1.0
                A_ub.append(row)
                b_ub.append(0.0)
        
        # 3. sum_i h[i] <= 1
        row = np.zeros(total_vars)
        for i in range(num_shelves):
            row[h_idx[i]] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
        
        # Bounds: all variables >= 0
        bounds = [(0, None)] * total_vars
        
        if A_ub:
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
        else:
            A_ub = np.empty((0, total_vars))
            b_ub = np.empty(0)
        
        if A_eq:
            A_eq = np.array(A_eq)
            b_eq = np.array(b_eq)
        else:
            A_eq = np.empty((0, total_vars))
            b_eq = np.empty(0)
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
        
        if result.success:
            # Extract solution
            x = result.x
            s_vals = np.zeros(n)
            pos = 0
            for i in range(num_shelves):
                for j in range(m[i]):
                    s_vals[pos] = x[s_idx[(i, j)]]
                    pos += 1
            h_vals = np.array([x[h_idx[i]] for i in range(num_shelves)])
            
            score = np.sum(s_vals)
            if score > best_score:
                best_score = score
                best_solution = (s_vals, h_vals, comp.copy())
    
    if best_solution is None:
        # Fallback to simple grid
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides
    
    s_vals, h_vals, comp = best_solution
    num_shelves = len(comp)
    
    # Construct centers
    centers = np.zeros((n, 2))
    sides_out = np.zeros(n)
    
    y_offset = 0.0
    pos = 0
    for i in range(num_shelves):
        m_i = comp[i]
        h_i = h_vals[i]
        # Squares in shelf i are placed side by side
        x_offset = 0.0
        for j in range(m_i):
            s = s_vals[pos]
            cx = x_offset + s / 2.0
            cy = y_offset + h_i / 2.0
            centers[pos, 0] = cx
            centers[pos, 1] = cy
            sides_out[pos] = s
            x_offset += s
            pos += 1
        y_offset += h_i
    
    # Verify feasibility
    # Check bounds
    for i in range(n):
        s = sides_out[i]
        cx, cy = centers[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
            # Fallback
            k = int(np.ceil(np.sqrt(n)))
            side = 1.0 / k
            idx = np.arange(n)
            cols = idx % k
            rows = idx // k
            cx = (cols + 0.5) * side
            cy = (rows + 0.5) * side
            centers = np.stack([cx, cy], axis=1).astype(np.float64)
            sides_out = np.full(n, side, dtype=np.float64)
            return centers, sides_out
    
    # Check pairwise non-overlap
    for i in range(n):
        for j in range(i+1, n):
            si, sj = sides_out[i], sides_out[j]
            cx_i, cy_i = centers[i]
            cx_j, cy_j = centers[j]
            t = (si + sj) / 2.0 - tol
            if abs(cx_i - cx_j) < t and abs(cy_i - cy_j) < t:
                # Overlap detected, fallback
                k = int(np.ceil(np.sqrt(n)))
                side = 1.0 / k
                idx = np.arange(n)
                cols = idx % k
                rows = idx // k
                cx = (cols + 0.5) * side
                cy = (rows + 0.5) * side
                centers = np.stack([cx, cy], axis=1).astype(np.float64)
                sides_out = np.full(n, side, dtype=np.float64)
                return centers, sides_out
    
    centers = centers.astype(np.float64)
    sides_out = sides_out.astype(np.float64)
    
    return centers, sides_out


# EVOLVE-BLOCK-END
