# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

def squares_in_square10() -> tuple:
    """
    Solve the '10 squares in a square' problem using a MILP formulation.
    Maximizes the sum of side lengths of 10 axis-aligned, pairwise non-overlapping
    squares inside the unit square [0, 1]^2.

    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    """
    n = 10
    eps = 1e-9
    
    # Variable indices:
    # s: 0..n-1
    # x: n..2n-1
    # y: 2n..3n-1
    # u: 3n ... 3n + num_pairs - 1 (binary, x-separation)
    # v: 3n + num_pairs ... 3n + 2*num_pairs - 1 (binary, y-separation)
    
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    num_pairs = len(pairs)
    
    # Total variables
    num_vars = 3 * n + 2 * num_pairs
    
    # Objective: maximize sum(s_i) => minimize -sum(s_i)
    c = np.zeros(num_vars)
    c[:n] = -1.0
    
    # Bounds
    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    
    # s_i >= eps, s_i <= 1
    lb[:n] = eps
    ub[:n] = 1.0
    
    # x_i, y_i >= 0, <= 1
    # Already set by default lb=0, ub=1 for indices n..3n-1
    
    # Binary variables: 0 or 1
    integrality = np.zeros(num_vars)
    integrality[3*n:] = 1.0
    
    # Constraints A @ z <= b
    # We will build lists of coefficients and RHS values, then convert to sparse matrix
    
    # Constraint lists: list of (var_idx, coeff) pairs
    con_rows = []
    con_rhs = []
    
    # 1. Containment constraints:
    # x_i + s_i <= 1
    for i in range(n):
        row = [(n + i, 1.0), (i, 1.0)]
        con_rows.append(row)
        con_rhs.append(1.0)
        
    # y_i + s_i <= 1
    for i in range(n):
        row = [(2 * n + i, 1.0), (i, 1.0)]
        con_rows.append(row)
        con_rhs.append(1.0)
        
    # x_i >= 0 and y_i >= 0 are handled by bounds.
    
    # 2. Separation constraints
    # M = 1.0 (unit square)
    M = 1.0
    for k, (i, j) in enumerate(pairs):
        u_idx = 3 * n + k
        v_idx = 3 * n + num_pairs + k
        
        # u_ij = 1 means x_i + s_i <= x_j (i is to the left of j)
        # Constraint A: x_i + s_i - x_j <= M * (1 - u_ij)
        # x_i + s_i - x_j + M * u_ij <= M
        row_a = [(n + i, 1.0), (i, 1.0), (n + j, -1.0), (u_idx, M)]
        con_rows.append(row_a)
        con_rhs.append(M)
        
        # Constraint B: x_j + s_j - x_i <= M * u_ij
        # x_j + s_j - x_i - M * u_ij <= 0
        row_b = [(n + j, 1.0), (j, 1.0), (n + i, -1.0), (u_idx, -M)]
        con_rows.append(row_b)
        con_rhs.append(0.0)
        
        # v_ij = 1 means y_i + s_i <= y_j (i is below j)
        # Constraint C: y_i + s_i - y_j <= M * (1 - v_ij)
        # y_i + s_i - y_j + M * v_ij <= M
        row_c = [(2 * n + i, 1.0), (i, 1.0), (2 * n + j, -1.0), (v_idx, M)]
        con_rows.append(row_c)
        con_rhs.append(M)
        
        # Constraint D: y_j + s_j - y_i <= M * v_ij
        # y_j + s_j - y_i - M * v_ij <= 0
        row_d = [(2 * n + j, 1.0), (j, 1.0), (2 * n + i, -1.0), (v_idx, -M)]
        con_rows.append(row_d)
        con_rhs.append(0.0)
        
        # Constraint E: u_ij + v_ij >= 1
        # -u_ij - v_ij <= -1
        row_e = [(u_idx, -1.0), (v_idx, -1.0)]
        con_rows.append(row_e)
        con_rhs.append(-1.0)
        
    # Convert list of rows to sparse matrix
    rows_indices = []
    cols_indices = []
    data = []
    
    for r_idx, row in enumerate(con_rows):
        for var_idx, coeff in row:
            rows_indices.append(r_idx)
            cols_indices.append(var_idx)
            data.append(coeff)
            
    A = coo_matrix((data, (rows_indices, cols_indices)), shape=(len(con_rows), num_vars))
    A = A.tocsr()
    
    # Create LinearConstraint
    # A @ z <= con_rhs
    lc = LinearConstraint(A, -np.inf, np.array(con_rhs))
    
    bounds = Bounds(lb, ub)
    
    # Symmetry breaking: Fix square 0 to bottom-left corner (x=0, y=0)
    # This reduces the search space for the MILP solver.
    # x_0 is at index n, y_0 is at index 2*n
    lb[n] = 0.0
    ub[n] = 0.0
    lb[2*n] = 0.0
    ub[2*n] = 0.0
    
    # Solve
    res = milp(
        c=c,
        constraints=lc,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": 590, "mip_rel_gap": 1e-9}
    )
    
    if not res.success:
        # Fallback to simple grid if MILP fails
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides
        
    x_opt = res.x
    
    s = x_opt[:n]
    x = x_opt[n:2*n]
    y = x_opt[2*n:3*n]
    
    # Calculate centers
    cx = x + s / 2.0
    cy = y + s / 2.0
    
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    return centers, sides

# EVOLVE-BLOCK-END
