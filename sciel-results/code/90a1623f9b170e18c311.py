# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from itertools import product

_TOL = 1e-7
_DEADLINE_S = 620.0


def _feasible(centers, sides, tol=_TOL):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _grid_init(n=10):
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


def _solve_milp(n, start_time, deadline):
    """
    Solve MILP for square packing.
    Variables:
      cx[0..n-1], cy[0..n-1], s[0..n-1] (continuous)
      For each pair (i,j), 4 binary vars: b_xneg, b_xpos, b_yneg, b_ypos
      Constraint: sum of 4 binaries = 1 (at least one separation direction chosen)
    
    Separation constraints (big-M formulation):
      If b_xneg=1: cx_j - cx_i >= 0.5*(s_i+s_j)
      If b_xpos=1: cx_i - cx_j >= 0.5*(s_i+s_j)
      If b_yneg=1: cy_j - cy_i >= 0.5*(s_i+s_j)
      If b_ypos=1: cy_i - cy_j >= 0.5*(s_i+s_j)
    
    Using big-M: for each pair and direction, constraint is:
      separation_expr >= 0.5*(s_i+s_j) - M*(1 - b_dir)
    where M is a large enough bound.
    """
    if time.time() > deadline:
        return None
    
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(pairs)
    
    # Variable indices
    cx_idx = np.arange(n)
    cy_idx = np.arange(n, 2 * n)
    s_idx = np.arange(2 * n, 3 * n)
    # Binary vars: 4 per pair
    b_start = 3 * n
    total_vars = 3 * n + 4 * num_pairs
    
    # Objective: maximize sum(s) -> minimize -sum(s)
    c = np.zeros(total_vars)
    for i in range(n):
        c[s_idx[i]] = -1.0
    
    A_rows = []
    b_lower = []
    b_upper = []
    
    # Boundary constraints
    for i in range(n):
        # cx_i - 0.5 s_i >= 0
        row = np.zeros(total_vars)
        row[cx_idx[i]] = 1.0
        row[s_idx[i]] = -0.5
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(np.inf)
        
        # cx_i + 0.5 s_i <= 1
        row = np.zeros(total_vars)
        row[cx_idx[i]] = 1.0
        row[s_idx[i]] = 0.5
        A_rows.append(row)
        b_lower.append(-np.inf)
        b_upper.append(1.0)
        
        # cy_i - 0.5 s_i >= 0
        row = np.zeros(total_vars)
        row[cy_idx[i]] = 1.0
        row[s_idx[i]] = -0.5
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(np.inf)
        
        # cy_i + 0.5 s_i <= 1
        row = np.zeros(total_vars)
        row[cy_idx[i]] = 1.0
        row[s_idx[i]] = 0.5
        A_rows.append(row)
        b_lower.append(-np.inf)
        b_upper.append(1.0)
    
    # Big-M value: max possible side is 1, so max separation needed is 1
    M = 2.0
    
    # Separation constraints using big-M
    for k, (i, j) in enumerate(pairs):
        b_xneg = b_start + 4 * k
        b_xpos = b_start + 4 * k + 1
        b_yneg = b_start + 4 * k + 2
        b_ypos = b_start + 4 * k + 3
        
        # At least one direction: b_xneg + b_xpos + b_yneg + b_ypos >= 1
        row = np.zeros(total_vars)
        row[b_xneg] = 1.0
        row[b_xpos] = 1.0
        row[b_yneg] = 1.0
        row[b_ypos] = 1.0
        A_rows.append(row)
        b_lower.append(1.0)
        b_upper.append(np.inf)
        
        # x_neg: cx_j - cx_i - 0.5*(s_i+s_j) >= -M*(1-b_xneg)
        # i.e., cx_j - cx_i - 0.5*s_i - 0.5*s_j + M*b_xneg >= 0
        row = np.zeros(total_vars)
        row[cx_idx[j]] = 1.0
        row[cx_idx[i]] = -1.0
        row[s_idx[i]] = -0.5
        row[s_idx[j]] = -0.5
        row[b_xneg] = M
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(np.inf)
        
        # x_pos: cx_i - cx_j - 0.5*(s_i+s_j) >= -M*(1-b_xpos)
        row = np.zeros(total_vars)
        row[cx_idx[i]] = 1.0
        row[cx_idx[j]] = -1.0
        row[s_idx[i]] = -0.5
        row[s_idx[j]] = -0.5
        row[b_xpos] = M
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(np.inf)
        
        # y_neg: cy_j - cy_i - 0.5*(s_i+s_j) >= -M*(1-b_yneg)
        row = np.zeros(total_vars)
        row[cy_idx[j]] = 1.0
        row[cy_idx[i]] = -1.0
        row[s_idx[i]] = -0.5
        row[s_idx[j]] = -0.5
        row[b_yneg] = M
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(np.inf)
        
        # y_pos: cy_i - cy_j - 0.5*(s_i+s_j) >= -M*(1-b_ypos)
        row = np.zeros(total_vars)
        row[cy_idx[i]] = 1.0
        row[cy_idx[j]] = -1.0
        row[s_idx[i]] = -0.5
        row[s_idx[j]] = -0.5
        row[b_ypos] = M
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(np.inf)
    
    A = np.array(A_rows)
    b_lower = np.array(b_lower)
    b_upper = np.array(b_upper)
    
    # Bounds
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    lb[s_idx] = 1e-6
    ub[s_idx] = 1.0
    # Binary bounds
    b_indices = np.arange(b_start, total_vars)
    ub[b_indices] = 1.0
    lb[b_indices] = 0.0
    
    # Integrality: first 3n continuous, rest binary
    integrality = np.zeros(total_vars)
    integrality[b_start:] = 1  # binary
    
    try:
        constraint = LinearConstraint(A, b_lower, b_upper)
        bounds = Bounds(lb, ub)
        res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0:  # Optimal
            cx = res.x[cx_idx]
            cy = res.x[cy_idx]
            s = res.x[s_idx]
            centers = np.stack([cx, cy], axis=1)
            return centers, s
        elif res.status == 1:  # Infeasible
            return None
        elif res.status == 2:  # Unbounded
            return None
        else:
            return None
    except Exception as e:
        return None


def _try_structured_patterns_lp(n, start_time, deadline):
    """
    Fallback: try structured LP patterns if MILP fails or times out.
    """
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(pairs)
    
    best_score = -np.inf
    best_result = None
    
    def _solve_lp_for_pattern(pattern):
        if time.time() > deadline:
            return None
        
        cx_idx = np.arange(n)
        cy_idx = np.arange(n, 2 * n)
        s_idx = np.arange(2 * n, 3 * n)
        total_vars = 3 * n
        
        c = np.zeros(total_vars)
        for i in range(n):
            c[s_idx[i]] = -1.0
        
        A_rows = []
        b_lower = []
        b_upper = []
        
        for i in range(n):
            row = np.zeros(total_vars)
            row[cx_idx[i]] = 1.0
            row[s_idx[i]] = -0.5
            A_rows.append(row)
            b_lower.append(0.0)
            b_upper.append(np.inf)
            
            row = np.zeros(total_vars)
            row[cx_idx[i]] = 1.0
            row[s_idx[i]] = 0.5
            A_rows.append(row)
            b_lower.append(-np.inf)
            b_upper.append(1.0)
            
            row = np.zeros(total_vars)
            row[cy_idx[i]] = 1.0
            row[s_idx[i]] = -0.5
            A_rows.append(row)
            b_lower.append(0.0)
            b_upper.append(np.inf)
            
            row = np.zeros(total_vars)
            row[cy_idx[i]] = 1.0
            row[s_idx[i]] = 0.5
            A_rows.append(row)
            b_lower.append(-np.inf)
            b_upper.append(1.0)
        
        for k, ((i, j), sep) in enumerate(zip(pairs, pattern)):
            if sep == 0:
                row = np.zeros(total_vars)
                row[cx_idx[j]] = 1.0
                row[cx_idx[i]] = -1.0
                row[s_idx[i]] = -0.5
                row[s_idx[j]] = -0.5
                A_rows.append(row)
                b_lower.append(0.0)
                b_upper.append(np.inf)
            elif sep == 1:
                row = np.zeros(total_vars)
                row[cx_idx[i]] = 1.0
                row[cx_idx[j]] = -1.0
                row[s_idx[i]] = -0.5
                row[s_idx[j]] = -0.5
                A_rows.append(row)
                b_lower.append(0.0)
                b_upper.append(np.inf)
            elif sep == 2:
                row = np.zeros(total_vars)
                row[cy_idx[j]] = 1.0
                row[cy_idx[i]] = -1.0
                row[s_idx[i]] = -0.5
                row[s_idx[j]] = -0.5
                A_rows.append(row)
                b_lower.append(0.0)
                b_upper.append(np.inf)
            elif sep == 3:
                row = np.zeros(total_vars)
                row[cy_idx[i]] = 1.0
                row[cy_idx[j]] = -1.0
                row[s_idx[i]] = -0.5
                row[s_idx[j]] = -0.5
                A_rows.append(row)
                b_lower.append(0.0)
                b_upper.append(np.inf)
        
        A = np.array(A_rows)
        b_lower = np.array(b_lower)
        b_upper = np.array(b_upper)
        
        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)
        lb[s_idx] = 1e-6
        ub[s_idx] = 1.0
        
        try:
            res = linprog(c, A_ub=A, b_ub=b_upper, A_eq=None, b_eq=None,
                          bounds=list(zip(lb, ub)), method='highs')
            if res.success:
                cx = res.x[cx_idx]
                cy = res.x[cy_idx]
                s = res.x[s_idx]
                centers = np.stack([cx, cy], axis=1)
                return centers, s
        except Exception:
            pass
        return None
    
    # Try 2x5 grid pattern
    if time.time() < deadline:
        pattern = []
        for i in range(n):
            for j in range(i + 1, n):
                row_i = i // 5
                row_j = j // 5
                if row_i == row_j:
                    pattern.append(0)  # x_neg
                else:
                    pattern.append(2 if row_i < row_j else 3)
        
        result = _solve_lp_for_pattern(pattern)
        if result is not None:
            centers, sides = result
            if _feasible(centers, sides):
                score = np.sum(sides)
                if score > best_score:
                    best_score = score
                    best_result = (centers, sides)
    
    # Try 5x2 grid pattern
    if time.time() < deadline:
        pattern = []
        for i in range(n):
            for j in range(i + 1, n):
                col_i = i % 2
                col_j = j % 2
                if col_i == col_j:
                    pattern.append(2)  # y_neg
                else:
                    pattern.append(0 if col_i < col_j else 1)
        
        result = _solve_lp_for_pattern(pattern)
        if result is not None:
            centers, sides = result
            if _feasible(centers, sides):
                score = np.sum(sides)
                if score > best_score:
                    best_score = score
                    best_result = (centers, sides)
    
    return best_result


def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + _DEADLINE_S - 15
    
    # First, try MILP for global optimization
    result = _solve_milp(n, start_time, deadline)
    if result is not None:
        centers, sides = result
        if _feasible(centers, sides):
            return centers, sides
    
    # Fallback to structured LP patterns
    if time.time() < deadline:
        result = _try_structured_patterns_lp(n, start_time, deadline)
        if result is not None:
            centers, sides = result
            if _feasible(centers, sides):
                return centers, sides
    
    # Fallback to grid
    centers, sides = _grid_init(n)
    if _feasible(centers, sides):
        return centers, sides
    
    # Last resort
    centers = np.zeros((n, 2))
    sides = np.full(n, 0.1)
    return centers, sides


# EVOLVE-BLOCK-END
