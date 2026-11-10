import numpy as np
import time
import random
from math import gcd
from scipy.optimize import milp, Bounds, LinearConstraint
import itertools

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    
    # Make first non-zero entry positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def count_incidences(points, lines):
    """
    points: (n, 2) array
    lines: (m, 3) array (A, B, C)
    Returns number of (p, l) such that A*x + B*y + C == 0
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    count = np.sum(vals == 0)
    return int(count)

def generate_candidate_pool():
    """
    Generate a diverse pool of candidate lines and points.
    We focus on lines that pass through multiple integer points in a reasonable range.
    """
    # Grid for generating lines
    # We want lines with small coefficients to maximize intersections in integer lattice
    lines_set = set()
    
    # Generate lines y = m*x + c with small m, c
    for m in range(-5, 6):
        for c in range(-20, 21):
            # Line: m*x - y + c = 0 => A=m, B=-1, C=c
            nl = normalize_line(m, -1, c)
            if nl:
                lines_set.add(nl)
                
    # Generate lines x = c
    for c in range(-20, 21):
        nl = normalize_line(1, 0, -c)
        if nl:
            lines_set.add(nl)
            
    # Generate lines y = c
    for c in range(-20, 21):
        nl = normalize_line(0, 1, -c)
        if nl:
            lines_set.add(nl)
            
    # Generate some diagonals x + y = c
    for c in range(-30, 31):
        nl = normalize_line(1, 1, -c)
        if nl:
            lines_set.add(nl)
            
    # Generate x - y = c
    for c in range(-30, 31):
        nl = normalize_line(1, -1, -c)
        if nl:
            lines_set.add(nl)

    lines_list = list(lines_set)
    # Shuffle to randomize selection if we need to trim
    random.shuffle(lines_list)
    
    # We will use a subset of these lines as candidates for the ILP
    # Let's take up to 150 lines to keep the ILP size manageable
    # The more lines, the more potential high-incidence configurations
    candidate_lines = np.array(lines_list[:150], dtype=np.int64)
    
    # Generate candidate points: Intersections of these lines
    points_set = set()
    n_lines = len(candidate_lines)
    
    # To keep point generation fast, we only consider intersections of a random subset of lines
    # or all pairs if n_lines is small. 150 lines -> ~11k pairs. That's fine.
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            A1, B1, C1 = candidate_lines[i]
            A2, B2, C2 = candidate_lines[j]
            
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue # Parallel or identical
            
            x_num = B1 * C2 - B2 * C1
            y_num = A2 * C1 - A1 * C2
            
            # Check if integer
            if x_num % det == 0 and y_num % det == 0:
                x = x_num // det
                y = y_num // det
                # Filter to reasonable range to avoid huge coordinates
                if -100 <= x <= 100 and -100 <= y <= 100:
                    points_set.add((x, y))
                    
    # Add grid points to ensure we have enough candidates
    for x in range(-20, 21):
        for y in range(-20, 21):
            points_set.add((x, y))
            
    points_list = list(points_set)
    random.shuffle(points_list)
    
    # Take up to 300 points
    candidate_points = np.array(points_list[:300], dtype=np.int64)
    
    return candidate_points, candidate_lines

def solve_ilp(candidate_points, candidate_lines, time_limit=300):
    """
    Solve the ILP to select 54 points and 27 lines maximizing incidences.
    """
    n_p = len(candidate_points)
    n_l = len(candidate_lines)
    
    # Precompute incidence matrix M (n_p, n_l)
    px = candidate_points[:, 0].reshape(-1, 1)
    py = candidate_points[:, 1].reshape(-1, 1)
    A = candidate_lines[:, 0].reshape(1, -1)
    B = candidate_lines[:, 1].reshape(1, -1)
    C = candidate_lines[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    M = (vals == 0).astype(np.int64)
    
    # Variables:
    # x_p (n_p) : 1 if point p is selected
    # y_l (n_l) : 1 if line l is selected
    # z_pl (n_p * n_l) : 1 if point p is on line l AND both are selected
    
    # Objective: Maximize sum(z_pl)
    # Minimize -sum(z_pl)
    
    # c vector for milp
    c = np.zeros(n_p + n_l + n_p * n_l)
    # z variables are at the end
    c[n_p + n_l:] = -1.0
    
    # Constraints:
    # 1. Sum x_p = 54
    # 2. Sum y_l = 27
    # 3. z_pl <= x_p
    # 4. z_pl <= y_l
    # 5. z_pl >= x_p + y_l - 1  =>  z_pl - x_p - y_l >= -1
    
    # Bounds
    integrality = np.ones(n_p + n_l + n_p * n_l)
    
    # We will build the constraint matrix A_ub and b_ub for the linear constraints
    # However, milp takes LinearConstraint objects.
    
    # Let's construct the matrix for the "z <= x" and "z <= y" and "z >= x+y-1" constraints.
    # This is sparse. We can use scipy.sparse.
    
    import scipy.sparse as sp
    
    # Constraint 1: sum x = 54
    # Constraint 2: sum y = 27
    # These are equality constraints.
    
    # Inequality constraints:
    # For each p, l:
    # z_pl - x_p <= 0
    # z_pl - y_l <= 0
    # -z_pl + x_p + y_l <= 1
    
    n_vars = n_p + n_l + n_p * n_l
    n_ineq = 3 * n_p * n_l
    
    # We will build the sparse matrix for inequalities
    rows = []
    cols = []
    data = []
    b_ub = []
    
    idx = 0
    for p in range(n_p):
        x_idx = p
        for l in range(n_l):
            y_idx = n_p + l
            z_idx = n_p + n_l + p * n_l + l
            
            # 1. z - x <= 0
            rows.append(idx); cols.append(z_idx); data.append(1.0)
            rows.append(idx); cols.append(x_idx); data.append(-1.0)
            b_ub.append(0.0)
            
            # 2. z - y <= 0
            rows.append(idx + 1); cols.append(z_idx); data.append(1.0)
            rows.append(idx + 1); cols.append(y_idx); data.append(-1.0)
            b_ub.append(0.0)
            
            # 3. -z + x + y <= 1
            rows.append(idx + 2); cols.append(z_idx); data.append(-1.0)
            rows.append(idx + 2); cols.append(x_idx); data.append(1.0)
            rows.append(idx + 2); cols.append(y_idx); data.append(1.0)
            b_ub.append(1.0)
            
            idx += 3
            
    A_ineq = sp.csc_matrix((data, (rows, cols)), shape=(n_ineq, n_vars))
    
    # Equality constraints:
    # sum x = 54
    # sum y = 27
    A_eq = np.zeros((2, n_vars))
    A_eq[0, :n_p] = 1.0
    A_eq[1, n_p:n_p+n_l] = 1.0
    b_eq = np.array([54.0, 27.0])
    
    # Combine into LinearConstraint
    # Lower bound -inf, upper bound b_ub for inequalities
    # Lower bound b_eq, upper bound b_eq for equalities
    
    # milp expects:
    # constraints: LinearConstraint(A, lb, ub)
    # We can pass a list of constraints or a single matrix.
    # Let's combine them.
    
    # A_total shape: (n_ineq + 2, n_vars)
    A_total = sp.vstack([A_ineq, sp.csc_matrix(A_eq)], format='csc')
    lb_total = np.concatenate([np.full(n_ineq, -np.inf), b_eq])
    ub_total = np.concatenate([np.array(b_ub), b_eq])
    
    constraints = LinearConstraint(A_total, lb_total, ub_total)
    
    # Bounds
    bounds = Bounds(lb=0, ub=1)
    
    # Solve
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status != 0:
            return None, None, 0
            
        x_sol = res.x
        x_points = x_sol[:n_p]
        y_lines = x_sol[n_p:n_p+n_l]
        
        # Threshold to binary
        sel_points = x_points > 0.5
        sel_lines = y_lines > 0.5
        
        # Ensure exactly 54 and 27
        # If solver is slightly off, fix it
        if np.sum(sel_points) != 54:
            # Pick top 54 based on incidence with selected lines
            # This is a fallback
            M_sel = M[:, sel_lines]
            scores = np.sum(M_sel, axis=1)
            top_idx = np.argsort(scores)[::-1][:54]
            sel_points = np.zeros(n_p, dtype=bool)
            sel_points[top_idx] = True
            
        if np.sum(sel_lines) != 27:
            M_sel = M[sel_points, :]
            scores = np.sum(M_sel, axis=0)
            top_idx = np.argsort(scores)[::-1][:27]
            sel_lines = np.zeros(n_l, dtype=bool)
            sel_lines[top_idx] = True
            
        best_points = candidate_points[sel_points]
        best_lines = candidate_lines[sel_lines]
        
        I = count_incidences(best_points, best_lines)
        
        return best_points, best_lines, I
        
    except Exception as e:
        return None, None, 0

def szemeredi_trotter_incidences():
    deadline = time.time() + 600
    best_I_global = 0
    best_points_global = None
    best_lines_global = None
    
    # Step 1: Generate candidate pool
    candidate_points, candidate_lines = generate_candidate_pool()
    
    # Step 2: Solve ILP
    # We might run multiple ILPs with different pools or just one big one
    # Given time constraints, one ILP with a good pool is best.
    
    start_time = time.time()
    points, lines, I = solve_ilp(candidate_points, candidate_lines, time_limit=deadline - start_time - 30)
    
    if points is not None and I > best_I_global:
        best_I_global = I
        best_points_global = points
        best_lines_global = lines
    else:
        # Fallback: Greedy approach if ILP fails or times out
        # Just pick lines and best points
        random.seed(42)
        idx = np.random.choice(len(candidate_lines), 27, replace=False)
        lines = candidate_lines[idx]
        px = candidate_points[:, 0].reshape(-1, 1)
        py = candidate_points[:, 1].reshape(-1, 1)
        A = lines[:, 0].reshape(1, -1)
        B = lines[:, 1].reshape(1, -1)
        C = lines[:, 2].reshape(1, -1)
        vals = A * px + B * py + C
        counts = np.sum(vals == 0, axis=1)
        top_idx = np.argsort(counts)[::-1][:54]
        points = candidate_points[top_idx]
        I = count_incidences(points, lines)
        best_I_global = I
        best_points_global = points
        best_lines_global = lines

    # Final Safety Check
    if best_points_global is None or best_lines_global is None:
        # Fallback to a simple grid
        lines = np.array([(a, -1, b) for a in range(1, 5) for b in range(1, 8)][:27], dtype=np.int64)
        # Need points for these lines
        # Generate points on these lines
        pts = set()
        for l in lines:
            A, B, C = l
            if A != 0:
                x_range = range(-20, 21)
                for x in x_range:
                    val = -(A*x + C)
                    if B != 0 and val % B == 0:
                        y = val // B
                        pts.add((x, y))
            elif B != 0:
                y_range = range(-20, 21)
                for y in y_range:
                    val = -(B*y + C)
                    if A != 0 and val % A == 0:
                        x = val // A
                        pts.add((x, y))
        pts_list = list(pts)
        if len(pts_list) < 54:
            # Pad with grid points
            for x in range(-50, 51):
                for y in range(-50, 51):
                    if (x, y) not in pts:
                        pts_list.append((x, y))
                        pts.add((x, y))
                        if len(pts_list) >= 54:
                            break
                if len(pts_list) >= 54:
                    break
        best_points_global = np.array(pts_list[:54], dtype=np.int64)
        best_lines_global = lines

    # Validate Points
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        # Fallback
        lines = np.array([(a, -1, b) for a in range(1, 5) for b in range(1, 8)][:27], dtype=np.int64)
        pts = set()
        for l in lines:
            A, B, C = l
            if A != 0:
                for x in range(-50, 51):
                    val = -(A*x + C)
                    if B != 0 and val % B == 0:
                        y = val // B
                        pts.add((x, y))
            elif B != 0:
                for y in range(-50, 51):
                    val = -(B*y + C)
                    if A != 0 and val % A == 0:
                        x = val // A
                        pts.add((x, y))
        pts_list = list(pts)
        if len(pts_list) < 54:
             for x in range(-100, 101):
                 for y in range(-100, 101):
                     if (x, y) not in pts:
                         pts_list.append((x, y))
                         pts.add((x, y))
                         if len(pts_list) >= 54:
                             break
                 if len(pts_list) >= 54:
                     break
        best_points_global = np.array(pts_list[:54], dtype=np.int64)
        best_lines_global = lines
    else:
        best_points_global = unique_points[indices]
        
    # Validate Lines
    unique_lines = []
    for line in best_lines_global:
        n_line = normalize_line(line[0], line[1], line[2])
        if n_line and n_line not in unique_lines:
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        # Fallback
        lines = np.array([(a, -1, b) for a in range(1, 5) for b in range(1, 8)][:27], dtype=np.int64)
        pts = set()
        for l in lines:
            A, B, C = l
            if A != 0:
                for x in range(-50, 51):
                    val = -(A*x + C)
                    if B != 0 and val % B == 0:
                        y = val // B
                        pts.add((x, y))
            elif B != 0:
                for y in range(-50, 51):
                    val = -(B*y + C)
                    if A != 0 and val % A == 0:
                        x = val // A
                        pts.add((x, y))
        pts_list = list(pts)
        if len(pts_list) < 54:
             for x in range(-100, 101):
                 for y in range(-100, 101):
                     if (x, y) not in pts:
                         pts_list.append((x, y))
                         pts.add((x, y))
                         if len(pts_list) >= 54:
                             break
                 if len(pts_list) >= 54:
                     break
        best_points_global = np.array(pts_list[:54], dtype=np.int64)
        best_lines_global = lines
    else:
        best_lines_global = np.array(unique_lines[:27], dtype=np.int64)

    return best_points_global, best_lines_global
