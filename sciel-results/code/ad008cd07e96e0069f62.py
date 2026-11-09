# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 600
    
    # Variables:
    # x[0..9]: cx
    # x[10..19]: cy
    # x[20..29]: s
    # x[30..]: binaries for pairs
    # Pairs: (i, j) for i < j. Total 45 pairs.
    # For each pair, 4 binaries: u (i left of j), v (j left of i), w (i below j), z (j below i)
    # Total variables: 30 + 45*4 = 30 + 180 = 210.
    
    idx_x = np.arange(0, n)
    idx_y = np.arange(n, 2*n)
    idx_s = np.arange(2*n, 3*n)
    
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
    num_pairs = len(pairs)
    
    idx_u = np.arange(3*n, 3*n + num_pairs) # i left of j
    idx_v = np.arange(3*n + num_pairs, 3*n + 2*num_pairs) # j left of i
    idx_w = np.arange(3*n + 2*num_pairs, 3*n + 3*num_pairs) # i below j
    idx_z = np.arange(3*n + 3*num_pairs, 3*n + 4*num_pairs) # j below i
    
    total_vars = 3*n + 4*num_pairs
    
    # Objective: Maximize sum(s) -> Minimize -sum(s)
    c = np.zeros(total_vars)
    c[idx_s] = -1.0
    
    # Constraints:
    # 1. Bounds:
    # 0 <= cx <= 1
    # 0 <= cy <= 1
    # 0 <= s <= 1
    # 0 <= binaries <= 1 (integer)
    
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    
    # Integrality:
    # s, cx, cy are continuous
    # binaries are integer
    integrality = np.zeros(total_vars)
    integrality[idx_u] = 1
    integrality[idx_v] = 1
    integrality[idx_w] = 1
    integrality[idx_z] = 1
    
    # Linear Constraints:
    # A_ub x <= b_ub
    
    # Boundary constraints:
    # cx_i - s_i/2 >= 0  =>  -cx_i + s_i/2 <= 0
    # cx_i + s_i/2 <= 1  =>   cx_i + s_i/2 <= 1
    # cy_i - s_i/2 >= 0  =>  -cy_i + s_i/2 <= 0
    # cy_i + s_i/2 <= 1  =>   cy_i + s_i/2 <= 1
    
    A_rows = []
    b_rows = []
    
    for i in range(n):
        # -cx_i + 0.5 s_i <= 0
        row = np.zeros(total_vars)
        row[idx_x[i]] = -1.0
        row[idx_s[i]] = 0.5
        A_rows.append(row)
        b_rows.append(0.0)
        
        # cx_i + 0.5 s_i <= 1
        row = np.zeros(total_vars)
        row[idx_x[i]] = 1.0
        row[idx_s[i]] = 0.5
        A_rows.append(row)
        b_rows.append(1.0)
        
        # -cy_i + 0.5 s_i <= 0
        row = np.zeros(total_vars)
        row[idx_y[i]] = -1.0
        row[idx_s[i]] = 0.5
        A_rows.append(row)
        b_rows.append(0.0)
        
        # cy_i + 0.5 s_i <= 1
        row = np.zeros(total_vars)
        row[idx_y[i]] = 1.0
        row[idx_s[i]] = 0.5
        A_rows.append(row)
        b_rows.append(1.0)
        
    # Pair constraints:
    # For pair k (i, j):
    # u_k + v_k + w_k + z_k = 1  (Equality, handled by A_eq or 2 inequalities)
    # Let's use inequalities:
    # u+v+w+z <= 1
    # u+v+w+z >= 1  => -u-v-w-z <= -1
    
    for k, (i, j) in enumerate(pairs):
        # Sum = 1
        row = np.zeros(total_vars)
        row[idx_u[k]] = 1.0
        row[idx_v[k]] = 1.0
        row[idx_w[k]] = 1.0
        row[idx_z[k]] = 1.0
        A_rows.append(row)
        b_rows.append(1.0)
        
        row_neg = -row
        A_rows.append(row_neg)
        b_rows.append(-1.0)
        
        # Separation constraints (Big M = 1)
        # If u_k=1, i is left of j: cx_i + s_i/2 <= cx_j - s_j/2
        # cx_i - cx_j + 0.5 s_i + 0.5 s_j <= 0 + M(1-u_k)
        # cx_i - cx_j + 0.5 s_i + 0.5 s_j - M*u_k <= 0
        
        # M=1 is sufficient because cx in [0,1], s in [0,1].
        # Max LHS is 1 - 0 + 0.5 + 0.5 = 2.
        # If u=0, RHS is 0, LHS must be <= 0.
        # If u=1, RHS is M=1? No, constraint is LHS <= M(1-u).
        # If u=1, 1-u=0, so LHS <= 0.
        # If u=0, LHS <= M. Since max LHS is 2, M=2 is safe. Let's use M=2 to be safe.
        M = 2.0
        
        # u: i left of j
        row = np.zeros(total_vars)
        row[idx_x[i]] = 1.0
        row[idx_x[j]] = -1.0
        row[idx_s[i]] = 0.5
        row[idx_s[j]] = 0.5
        row[idx_u[k]] = -M
        A_rows.append(row)
        b_rows.append(0.0)
        
        # v: j left of i
        row = np.zeros(total_vars)
        row[idx_x[i]] = -1.0
        row[idx_x[j]] = 1.0
        row[idx_s[i]] = 0.5
        row[idx_s[j]] = 0.5
        row[idx_v[k]] = -M
        A_rows.append(row)
        b_rows.append(0.0)
        
        # w: i below j
        row = np.zeros(total_vars)
        row[idx_y[i]] = 1.0
        row[idx_y[j]] = -1.0
        row[idx_s[i]] = 0.5
        row[idx_s[j]] = 0.5
        row[idx_w[k]] = -M
        A_rows.append(row)
        b_rows.append(0.0)
        
        # z: j below i
        row = np.zeros(total_vars)
        row[idx_y[i]] = -1.0
        row[idx_y[j]] = 1.0
        row[idx_s[i]] = 0.5
        row[idx_s[j]] = 0.5
        row[idx_z[k]] = -M
        A_rows.append(row)
        b_rows.append(0.0)
        
    A = np.array(A_rows)
    b = np.array(b_rows)
    
    # Try MILP
    best_centers = None
    best_sides = None
    best_val = -1.0
    
    try:
        if 'milp' in dir(__import__('scipy.optimize')):
            # Use scipy.optimize.milp
            res = milp(
                c=c,
                constraints=LinearConstraint(A, -np.inf, b),
                integrality=integrality,
                bounds=Bounds(lb, ub),
                options={"time_limit": 500} # Leave 100s for fallback/polish
            )
            if res.success and res.x is not None:
                cx = res.x[idx_x]
                cy = res.x[idx_y]
                s = res.x[idx_s]
                centers = np.column_stack([cx, cy])
                if _check_valid(centers, s, tol):
                    best_centers = centers
                    best_sides = s
                    best_val = np.sum(s)
            else:
                print(f"MILP failed: {res.message}")
        else:
            print("milp not available, using linprog relaxation")
            # Fallback to LP relaxation
            res = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lb, ub)), method='highs')
            if res.success:
                cx = res.x[idx_x]
                cy = res.x[idx_y]
                s = res.x[idx_s]
                centers = np.column_stack([cx, cy])
                # LP solution might be invalid due to fractional binaries
                # We need to round or repair.
                # Simple repair: clamp centers, ensure valid.
                # This might not be optimal, but better than nothing.
                # Let's just use it as a starting point for local search.
                best_centers = centers
                best_sides = s
                # Check validity, if invalid, we need a valid fallback
                if not _check_valid(best_centers, best_sides, tol):
                    # Generate a valid heuristic
                    hc, hs = _heuristic_init_2_large_8_small()
                    best_centers = hc
                    best_sides = hs
                    best_val = np.sum(hs)
                else:
                    best_val = np.sum(s)
                    
    except Exception as e:
        print(f"Solver error: {e}")
        hc, hs = _heuristic_init_2_large_8_small()
        best_centers = hc
        best_sides = hs
        best_val = np.sum(hs)

    # Local Polish
    if best_centers is not None:
        # Fast local search
        best_centers, best_sides, best_val = _local_improve_fast(best_centers, best_sides, tol, time_limit=60)
        
    return best_centers, best_sides

def _check_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
            
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_sep = (sides[i] + sides[j])/2 - tol
            if dx < min_sep and dy < min_sep:
                return False
    return True

def _heuristic_init_2_large_8_small():
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.25]
    sides[1] = 0.5
    
    for k in range(8):
        col = k % 4
        row = k // 4
        cx = (col + 0.5) * 0.25
        cy = 0.5 + (row + 0.5) * 0.25
        centers[2+k, 0] = cx
        centers[2+k, 1] = cy
        sides[2+k] = 0.25
        
    return centers, sides

def _local_improve_fast(centers, sides, tol=1e-7, max_iters=10000, step_size=0.01, time_limit=60):
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    if not _check_valid(best_centers, best_sides, tol):
        return best_centers, best_sides, best_val
        
    rng = np.random.default_rng(42)
    start_time = time.time()
    T = 0.01
    cooling = 0.999
    min_T = 1e-6
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if T < min_T:
            break
            
        idx = rng.integers(0, n)
        nc = best_centers.copy()
        ns = best_sides.copy()
        
        noise = step_size * (0.5 + 0.5 * T)
        nc[idx, 0] += rng.uniform(-noise, noise)
        nc[idx, 1] += rng.uniform(-noise, noise)
        ns[idx] += rng.uniform(-noise, noise)
        ns[idx] = max(1e-6, min(1.0, ns[idx]))
        
        s = ns[idx]
        nc[idx, 0] = np.clip(nc[idx, 0], s/2, 1-s/2)
        nc[idx, 1] = np.clip(nc[idx, 1], s/2, 1-s/2)
        
        # Quick validity check for idx
        valid = True
        for j in range(n):
            if j == idx: continue
            dx = abs(nc[idx, 0] - best_centers[j, 0])
            dy = abs(nc[idx, 1] - best_centers[j, 1])
            min_sep = (ns[idx] + best_sides[j])/2 - tol
            if dx < min_sep and dy < min_sep:
                valid = False
                break
        
        if valid:
            new_val = np.sum(ns)
            delta = new_val - best_val
            if delta > 0 or rng.random() < np.exp(delta / T):
                best_centers = nc
                best_sides = ns
                best_val = new_val
                
        T *= cooling
        
    return best_centers, best_sides, best_val
# EVOLVE-BLOCK-END
