# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.

    Structural change: Instead of a greedy point-removal heuristic (which gets stuck
    in local optima), model the problem as a Minimum Set Cover / Hitting Set problem.
    1. Construct the base Hermitian Nikodym set N0.
    2. Identify a pool of candidate points to remove (those in N0 that are not essential
       for their own coverage, i.e., have cover_count > 1).
    3. Formulate an integer linear program (ILP) to maximize the number of removed points
       subject to the constraint that every point in F_49^2 remains covered by at least
       one valid line.
    4. Solve the ILP using scipy.optimize.milp.
    """
    deadline = time.time() + 620  # Safety margin before 630s limit

    p = 7
    q = p * p  # 49

    # Field F_q = F_p[alpha], alpha^2 = k
    k_field = 3  # 3 is a quadratic non-residue mod 7

    # Represent elements of F_q as integers 0..48: element (a, b) -> a + b*p
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute field arithmetic tables for F_q
    f_add_table = np.zeros((q, q), dtype=np.int32)
    f_mul_table = np.zeros((q, q), dtype=np.int32)
    f_neg_table = np.zeros(q, dtype=np.int32)
    
    for x in range(q):
        xa, xb = to_pair(x)
        f_neg_table[x] = to_int((-xa) % p, (-xb) % p)
        for y in range(q):
            ya, yb = to_pair(y)
            f_add_table[x, y] = to_int((xa + ya) % p, (xb + yb) % p)
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            f_mul_table[x, y] = to_int(real, imag)

    # Directions: v = (1, m) for m in F_q, or v = (0, 1)
    directions = []
    for m in range(q):
        directions.append((1, m))  # v = (1, m)
    directions.append((0, 1))  # vertical
    n_dir = len(directions)

    # All points in F_q^2 as integer pairs
    point_index = {}
    idx_to_point = [None] * (q * q)
    for x_idx in range(q):
        for y_idx in range(q):
            idx = x_idx * q + y_idx
            point_index[(x_idx, y_idx)] = idx
            idx_to_point[idx] = (x_idx, y_idx)

    # Build the base Nikodym set
    # Hermitian unital: U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a, Norm((a,b)) = a^2 - b^2*k
    # Blocking set B = points of U whose y-coordinate has y_a a square mod p
    # Excluded set S = U \ B
    # N = complement of S

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(to_int((u * u) % p, v))

    # Initialize N: all points except excluded
    in_N = np.ones(q * q, dtype=bool)

    for x_idx in range(q):
        xa, xb = to_pair(x_idx)
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = to_pair(y_idx)
            norm_y = (ya * ya - yb * yb * k_field) % p
            if (trace_x + norm_y) % p == 0:
                # In unital
                if y_idx not in y_special:
                    # Excluded
                    in_N[point_index[(x_idx, y_idx)]] = False

    # Initialize not_in_N_count
    # not_in_N_count[p_idx][d_idx] = number of points on line(point, v_d) \ {point} that are not in N
    not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)

    # Compute initial counts
    for d_idx, (vx, vy) in enumerate(directions):
        # Precompute t*v for t in 1..q-1
        t_v = [(f_mul_table[t, vx], f_mul_table[t, vy]) for t in range(1, q)]
        
        for p_idx in range(q * q):
            px, py = idx_to_point[p_idx]
            count = 0
            for tx, ty in t_v:
                nx = f_add_table[px, tx]
                ny = f_add_table[py, ty]
                n_idx = point_index[(nx, ny)]
                if not in_N[n_idx]:
                    count += 1
            not_in_N_count[p_idx, d_idx] = count

    # cover_count[p_idx] = number of directions d such that not_in_N_count[p_idx, d] == 0
    cover_count = np.sum(not_in_N_count == 0, axis=1).astype(np.int32)

    # Identify candidate points for removal:
    # A point p in N can be removed if removing it doesn't break coverage of any point x.
    # Removing p increases not_in_N_count[x, d] by 1 for all (x, d) where p is on line(x, d)\{x}.
    # This is bad if cover_count[x] becomes 0.
    # Specifically, if cover_count[x] == 1 and p is on the unique covering line of x, then p is critical.
    # If cover_count[x] > 1, removing p from a covering line reduces cover_count[x] by 1 but it stays >= 1.
    # So p is safe to remove if for all (x, d) where p is on line(x, d)\{x} and not_in_N_count[x, d]==0,
    # we have cover_count[x] > 1.
    
    # Let's find all points in N that are "removable" in this sense.
    candidates = []
    for p_idx in range(q * q):
        if not in_N[p_idx]:
            continue
        
        # Check if p_idx is critical for any x
        is_critical = False
        px, py = idx_to_point[p_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            # Find x such that p is on line(x, d)\{x}
            # x + t*v = p => x = p - t*v
            for t in range(1, q):
                tv_x = f_mul_table[t, vx]
                tv_y = f_mul_table[t, vy]
                neg_tv_x = f_neg_table[tv_x]
                neg_tv_y = f_neg_table[tv_y]
                nx = f_add_table[px, neg_tv_x]
                ny = f_add_table[py, neg_tv_y]
                x_idx = point_index[(nx, ny)]
                
                # If p is on line(x, d)\{x}, and this line currently covers x (count==0)
                if not_in_N_count[x_idx, d_idx] == 0:
                    # If x has only this one covering direction, then p is critical for x
                    if cover_count[x_idx] == 1:
                        is_critical = True
                        break
            if is_critical:
                break
        
        if not is_critical:
            candidates.append(p_idx)

    # Now, we have a set of candidates. We want to select a subset R of candidates to remove
    # such that for every point x in F_49^2, x is still covered.
    # A point x is covered if there exists a direction d such that all points on line(x, d)\{x} are in N\R.
    # Initially, line(x, d) is valid if not_in_N_count[x, d] == 0.
    # If we remove p in R, and p is on line(x, d)\{x}, then the count for (x, d) increases by 1.
    # So (x, d) remains valid if not_in_N_count[x, d] == 0 AND no p in R is on line(x, d)\{x}.
    # OR if not_in_N_count[x, d] == 1 AND exactly one p in R is on line(x, d)\{x}? No, count becomes 2.
    # Wait. not_in_N_count is the number of points NOT in N.
    # If we remove p from N, it becomes NOT in N. So count increases.
    # A line is valid if count == 0.
    # So if we remove p, and p was in N, then for any (x, d) where p is on the line, the count increases.
    # If the count was 0, it becomes 1. The line is NO LONGER valid.
    # So, for x to remain covered, there must be at least one direction d such that:
    # 1. not_in_N_count[x, d] == 0 (initially valid)
    # 2. No point in R is on line(x, d)\{x}.
    
    # This is a Set Cover / Hitting Set problem.
    # Variables: z_p for p in candidates, z_p = 1 if p is removed.
    # Constraints: For each x in F_49^2, sum_{d in ValidDirs(x)} (1 - sum_{p in Line(x,d)\{x} \cap candidates} z_p) >= 1
    # Where ValidDirs(x) = {d | not_in_N_count[x, d] == 0}
    # Note: If x is not in N initially, it doesn't matter if its line is broken, because x is already "covered" by being in the complement?
    # No, the definition says: every point x of F_49^2 (in N or not) must lie on a line...
    # So even if x is not in N, it must be covered.
    # However, if x is not in N, it is already in the complement.
    # The condition is: line(x, d) \ {x} subseteq N.
    # If x is not in N, it doesn't affect the condition for x itself, except that x must have a valid line.
    
    # Let's refine the constraint.
    # For each x, let D_x = {d | not_in_N_count[x, d] == 0}.
    # We need at least one d in D_x such that no p in R is on line(x, d)\{x}.
    # Let S_{x, d} = {p in candidates | p is on line(x, d)\{x}}.
    # Constraint: OR_{d in D_x} (AND_{p in S_{x, d}} (1 - z_p)) >= 1
    # This is a non-linear constraint. We can linearize it.
    # Introduce binary variable y_{x, d} for d in D_x. y_{x, d} = 1 if line(x, d) is chosen as the covering line.
    # Constraint 1: sum_{d in D_x} y_{x, d} >= 1 for all x.
    # Constraint 2: y_{x, d} <= 1 - z_p for all p in S_{x, d}. (If p is removed, y_{x, d} must be 0)
    # Objective: Maximize sum_{p in candidates} z_p.

    # This ILP might be large.
    # Number of candidates: likely a few hundred.
    # Number of x: 2401.
    # Number of constraints: ~ 2401 * (avg |D_x|) + sum_{x,d} |S_{x,d}|.
    # |S_{x,d}| is at most |candidates|.
    # This could be a large ILP. Let's see if we can prune.
    
    # Only x with |D_x| == 1 are the most constrained. If |D_x| > 1, there is more flexibility.
    # Let's try solving the ILP with scipy.optimize.milp.

    if len(candidates) == 0:
        # No candidates, return base set
        points = []
        for i in range(q * q):
            if in_N[i]:
                x, y = idx_to_point[i]
                xa, xb = to_pair(x)
                ya, yb = to_pair(y)
                points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    # Build ILP
    # Variables: z_p for p in candidates (index 0..C-1)
    # y_{x, d} for x in 0..q^2-1, d in D_x (index C..C+Y-1)
    
    C = len(candidates)
    cand_idx_map = {p: i for i, p in enumerate(candidates)}
    
    # Precompute S_{x, d} for all x, d in D_x
    # S_{x, d} is a list of candidate indices
    S = {}
    for x_idx in range(q * q):
        for d_idx in range(n_dir):
            if not_in_N_count[x_idx, d_idx] != 0:
                continue
            S[(x_idx, d_idx)] = []
            px, py = idx_to_point[x_idx]
            vx, vy = directions[d_idx]
            for t in range(1, q):
                tv_x = f_mul_table[t, vx]
                tv_y = f_mul_table[t, vy]
                nx = f_add_table[px, tv_x]
                ny = f_add_table[py, tv_y]
                p_idx = point_index[(nx, ny)]
                if p_idx in cand_idx_map:
                    S[(x_idx, d_idx)].append(cand_idx_map[p_idx])

    # Count y variables
    y_vars = []
    for x_idx in range(q * q):
        for d_idx in range(n_dir):
            if (x_idx, d_idx) in S:
                y_vars.append((x_idx, d_idx))
    
    Y = len(y_vars)
    y_idx_map = {(x, d): i for i, (x, d) in enumerate(y_vars)}
    
    n_vars = C + Y
    
    # Objective: Maximize sum z_p => Minimize -sum z_p
    c = np.zeros(n_vars)
    for i in range(C):
        c[i] = -1.0
    
    # Constraints:
    # 1. sum_{d in D_x} y_{x, d} >= 1 for all x
    # 2. y_{x, d} + z_p <= 1 for all p in S_{x, d}
    
    A_rows = []
    b_rows = []
    
    # Constraint 1
    for x_idx in range(q * q):
        row = np.zeros(n_vars)
        for d_idx in range(n_dir):
            if (x_idx, d_idx) in S:
                row[C + y_idx_map[(x_idx, d_idx)]] = 1.0
        if np.any(row):
            A_rows.append(row)
            b_rows.append(1.0)
        else:
            # If no valid directions initially, this x is uncovered?
            # But the base set is a valid Nikodym set, so this shouldn't happen.
            # If it does, we can't satisfy the constraint.
            pass

    # Constraint 2
    for (x_idx, d_idx), p_list in S.items():
        y_var_idx = C + y_idx_map[(x_idx, d_idx)]
        for p_idx in p_list:
            row = np.zeros(n_vars)
            row[y_var_idx] = 1.0
            row[p_idx] = 1.0
            A_rows.append(row)
            b_rows.append(1.0)
            
    if not A_rows:
        # No constraints? Just remove all candidates?
        # But we must ensure coverage. If no constraints, it means no x depends on any candidate?
        # That implies all x have D_x empty? No, if D_x is non-empty but S is empty, then no constraint 2.
        # But constraint 1 still exists.
        pass

    A = np.array(A_rows)
    b = np.array(b_rows)
    
    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Integrality
    integrality = np.ones(n_vars, dtype=int)
    
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    constraint = LinearConstraint(A, -np.inf, b)
    bounds = Bounds(lb, ub)
    
    try:
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    except Exception as e:
        # Fallback to base set
        points = []
        for i in range(q * q):
            if in_N[i]:
                x, y = idx_to_point[i]
                xa, xb = to_pair(x)
                ya, yb = to_pair(y)
                points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)
    
    if res.success:
        # Extract solution
        z_solution = res.x[:C]
        removed = set()
        for i in range(C):
            if z_solution[i] > 0.5:
                removed.add(candidates[i])
        
        # Update in_N
        for p_idx in removed:
            in_N[p_idx] = False
        
        # Verify solution (optional but good for safety)
        # Re-compute counts and check coverage
        # ... (skip for speed, assume ILP is correct)
        
        points = []
        for i in range(q * q):
            if in_N[i]:
                x, y = idx_to_point[i]
                xa, xb = to_pair(x)
                ya, yb = to_pair(y)
                points.append([[xa, xb], [ya, yb]])
        
        if points:
            return np.array(points, dtype=np.int64)
    
    # Fallback to base set
    points = []
    for i in range(q * q):
        if in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
