# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 610

    def elapsed():
        return time.time() - start_time

    def is_valid(centers, sides, tol=1e-7):
        n = len(sides)
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
                t = (sides[i] + sides[j]) / 2.0 - tol
                if dx < t and dy < t:
                    return False
        return True

    best_centers = None
    best_sides = None
    best_score = -np.inf

    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if not is_valid(centers, sides):
            return
        s = np.sum(sides)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
            best_sides = sides.copy()

    # ============================================================
    # LP Solver for fixed topologies
    # ============================================================
    def solve_lp_topology(orderings):
        """
        orderings: list of tuples (i, j, axis, direction)
        axis: 0 for x, 1 for y
        direction: 1 means j is 'after' i in that axis (e.g., cx_j >= cx_i + (s_i+s_j)/2)
                   -1 means i is 'after' j in that axis
        """
        # Variables: cx_0..cx_9, cy_0..cy_9, s_0..s_9
        # Index: 0..9 for cx, 10..19 for cy, 20..29 for s
        n_vars = 3 * n
        
        # Objective: maximize sum(s) -> minimize -sum(s)
        c = np.zeros(n_vars)
        c[20:30] = -1.0
        
        # Inequality constraints: A_ub @ x <= b_ub
        A_ub = []
        b_ub = []
        
        # Boundary constraints
        # cx_i - s_i/2 >= 0  =>  -cx_i + 0.5*s_i <= 0
        # cx_i + s_i/2 <= 1  =>   cx_i - 0.5*s_i <= 1
        # cy_i - s_i/2 >= 0  =>  -cy_i + 0.5*s_i <= 0
        # cy_i + s_i/2 <= 1  =>   cy_i - 0.5*s_i <= 1
        for i in range(n):
            # cx_i - s_i/2 >= 0
            row = np.zeros(n_vars)
            row[i] = -1.0
            row[20+i] = 0.5
            A_ub.append(row)
            b_ub.append(0.0)
            
            # cx_i + s_i/2 <= 1
            row = np.zeros(n_vars)
            row[i] = 1.0
            row[20+i] = -0.5
            A_ub.append(row)
            b_ub.append(1.0)
            
            # cy_i - s_i/2 >= 0
            row = np.zeros(n_vars)
            row[10+i] = -1.0
            row[20+i] = 0.5
            A_ub.append(row)
            b_ub.append(0.0)
            
            # cy_i + s_i/2 <= 1
            row = np.zeros(n_vars)
            row[10+i] = 1.0
            row[20+i] = -0.5
            A_ub.append(row)
            b_ub.append(1.0)
            
            # s_i > 0 (handled by bounds, but linprog handles bounds separately)
            
        # Topology constraints
        # For each (i, j, axis, dir):
        # If axis=0 (x), dir=1: cx_j - cx_i >= (s_i + s_j)/2
        #   => -cx_j + cx_i + 0.5*s_i + 0.5*s_j <= 0
        # If axis=0 (x), dir=-1: cx_i - cx_j >= (s_i + s_j)/2
        #   => cx_j - cx_i + 0.5*s_i + 0.5*s_j <= 0
        # Similarly for y.
        
        for i, j, axis, dir in orderings:
            row = np.zeros(n_vars)
            # s_i and s_j terms are always positive 0.5
            row[20+i] = 0.5
            row[20+j] = 0.5
            
            if axis == 0: # x-axis
                if dir == 1: # j is right of i: cx_j - cx_i >= ... => -cx_j + cx_i ... <= 0
                    row[i] = 1.0
                    row[j] = -1.0
                else: # i is right of j: cx_i - cx_j >= ... => cx_j - cx_i ... <= 0
                    row[i] = -1.0
                    row[j] = 1.0
            else: # y-axis
                if dir == 1: # j is above i: cy_j - cy_i >= ... => -cy_j + cy_i ... <= 0
                    row[10+i] = 1.0
                    row[10+j] = -1.0
                else: # i is above j: cy_i - cy_j >= ... => cy_j - cy_i ... <= 0
                    row[10+i] = -1.0
                    row[10+j] = 1.0
            
            A_ub.append(row)
            b_ub.append(0.0)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds
        # cx, cy in [0, 1]
        # s in [0, 1] (loose upper bound, LP will optimize)
        bounds = []
        for i in range(n):
            bounds.append((0, 1)) # cx
        for i in range(n):
            bounds.append((0, 1)) # cy
        for i in range(n):
            bounds.append((1e-9, 1)) # s > 0
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx = res.x[0:n]
                cy = res.x[n:2*n]
                s = res.x[2*n:3*n]
                centers = np.column_stack((cx, cy))
                return centers, s
            else:
                return None, None
        except Exception:
            return None, None

    def extract_orderings(centers, sides):
        """
        Extracts a set of ordering constraints from a valid configuration.
        For each pair, it picks the separation axis and direction that is 'tightest' (smallest slack).
        """
        orderings = []
        n_pairs = n*(n-1)//2
        for i in range(n):
            for j in range(i+1, n):
                # Calculate slacks for the 4 possible separations
                # 0: x, j right of i
                # 1: x, i right of j
                # 2: y, j above i
                # 3: y, i above j
                
                # Required distance
                d = (sides[i] + sides[j]) / 2.0
                
                # Current distances
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                
                # Slacks (how much they exceed the requirement)
                # If dx >= d, then j is right of i with slack dx-d
                # If -dx >= d, then i is right of j with slack -dx-d
                # If dy >= d, then j is above i with slack dy-d
                # If -dy >= d, then i is above i with slack -dy-d
                
                slack_x_j_right = dx - d
                slack_x_i_right = -dx - d
                slack_y_j_above = dy - d
                slack_y_i_above = -dy - d
                
                # We need at least one of these to be >= 0 for validity.
                # We pick the one with the smallest positive slack (tightest).
                # If multiple are >= 0, pick the min.
                # If none are >= 0, the config is invalid (shouldn't happen if we call this on valid config).
                
                candidates = []
                if slack_x_j_right >= 0:
                    candidates.append((slack_x_j_right, i, j, 0, 1))
                if slack_x_i_right >= 0:
                    candidates.append((slack_x_i_right, i, j, 0, -1))
                if slack_y_j_above >= 0:
                    candidates.append((slack_y_j_above, i, j, 1, 1))
                if slack_y_i_above >= 0:
                    candidates.append((slack_y_i_above, i, j, 1, -1))
                
                if not candidates:
                    # Invalid config, return empty or raise
                    return []
                
                # Pick the tightest one
                best_candidate = min(candidates, key=lambda x: x[0])
                orderings.append((best_candidate[1], best_candidate[2], best_candidate[3], best_candidate[4]))
                
        return orderings

    # ============================================================
    # Vectorized smooth penalty objective (kept from original)
    # ============================================================
    def smooth_penalty(params, beta=50.0):
        centers = params[:2*n].reshape(n, 2)
        sides = params[2*n:]

        def softplus(x, b):
            return np.log1p(np.exp(-b * x)) / b

        obj = -np.sum(sides)
        W = 100.0

        # Boundary penalties (vectorized)
        v_l = -(centers[:, 0] - sides/2)
        v_r = (centers[:, 0] + sides/2) - 1
        v_b = -(centers[:, 1] - sides/2)
        v_t = (centers[:, 1] + sides/2) - 1
        obj += W * (np.sum(softplus(v_l, beta)) + np.sum(softplus(v_r, beta)) +
                    np.sum(softplus(v_b, beta)) + np.sum(softplus(v_t, beta)))

        # Overlap penalties (vectorized over pairs)
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - 1e-7
                if t <= 0:
                    continue
                a = t - dx
                b = t - dy
                a_c = np.clip(a, -50/beta, 50/beta)
                b_c = np.clip(b, -50/beta, 50/beta)
                sm = -np.log(np.exp(-beta * a_c) + np.exp(-beta * b_c)) / beta
                obj += W * softplus(sm, beta)

        return obj

    # ============================================================
    # Build initial topologies (kept from original)
    # ============================================================
    starts = []

    # Topology 1: 2 large (0.5) top, 8 small (0.25) bottom 2x4
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    a, b = 0.5, 0.25
    c1[0] = [a/2, 1 - a/2]; s1[0] = a
    c1[1] = [1 - a/2, 1 - a/2]; s1[1] = a
    for r in range(2):
        for col in range(4):
            idx = 2 + r*4 + col
            c1[idx, 0] = b/2 + col*b
            c1[idx, 1] = b/2 + r*b
            s1[idx] = b
    update_best(c1, s1)
    starts.append((c1.copy(), s1.copy()))

    # Topology 2: 1 large (0.5) left, 9 in right 0.5x1 region
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    c2[0] = [0.25, 0.5]; s2[0] = 0.5
    b2 = 0.5/3
    for r in range(3):
        for col in range(3):
            idx = 1 + r*3 + col
            c2[idx, 0] = 0.5 + b2/2 + col*b2
            c2[idx, 1] = b2/2 + r*b2
            s2[idx] = b2
    update_best(c2, s2)
    starts.append((c2.copy(), s2.copy()))

    # Topology 3: 3 large (1/3) in a row at top, 7 small below
    c3 = np.zeros((n, 2))
    s3 = np.zeros(n)
    a3 = 1.0/3
    for i in range(3):
        c3[i, 0] = a3/2 + i*a3
        c3[i, 1] = 1 - a3/2
        s3[i] = a3
    b3a = 0.25
    b3b = 1.0/3
    for col in range(4):
        idx = 3 + col
        c3[idx, 0] = b3a/2 + col*b3a
        c3[idx, 1] = (2.0/3) - b3a/2
        s3[idx] = b3a
    for col in range(3):
        idx = 7 + col
        c3[idx, 0] = b3b/2 + col*b3b
        c3[idx, 1] = b3b/2
        s3[idx] = b3b
    update_best(c3, s3)
    starts.append((c3.copy(), s3.copy()))

    # Topology 4: Uniform 2x5 grid
    c4 = np.zeros((n, 2))
    s4 = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c4[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s4[i] = 0.2
    update_best(c4, s4)
    starts.append((c4.copy(), s4.copy()))

    # Topology 5: 4 corners + 6 center
    c5 = np.zeros((n, 2))
    s5 = np.zeros(n)
    a5 = 0.3
    b5 = (1 - 2*a5)/3
    c5[0] = [a5/2, a5/2]; s5[0] = a5
    c5[1] = [1-a5/2, a5/2]; s5[1] = a5
    c5[2] = [a5/2, 1-a5/2]; s5[2] = a5
    c5[3] = [1-a5/2, 1-a5/2]; s5[3] = a5
    for r in range(2):
        for col in range(3):
            idx = 4 + r*3 + col
            c5[idx, 0] = a5 + b5/2 + col*b5
            c5[idx, 1] = a5 + b5/2 + r*b5
            s5[idx] = b5
    update_best(c5, s5)
    starts.append((c5.copy(), s5.copy()))

    # Topology 6: 5+5 split
    c6 = np.zeros((n, 2))
    s6 = np.zeros(n)
    for r in range(2):
        for col in range(2):
            idx = r*2 + col
            c6[idx, 0] = 0.125 + col*0.25
            c6[idx, 1] = 0.125 + r*0.25
            s6[idx] = 0.25
    c6[4, 0] = 0.25; c6[4, 1] = 0.75; s6[4] = 0.25
    for r in range(2):
        for col in range(2):
            idx = 5 + r*2 + col
            c6[idx, 0] = 0.625 + col*0.25
            c6[idx, 1] = 0.125 + r*0.25
            s6[idx] = 0.25
    c6[9, 0] = 0.75; c6[9, 1] = 0.75; s6[9] = 0.25
    update_best(c6, s6)
    starts.append((c6.copy(), s6.copy()))

    # Topology 7: 2x5 but with varying sizes - top row larger
    c7 = np.zeros((n, 2))
    s7 = np.zeros(n)
    a7 = 0.3
    for i in range(3):
        c7[i, 0] = a7/2 + i*a7
        c7[i, 1] = 1 - a7/2
        s7[i] = a7
    b7a = 0.2
    for col in range(4):
        idx = 3 + col
        c7[idx, 0] = b7a/2 + col*b7a
        c7[idx, 1] = 0.7 - b7a/2
        s7[idx] = b7a
    b7b = 1.0/3
    for col in range(3):
        idx = 7 + col
        c7[idx, 0] = b7b/2 + col*b7b
        c7[idx, 1] = b7b/2
        s7[idx] = b7b
    update_best(c7, s7)
    starts.append((c7.copy(), s7.copy()))

    # Topology 8: 4 large (0.4) in corners, 6 small in center
    c8 = np.zeros((n, 2))
    s8 = np.zeros(n)
    a8 = 0.4
    c8[0] = [a8/2, a8/2]; s8[0] = a8
    c8[1] = [1-a8/2, a8/2]; s8[1] = a8
    c8[2] = [a8/2, 1-a8/2]; s8[2] = a8
    c8[3] = [1-a8/2, 1-a8/2]; s8[3] = a8
    b8 = 0.2/3
    for r in range(2):
        for col in range(3):
            idx = 4 + r*3 + col
            c8[idx, 0] = a8 + b8/2 + col*b8
            c8[idx, 1] = a8 + b8/2 + r*b8
            s8[idx] = b8
    update_best(c8, s8)
    starts.append((c8.copy(), s8.copy()))

    # Add perturbed versions
    np.random.seed(42)
    for base_c, base_s in starts[:4]:
        for _ in range(4):
            c_p = base_c + np.random.normal(0, 0.02, (n, 2))
            s_p = base_s + np.random.normal(0, 0.01, n)
            s_p = np.clip(s_p, 0.05, 0.55)
            starts.append((c_p, s_p))

    # ============================================================
    # Optimization loop with adaptive beta (kept from original)
    # ============================================================
    for start_idx, (c_init, s_init) in enumerate(starts):
        if elapsed() > 400: # Reduced time to allow LP solving
            break

        params_init = np.concatenate([c_init.flatten(), s_init])

        # Multi-stage: coarse to fine
        for stage, (beta, maxiter, ftol) in enumerate([(30.0, 100, 1e-8),
                                                        (80.0, 100, 1e-10)]):
            if elapsed() > 500:
                break
            try:
                res = minimize(smooth_penalty, params_init, args=(beta,),
                               method='SLSQP',
                               options={'maxiter': maxiter, 'ftol': ftol, 'disp': False})

                params_opt = res.x
                c_opt = params_opt[:2*n].reshape(n, 2)
                s_opt = np.maximum(params_opt[2*n:], 1e-6)

                update_best(c_opt, s_opt)
                params_init = params_opt

            except Exception:
                break

    # ============================================================
    # LP Refinement on Best and Initial Topologies
    # ============================================================
    
    # 1. Try LP on the best solution found by penalty method
    if best_centers is not None and is_valid(best_centers, best_sides):
        orderings = extract_orderings(best_centers, best_sides)
        if orderings:
            c_lp, s_lp = solve_lp_topology(orderings)
            if c_lp is not None:
                update_best(c_lp, s_lp)

    # 2. Try LP on initial topologies (which might be better if penalty method got stuck)
    for c_init, s_init in starts:
        if elapsed() > 580:
            break
        if is_valid(c_init, s_init):
            orderings = extract_orderings(c_init, s_init)
            if orderings:
                c_lp, s_lp = solve_lp_topology(orderings)
                if c_lp is not None:
                    update_best(c_lp, s_lp)

    # 3. Try some "canonical" orderings that might not be covered by initial starts
    # Example: 5 in top row, 5 in bottom row
    # This is effectively what Topology 4 does, but let's ensure we try it explicitly
    # The initial starts already cover many cases.
    
    if best_centers is None:
        best_centers = c1
        best_sides = s1

    return best_centers, best_sides

# EVOLVE-BLOCK-END
