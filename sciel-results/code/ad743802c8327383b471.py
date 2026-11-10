import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import linprog

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

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
        return True, float(np.sum(s))

    def clamp_to_box(centers, sides):
        """Clamp centers so squares fit in [0,1]^2."""
        s = np.maximum(sides, 1e-8)
        cx = centers[:, 0].copy()
        cy = centers[:, 1].copy()
        for i in range(n):
            cx[i] = np.clip(cx[i], s[i]/2, 1 - s[i]/2)
            cy[i] = np.clip(cy[i], s[i]/2, 1 - s[i]/2)
        return np.stack([cx, cy], axis=1), s

    def solve_lp_for_pattern(separation_pattern):
        """
        Solve LP for a given separation pattern.
        separation_pattern: dict mapping (i,j) to one of 'x', 'y', 'xneg', 'yneg'
        'x' means i is left of j: cx_j - cx_i >= (s_i+s_j)/2
        'xneg' means i is right of j: cx_i - cx_j >= (s_i+s_j)/2
        'y' means i is below j: cy_j - cy_i >= (s_i+s_j)/2
        'yneg' means i is above j: cy_i - cy_j >= (s_i+s_j)/2
        """
        # Variables: s_0..s_9 (10 vars), cx_0..cx_9 (10 vars), cy_0..cy_9 (10 vars) = 30 vars
        # We want to maximize sum(s_i)
        # c = -[1,1,...,1, 0,...,0, 0,...,0]
        c = np.zeros(3*n)
        c[:n] = -1.0  # maximize sum of sides
        
        # Inequality constraints: A_ub @ x <= b_ub
        A_rows = []
        b_rows = []
        
        # Boundary constraints:
        # cx_i - s_i/2 >= 0  =>  -cx_i + s_i/2 <= 0
        # cx_i + s_i/2 <= 1  =>  cx_i + s_i/2 <= 1
        # cy_i - s_i/2 >= 0  =>  -cy_i + s_i/2 <= 0
        # cy_i + s_i/2 <= 1  =>  cy_i + s_i/2 <= 1
        # s_i >= 0 (handled by bounds)
        
        for i in range(n):
            # -cx_i + s_i/2 <= 0
            row = np.zeros(3*n)
            row[n + i] = -1.0
            row[2*n + i] = 0.5
            A_rows.append(row)
            b_rows.append(0.0)
            
            # cx_i + s_i/2 <= 1
            row = np.zeros(3*n)
            row[n + i] = 1.0
            row[2*n + i] = 0.5
            A_rows.append(row)
            b_rows.append(1.0)
            
            # -cy_i + s_i/2 <= 0
            row = np.zeros(3*n)
            row[2*n + i] = 0.5
            row[n + n + i] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
            
            # cy_i + s_i/2 <= 1
            row = np.zeros(3*n)
            row[2*n + i] = 0.5
            row[n + n + i] = 1.0
            A_rows.append(row)
            b_rows.append(1.0)
        
        # Separation constraints from pattern
        for (i, j), sep in separation_pattern.items():
            if sep == 'x':
                # cx_j - cx_i >= (s_i + s_j)/2
                # -cx_i - s_i/2 + cx_j - s_j/2 >= 0
                # cx_i + s_i/2 - cx_j + s_j/2 <= 0
                row = np.zeros(3*n)
                row[n + i] = 1.0
                row[2*n + i] = 0.5
                row[n + j] = -1.0
                row[2*n + j] = 0.5
                A_rows.append(row)
                b_rows.append(0.0)
            elif sep == 'xneg':
                # cx_i - cx_j >= (s_i + s_j)/2
                # -cx_i - s_i/2 + cx_j - s_j/2 <= 0
                # cx_i + s_i/2 - cx_j + s_j/2 >= 0  =>  -cx_i - s_i/2 + cx_j - s_j/2 <= 0
                row = np.zeros(3*n)
                row[n + i] = -1.0
                row[2*n + i] = -0.5
                row[n + j] = 1.0
                row[2*n + j] = -0.5
                A_rows.append(row)
                b_rows.append(0.0)
            elif sep == 'y':
                # cy_j - cy_i >= (s_i + s_j)/2
                # cy_i + s_i/2 - cy_j + s_j/2 <= 0
                row = np.zeros(3*n)
                row[2*n + i] = 0.5
                row[2*n + n + i] = 1.0
                row[2*n + j] = -0.5
                row[2*n + n + j] = -1.0
                A_rows.append(row)
                b_rows.append(0.0)
            elif sep == 'yneg':
                # cy_i - cy_j >= (s_i + s_j)/2
                # -cy_i - s_i/2 + cy_j - s_j/2 <= 0
                row = np.zeros(3*n)
                row[2*n + i] = -0.5
                row[2*n + n + i] = -1.0
                row[2*n + j] = 0.5
                row[2*n + n + j] = 1.0
                A_rows.append(row)
                b_rows.append(0.0)
        
        if A_rows:
            A_ub = np.array(A_rows)
            b_ub = np.array(b_rows)
        else:
            A_ub = None
            b_ub = None
        
        bounds = [(1e-8, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 1.0)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                s = res.x[:n]
                cx = res.x[n:2*n]
                cy = res.x[2*n:3*n]
                centers = np.stack([cx, cy], axis=1)
                return centers, s, float(np.sum(s))
        except Exception:
            pass
        return None, None, None

    def solve_lp_with_fixed_positions(centers, sides):
        """
        Given fixed centers, maximize sum of sides subject to non-overlap and boundary.
        This is an LP in the side variables only.
        """
        cx = centers[:, 0]
        cy = centers[:, 1]
        
        # Variables: s_0..s_9
        c = -np.ones(n)
        
        A_rows = []
        b_rows = []
        
        # Boundary: s_i/2 <= cx_i, s_i/2 <= 1-cx_i, s_i/2 <= cy_i, s_i/2 <= 1-cy_i
        # => s_i <= 2*min(cx_i, 1-cx_i, cy_i, 1-cy_i)
        for i in range(n):
            max_s = 2.0 * min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(max_s)
        
        # Non-overlap: for each pair (i,j), need |cx_i - cx_j| >= (s_i+s_j)/2 OR |cy_i - cy_j| >= (s_i+s_j)/2
        # Since centers are fixed, we can determine which axis provides separation.
        # If |cx_i - cx_j| >= |cy_i - cy_j|, then x-separation is the binding one (or we can use it)
        # Actually, we need: NOT(|cx_i - cx_j| < (s_i+s_j)/2 AND |cy_i - cy_j| < (s_i+s_j)/2)
        # This means: (s_i+s_j)/2 <= |cx_i - cx_j| OR (s_i+s_j)/2 <= |cy_i - cy_j|
        # Since we want to maximize, we'll use the axis that gives more room.
        # The constraint is: s_i + s_j <= 2*max(|cx_i - cx_j|, |cy_i - cy_j|)
        # Wait, that's not quite right. We need:
        # s_i + s_j <= 2*|cx_i - cx_j|  OR  s_i + s_j <= 2*|cy_i - cy_j|
        # This is a disjunction. For LP, we need to pick one.
        # We'll pick the axis that gives the larger separation, as that allows larger squares.
        # Actually, the correct interpretation: the squares don't overlap if they are separated in x OR in y.
        # So the constraint is: s_i + s_j <= 2*|cx_i - cx_j| OR s_i + s_j <= 2*|cy_i - cy_j|
        # For LP, we can use: s_i + s_j <= 2*max(|cx_i - cx_j|, |cy_i - cy_j|)
        # Because if s_i + s_j <= 2*max(...), then it's <= 2*|cx| or <= 2*|cy| (whichever is max).
        
        for i in range(n):
            for j in range(i+1, n):
                sep_x = abs(cx[i] - cx[j])
                sep_y = abs(cy[i] - cy[j])
                max_sep = max(sep_x, sep_y)
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_rows.append(row)
                b_rows.append(2.0 * max_sep)
        
        if A_rows:
            A_ub = np.array(A_rows)
            b_ub = np.array(b_rows)
        else:
            A_ub = None
            b_ub = None
        
        bounds = [(1e-8, 1.0)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                s = res.x
                return centers, s, float(np.sum(s))
        except Exception:
            pass
        return None, None, None

    def refine_with_slsqp(centers, sides):
        """Use SLSQP to refine a valid configuration."""
        def objective(x):
            s = np.maximum(x[2::3], 1e-10)
            return -np.sum(s)
        
        def gradient_obj(x):
            s = x[2::3]
            grad = np.zeros(3*n)
            for i in range(n):
                grad[3*i+2] = -1.0 if s[i] > 1e-10 else 0.0
            return grad
        
        constraints = []
        # Boundary constraints
        for i in range(n):
            constraints.append({'type': 'ineq',
                                'fun': lambda x, i=i: x[3*i] - x[3*i+2]/2,
                                'jac': lambda x, i=i: np.concatenate([[1.0, 0.0, -0.5], np.zeros(3*n-3)])})
            constraints.append({'type': 'ineq',
                                'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]/2,
                                'jac': lambda x, i=i: np.concatenate([[-1.0, 0.0, -0.5], np.zeros(3*n-3)])})
            constraints.append({'type': 'ineq',
                                'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]/2,
                                'jac': lambda x, i=i: np.concatenate([[0.0, 1.0, -0.5], np.zeros(3*n-3)])})
            constraints.append({'type': 'ineq',
                                'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]/2,
                                'jac': lambda x, i=i: np.concatenate([[0.0, -1.0, -0.5], np.zeros(3*n-3)])})
            constraints.append({'type': 'ineq',
                                'fun': lambda x, i=i: x[3*i+2],
                                'jac': lambda x, i=i: np.concatenate([[0.0, 0.0, 1.0], np.zeros(3*n-3)])})
        
        # Overlap constraints using smooth product form
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
                    sign_dx = 1.0 if x[3*i] > x[3*j] else (-1.0 if x[3*i] < x[3*j] else 0.0)
                    sign_dy = 1.0 if x[3*i+1] > x[3*j+1] else (-1.0 if x[3*i+1] < x[3*j+1] else 0.0)
                    
                    grad = np.zeros(3*n)
                    a = req - dx
                    b = req - dy
                    
                    grad[3*i] = sign_dx * b
                    grad[3*i+1] = sign_dy * a
                    grad[3*i+2] = -0.5 * (a + b)
                    
                    grad[3*j] = -sign_dx * b
                    grad[3*j+1] = -sign_dy * a
                    grad[3*j+2] = -0.5 * (a + b)
                    
                    return grad
                
                constraints.append({'type': 'ineq',
                                     'fun': overlap_con,
                                     'jac': overlap_jac})
        
        bounds = [(0.0, 1.0)] * (2*n) + [(1e-8, 1.0)] * n
        
        x0 = to_flat(centers, sides)
        try:
            res = minimize(objective, x0, jac=gradient_obj, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-9})
            if res.success:
                centers_res, sides_res = from_flat(res.x)
                centers_res, sides_res = clamp_to_box(centers_res, sides_res)
                return centers_res, sides_res
        except Exception:
            pass
        return centers, sides

    # Initial heuristic layout
    centers_init = np.array([
        [0.25, 0.25],
        [0.75, 0.25],
        [0.125, 0.625],
        [0.375, 0.625],
        [0.625, 0.625],
        [0.875, 0.625],
        [0.125, 0.875],
        [0.375, 0.875],
        [0.625, 0.875],
        [0.875, 0.875]
    ])
    sides_init = np.array([
        0.5, 0.5,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25
    ])

    best_centers = centers_init.copy()
    best_sides = sides_init.copy()
    valid, best_score = validate(best_centers, best_sides)
    if not valid:
        best_centers, best_sides = clamp_to_box(best_centers, best_sides)
        valid, best_score = validate(best_centers, best_sides)

    # Generate diverse initial layouts
    layouts = []
    layouts.append((centers_init, sides_init))
    
    # 2x5 grid
    c2 = np.array([[0.1,0.25],[0.3,0.25],[0.5,0.25],[0.7,0.25],[0.9,0.25],
                    [0.1,0.75],[0.3,0.75],[0.5,0.75],[0.7,0.75],[0.9,0.75]])
    s2 = np.full(n, 0.2)
    layouts.append((c2, s2))
    
    # 5x2 grid
    c5 = np.array([[0.25,0.1],[0.25,0.3],[0.25,0.5],[0.25,0.7],[0.25,0.9],
                    [0.75,0.1],[0.75,0.3],[0.75,0.5],[0.75,0.7],[0.75,0.9]])
    s5 = np.full(n, 0.2)
    layouts.append((c5, s5))
    
    # 1x10 row
    c10 = np.array([[0.05 + 0.1*i, 0.5] for i in range(10)])
    s10 = np.full(n, 0.1)
    layouts.append((c10, s10))
    
    # Random layouts
    np.random.seed(42)
    for _ in range(5):
        cr = np.random.uniform(0.1, 0.9, (n, 2))
        sr = np.random.uniform(0.08, 0.3, n)
        layouts.append((cr, sr))

    # Try each layout with LP sizing + SLSQP refinement
    for c_init, s_init in layouts:
        if time.time() > deadline - 30:
            break
        
        c_fix, s_fix = clamp_to_box(c_init, s_init)
        
        # LP sizing with fixed positions
        lp_centers, lp_sides, lp_score = solve_lp_with_fixed_positions(c_fix, s_fix)
        if lp_centers is not None:
            valid_lp, lp_score_val = validate(lp_centers, lp_sides)
            if valid_lp and lp_score_val > best_score:
                best_score = lp_score_val
                best_centers = lp_centers.copy()
                best_sides = lp_sides.copy()
        
        # SLSQP refinement
        if time.time() > deadline - 30:
            break
        refined_centers, refined_sides = refine_with_slsqp(c_fix, s_fix)
        valid_ref, ref_score = validate(refined_centers, refined_sides)
        if valid_ref and ref_score > best_score:
            best_score = ref_score
            best_centers = refined_centers.copy()
            best_sides = refined_sides.copy()

    # Final high-precision polish
    if time.time() < deadline - 20:
        refined_centers, refined_sides = refine_with_slsqp(best_centers, best_sides)
        valid_pol, pol_score = validate(refined_centers, refined_sides)
        if valid_pol and pol_score > best_score:
            best_score = pol_score
            best_centers = refined_centers.copy()
            best_sides = refined_sides.copy()

    # Second polish
    if time.time() < deadline - 10:
        refined_centers, refined_sides = refine_with_slsqp(best_centers, best_sides)
        valid_pol2, pol2_score = validate(refined_centers, refined_sides)
        if valid_pol2 and pol2_score > best_score:
            best_score = pol2_score
            best_centers = refined_centers.copy()
            best_sides = refined_sides.copy()

    # Final validation
    valid, final_score = validate(best_centers, best_sides)
    if not valid:
        return centers_init, sides_init

    return best_centers, best_sides
