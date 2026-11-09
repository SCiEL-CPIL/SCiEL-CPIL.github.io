# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 620

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
    # Smooth penalty objective with soft-min for overlap
    # ============================================================
    def smooth_penalty(params, beta=50.0):
        """
        Minimize: -sum(sides) + penalty
        Penalty uses smooth approximations:
          - Boundary: softplus(v) * w
          - Overlap: for each pair, smooth_min(t - dx, t - dy) * w
            where smooth_min(a,b) = -log(exp(-a)+exp(-b))/beta approximates min(a,b)
            We want to penalize when both (t-dx)>0 AND (t-dy)>0, i.e., min(t-dx, t-dy) > 0
        """
        centers = params[:2*n].reshape(n, 2)
        sides = params[2*n:]

        # Softplus for boundary violations
        def softplus(x, b):
            return np.log1p(np.exp(-b * x)) / b

        obj = -np.sum(sides)
        W = 100.0

        # Boundary penalties
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            v_l = -(cx - s/2)
            v_r = (cx + s/2) - 1
            v_b = -(cy - s/2)
            v_t = (cy + s/2) - 1
            obj += W * (softplus(v_l, beta) + softplus(v_r, beta) +
                        softplus(v_b, beta) + softplus(v_t, beta))

        # Overlap penalties using smooth min
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - 1e-7
                if t <= 0:
                    continue
                a = t - dx
                b = t - dy
                # smooth min(a, b) ≈ -(1/beta)*log(exp(-beta*a) + exp(-beta*b))
                # Clamp for numerical stability
                a_c = np.clip(a, -50/beta, 50/beta)
                b_c = np.clip(b, -50/beta, 50/beta)
                sm = -np.log(np.exp(-beta * a_c) + np.exp(-beta * b_c)) / beta
                # Penalize when sm > 0 (overlap exists)
                obj += W * softplus(sm, beta)

        return obj

    def analytic_gradient(params, beta=50.0):
        """Numerical gradient fallback - use finite differences if needed."""
        pass  # We'll use numerical gradient via SLSQP

    # ============================================================
    # Build initial topologies
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
    # Right region: 0.5 wide, 1 tall. Place 9 squares.
    # 3 rows of 3: each 0.5/3 ≈ 0.1667
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
    # Bottom: height 1-a3=2/3, width 1. Place 7 squares.
    # 2 rows: top row 4, bottom row 3
    b3a = 0.25  # 4 in a row
    b3b = 1.0/3  # 3 in a row
    # Row 1 (4 squares of 0.25): height 0.25, from y=2/3-0.25 to 2/3
    for col in range(4):
        idx = 3 + col
        c3[idx, 0] = b3a/2 + col*b3a
        c3[idx, 1] = (2.0/3) - b3a/2
        s3[idx] = b3a
    # Row 2 (3 squares of 1/3): height 1/3, from y=0 to 1/3
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

    # Topology 5: 4 corners (a) + 6 center (b), optimized
    # Corner squares at corners, central region (1-2a)^2
    # 6 squares in 2x3: b = (1-2a)/3
    # Sum = 4a + 6*(1-2a)/3 = 4a + 2 - 4a = 2. Not great but valid start.
    c5 = np.zeros((n, 2))
    s5 = np.zeros(n)
    a5 = 0.3
    b5 = (1 - 2*a5)/3
    c5[0] = [a5/2, a5/2]; s5[0] = a5
    c5[1] = [1-a5/2, a5/2]; s5[1] = a5
    c5[2] = [a5/2, 1-a5/2]; s5[2] = a5
    c5[3] = [1-a5/2, 1-a5/2]; s5[3] = a5
    # 6 squares in center: 2 rows x 3 cols
    for r in range(2):
        for col in range(3):
            idx = 4 + r*3 + col
            c5[idx, 0] = a5 + b5/2 + col*b5
            c5[idx, 1] = a5 + b5/2 + r*b5
            s5[idx] = b5
    update_best(c5, s5)
    starts.append((c5.copy(), s5.copy()))

    # Topology 6: 5+5 split (left 5, right 5)
    c6 = np.zeros((n, 2))
    s6 = np.zeros(n)
    # Left: 5 squares in 1x1, arranged as 2+3
    # Right: 5 squares in 1x1
    # Actually let's do: left half 0.5x1 with 5 squares, right half 0.5x1 with 5
    bl = 0.5/2  # 2 in a row in 0.5 width = 0.25
    # Left: 2 rows of 2 (4 squares) + 1 on top? 
    # Simpler: left 0.5x1, 5 squares: 2x2 grid of 0.25 + 1 of 0.25 on top
    for r in range(2):
        for col in range(2):
            idx = r*2 + col
            c6[idx, 0] = 0.125 + col*0.25
            c6[idx, 1] = 0.125 + r*0.25
            s6[idx] = 0.25
    c6[4, 0] = 0.25
    c6[4, 1] = 0.75
    s6[4] = 0.25
    # Right: mirror
    for r in range(2):
        for col in range(2):
            idx = 5 + r*2 + col
            c6[idx, 0] = 0.625 + col*0.25
            c6[idx, 1] = 0.125 + r*0.25
            s6[idx] = 0.25
    c6[9, 0] = 0.75
    c6[9, 1] = 0.75
    s6[9] = 0.25
    update_best(c6, s6)
    starts.append((c6.copy(), s6.copy()))

    # Add perturbed versions of best starts
    np.random.seed(42)
    for base_c, base_s in starts[:3]:
        for _ in range(5):
            c_p = base_c + np.random.normal(0, 0.02, (n, 2))
            s_p = base_s + np.random.normal(0, 0.01, n)
            s_p = np.clip(s_p, 0.05, 0.55)
            starts.append((c_p, s_p))

    # ============================================================
    # Optimization loop with adaptive beta
    # ============================================================
    for start_idx, (c_init, s_init) in enumerate(starts):
        if elapsed() > 580:
            break

        params_init = np.concatenate([c_init.flatten(), s_init])

        # Multi-stage: coarse to fine
        for stage, (beta, maxiter, ftol) in enumerate([(30.0, 200, 1e-8),
                                                        (80.0, 300, 1e-10),
                                                        (200.0, 200, 1e-12)]):
            if elapsed() > 585:
                break
            try:
                res = minimize(smooth_penalty, params_init, args=(beta,),
                               method='SLSQP',
                               options={'maxiter': maxiter, 'ftol': ftol, 'disp': False})

                params_opt = res.x
                c_opt = params_opt[:2*n].reshape(n, 2)
                s_opt = np.maximum(params_opt[2*n:], 1e-6)

                update_best(c_opt, s_opt)

                # Use this as init for next stage
                params_init = params_opt

            except Exception:
                break

        # Final high-precision polish with very high beta
        if elapsed() < 590:
            try:
                c_pol = params_init[:2*n].reshape(n, 2)
                s_pol = np.maximum(params_init[2*n:], 1e-6)
                if is_valid(c_pol, s_pol):
                    res_pol = minimize(smooth_penalty, params_init, args=(500.0,),
                                       method='SLSQP',
                                       options={'maxiter': 500, 'ftol': 1e-14, 'disp': False})
                    params_pol = res_pol.x
                    c_pol = params_pol[:2*n].reshape(n, 2)
                    s_pol = np.maximum(params_pol[2*n:], 1e-6)
                    update_best(c_pol, s_pol)
            except Exception:
                pass

    # ============================================================
    # Local search: try small random perturbations around best
    # ============================================================
    if best_centers is not None and elapsed() < 600:
        np.random.seed(999)
        for _ in range(200):
            if elapsed() > 605:
                break
            c_try = best_centers + np.random.normal(0, 0.005, (n, 2))
            s_try = best_sides + np.random.normal(0, 0.003, n)
            s_try = np.clip(s_try, 0.01, 0.6)
            params_try = np.concatenate([c_try.flatten(), s_try])
            try:
                res = minimize(smooth_penalty, params_try, args=(100.0,),
                               method='SLSQP',
                               options={'maxiter': 100, 'ftol': 1e-10, 'disp': False})
                c_r = res.x[:2*n].reshape(n, 2)
                s_r = np.maximum(res.x[2*n:], 1e-6)
                update_best(c_r, s_r)
            except Exception:
                pass

    if best_centers is None:
        best_centers = c1
        best_sides = s1

    return best_centers, best_sides

# EVOLVE-BLOCK-END
