import numpy as np
import time

def find_root_system(F, x0):
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    def time_ok():
        return time.time() - start_time < time_limit

    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)

    call_count = 0

    def F_count(x_pt):
        nonlocal call_count
        call_count += 1
        return F(x_pt)

    # Initial evaluation
    try:
        Fx = F_count(x)
    except:
        return np.zeros(2)

    if not np.all(np.isfinite(Fx)):
        return x

    norm_sq = float(np.dot(Fx, Fx))
    if norm_sq <= tol_sq:
        return x

    best_x = x.copy()
    best_norm_sq = norm_sq

    # Homotopy/Continuation method:
    # Solve F_t(x) = t * F(x) + (1-t) * (x - x0) = 0
    # For t=0, root is x0.
    # We track the path as t increases to 1.
    # This helps avoid local minima by following the "unfolding" of the solution.

    # Parameters for continuation
    t = 0.0
    # Start with a small t to ensure stability
    # Or start with t such that the step is reasonable.
    # Let's start with t=0.1 and increase.
    
    # We need to solve G_t(x) = 0 where G_t(x) = t*F(x) + (1-t)*(x-x0)
    # Jacobian J_G = t * J_F + (1-t) * I
    # We don't know J_F, so we estimate it.
    # However, for t small, J_G is close to I, so Newton should work well with J_F ~ 0?
    # No, J_G = t J_F + (1-t) I.
    # If t is small, J_G is close to I.
    # We can approximate J_F by finite differences if needed, or just use the fact that
    # for small t, the term (1-t)(x-x0) dominates, so the root is close to x0.
    
    # Strategy:
    # 1. Start at t=0. x=x0.
    # 2. Increase t in steps.
    # 3. At each t, use Newton's method to find the root of G_t.
    #    Use the previous solution as initial guess.
    #    Estimate J_F using finite differences at the current x?
    #    Or use Broyden's method for G_t?
    #    Since J_G depends on J_F, and J_F is constant (or slowly varying),
    #    we can maintain an estimate of J_F.
    
    # Let's maintain J_F estimate using Broyden's method on F itself?
    # But we are solving G_t.
    # Let's just use finite differences for J_G directly?
    # J_G(x) = t * J_F(x) + (1-t) * I
    # We can compute J_F(x) via finite differences.
    # Cost: 2 calls per Jacobian update.
    # Newton step: solve J_G s = -G_t(x).
    
    # To save calls, we can use Broyden's method for G_t.
    # But G_t changes with t.
    # When t changes, J_G changes.
    # So we might need to re-estimate J_F or update J_G.
    
    # Simpler approach:
    # Use Broyden's method on F(x) = 0 directly, but with a "bias" term?
    # No, the homotopy idea is to solve a sequence of easier problems.
    
    # Let's try a different structural change:
    # **Levenberg-Marquardt (LM) with adaptive lambda and multiple restarts.**
    # LM is robust for non-linear least squares (minimizing ||F(x)||^2).
    # It handles singular Jacobians well.
    # It has global convergence properties.
    # 
    # Algorithm:
    # 1. Estimate J_F.
    # 2. LM iteration:
    #    (J^T J + lambda I) s = -J^T F
    #    Take step x_new = x + s.
    #    If ||F(x_new)|| < ||F(x)||, accept, decrease lambda.
    #    Else, reject, increase lambda.
    # 3. If stagnation, restart from a different point.
    
    # This is a standard, robust method for root finding in 2D.
    # The "structural change" is moving from pure Newton/Broyden to LM with robust global convergence.
    
    def compute_jacobian(x_pt, Fx_pt):
        h = 1e-7
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        try:
            Fx1 = F_count(x_pt + e1)
            Fx2 = F_count(x_pt + e2)
            J = np.array([(Fx1 - Fx_pt)/h, (Fx2 - Fx_pt)/h]).T
            if not np.all(np.isfinite(J)):
                return None
            return J
        except:
            return None

    # Initial Jacobian
    J = compute_jacobian(x, Fx)
    if J is None:
        J = np.eye(2)

    lambda_lm = 1e-3
    max_iter = 200
    stagnation_count = 0
    best_norm_so_far = norm_sq
    
    for _ in range(max_iter):
        if not time_ok() or call_count > 480:
            break
            
        if norm_sq <= tol_sq:
            return x

        # LM Step
        try:
            # Solve (J^T J + lambda I) s = -J^T F
            JtJ = J.T @ J
            JtF = J.T @ Fx
            A = JtJ + lambda_lm * np.eye(2)
            s = np.linalg.solve(A, -JtF)
        except np.linalg.LinAlgError:
            # If singular, increase lambda
            lambda_lm *= 10
            continue

        # Limit step size
        s_norm = np.linalg.norm(s)
        if s_norm > 10.0:
            s = s * (10.0 / s_norm)

        x_try = x + s
        try:
            Fx_try = F_count(x_try)
        except:
            lambda_lm *= 10
            continue

        if not np.all(np.isfinite(Fx_try)):
            lambda_lm *= 10
            continue

        norm_sq_try = float(np.dot(Fx_try, Fx_try))

        if norm_sq_try < norm_sq:
            # Accept
            x = x_try
            Fx = Fx_try
            norm_sq = norm_sq_try
            lambda_lm = max(1e-10, lambda_lm * 0.5)
            stagnation_count = 0
            
            # Update Jacobian periodically or if change is large
            # For LM, it's common to recompute J every few steps or when accepted.
            # To save calls, let's recompute every 5 steps or if step is large.
            # But for 2D, 2 calls is cheap. Let's recompute every accepted step to ensure accuracy.
            # Actually, Broyden update is more efficient.
            # Let's switch to Broyden update for J to save calls.
            dx = s
            dF = Fx_try - Fx
            # Wait, Fx is already updated. We need old Fx.
            # Let's store old Fx before update.
            # Refactor:
            # old_Fx = Fx
            # ...
            # dF = Fx_new - old_Fx
            
            # Let's stick to recomputing J every few steps for simplicity and robustness.
            # If we recompute every step, that's 2 extra calls per step.
            # Total calls per step: 1 (F) + 2 (J) = 3.
            # 100 steps = 300 calls. Safe.
            
            if _ % 5 == 0 or s_norm > 1e-3:
                J_new = compute_jacobian(x, Fx)
                if J_new is not None:
                    J = J_new
        else:
            # Reject
            lambda_lm *= 10
            stagnation_count += 1
            
            if stagnation_count > 10:
                # Restart
                break

        if norm_sq <= best_norm_so_far:
            best_norm_so_far = norm_sq
            best_x = x.copy()

    # If LM failed or stagnated, try a few random restarts with LM
    if call_count < 450 and time_ok():
        rng = np.random.default_rng(42)
        for _ in range(3):
            if call_count > 450 or not time_ok():
                break
            x_r = x + rng.normal(0, 1.0, size=2)
            try:
                Fx_r = F_count(x_r)
            except:
                continue
            if not np.all(np.isfinite(Fx_r)):
                continue
            nr = float(np.dot(Fx_r, Fx_r))
            if nr < best_norm_sq:
                best_x = x_r.copy()
                best_norm_sq = nr
                if nr <= tol_sq:
                    return best_x
            
            # Run LM from x_r
            x_cur = x_r
            Fx_cur = Fx_r
            norm_cur = nr
            J_cur = compute_jacobian(x_cur, Fx_cur)
            if J_cur is None:
                J_cur = np.eye(2)
            
            lambda_cur = 1e-3
            for _ in range(50):
                if call_count > 480 or not time_ok():
                    break
                if norm_cur <= tol_sq:
                    return x_cur
                
                try:
                    JtJ = J_cur.T @ J_cur
                    JtF = J_cur.T @ Fx_cur
                    A = JtJ + lambda_cur * np.eye(2)
                    s = np.linalg.solve(A, -JtF)
                except:
                    lambda_cur *= 10
                    continue
                
                s_norm = np.linalg.norm(s)
                if s_norm > 10:
                    s *= 10 / s_norm
                
                x_t = x_cur + s
                try:
                    Fx_t = F_count(x_t)
                except:
                    lambda_cur *= 10
                    continue
                
                if not np.all(np.isfinite(Fx_t)):
                    lambda_cur *= 10
                    continue
                
                nt = float(np.dot(Fx_t, Fx_t))
                if nt < norm_cur:
                    x_cur = x_t
                    Fx_cur = Fx_t
                    norm_cur = nt
                    lambda_cur = max(1e-10, lambda_cur * 0.5)
                    if _ % 5 == 0:
                        J_new = compute_jacobian(x_cur, Fx_cur)
                        if J_new is not None:
                            J_cur = J_new
                    if nt < best_norm_sq:
                        best_norm_sq = nt
                        best_x = x_cur.copy()
                else:
                    lambda_cur *= 10

    return best_x
