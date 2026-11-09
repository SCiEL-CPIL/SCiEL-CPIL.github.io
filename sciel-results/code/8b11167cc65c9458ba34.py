import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Solve the Robertson problem using a custom BDF solver with multi-pass refinement.
    
    Strategy:
    1. Detect the Robertson problem structure to use analytical Jacobian.
    2. Implement a custom BDF(2) solver with adaptive step control.
    3. Use a multi-pass strategy:
       - Pass 1: Coarse tolerance to get a baseline solution and estimate error.
       - Pass 2+: If time remains, run with tighter tolerances (smaller steps) 
         to reduce terminal error, leveraging the fact that the problem is fixed 
         and we can afford to spend more time for higher precision if the score 
         improves (score = -log10(err) - 0.5*log10(calls)).
         
    Note: The score penalizes calls. However, for stiff problems, larger steps 
    in BDF often require more Newton iterations but fewer steps. Tighter tolerances 
    generally force smaller steps, increasing calls. 
    Wait, the score is -log10(err) - 0.5*log10(calls).
    If I double the calls, the penalty is 0.5 * log10(2) ~ 0.15.
    If I reduce the error by 10x, the gain is 1.0.
    So it is very beneficial to reduce error even if it costs some calls, 
    as long as the error reduction is exponential (log scale) vs call increase (log scale).
    
    The current program uses Radau with rtol=1e-12. 
    If I can achieve rtol=1e-14 with a slightly more efficient solver or by 
    optimizing the Newton solver for the specific 3x3 structure, I might gain.
    
    Actually, the biggest gain might come from avoiding the overhead of 
    scipy's solve_ivp which might be doing unnecessary work or not 
    optimizing the specific 3x3 linear solves as efficiently as a 
    hardcoded 3x3 inverse or LU decomposition.
    
    Let's implement a custom BDF(2) with:
    - Analytical Jacobian
    - Hardcoded 3x3 matrix operations (or optimized numpy for 3x3)
    - Adaptive step size
    - Multi-pass: Start with rtol=1e-10, if time < 250s, run again with rtol=1e-12, then 1e-14.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Verify Robertson structure
    is_robertson = False
    if n == 3:
        try:
            y_test = np.array([1.0, 1.0, 0.0])
            f_test = f(t0, y_test)
            f_expected = np.array([-1e4, 7e3, 3e3])
            if np.allclose(f_test, f_expected, rtol=1e-5, atol=1e-8):
                is_robertson = True
        except:
            pass
    
    if not is_robertson:
        # Fallback to SciPy for non-Robertson
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-15)
        if sol.success:
            return sol.y[:, -1]
        return y0
    
    # Define analytical Jacobian for Robertson
    def robertson_jac(t, y):
        y1, y2, y3 = y
        return np.array([
            [-1e4*y2, -1e4*y1, 0],
            [ 1e4*y2, 1e4*y1 - 6e3*y2, 0],
            [ 0,      6e3*y2,      0]
        ])
    
    # Custom BDF(2) Solver
    def solve_bdf(f, jac, t_span, y0, rtol, atol, max_calls):
        t0, t1 = t_span
        y = np.array(y0, dtype=float)
        n = len(y)
        
        # Initial step using Euler to get y_{-1} or just start with BDF(1)
        # BDF(1) is Backward Euler
        # BDF(2) is 2-step
        
        # We will start with BDF(1) for the first step, then switch to BDF(2)
        
        # State variables
        t = t0
        y_prev = y.copy()
        f_prev = f(t, y)
        
        # Initial step size estimate
        # Heuristic for stiff problems
        h = 1e-3
        if t1 > t0:
            # Estimate based on initial derivative magnitude
            f_norm = np.linalg.norm(f_prev)
            if f_norm > 1e-10:
                h = 1e-3 * (1.0 / f_norm)
                h = min(h, (t1 - t0) / 100.0)
                h = max(h, 1e-8)
            else:
                h = (t1 - t0) / 100.0
        else:
            return y0
        
        calls = 1 # initial f call
        
        while t < t1:
            # Ensure we don't overshoot
            if t + h > t1:
                h = t1 - t
            if h < 1e-15:
                break
                
            # BDF(2) coefficients
            # y_{n+1} = (4/3)y_n - (1/3)y_{n-1} - h * f_{n+1}
            # For BDF(1) (first step): y_{n+1} = y_n - h * f_{n+1}
            
            if calls < 2: # First step, use BE
                # Actually, we need y_{n-1} for BDF(2). 
                # Let's just use BDF(1) for the first step to get y1.
                # Then for step 2, we have y0, y1.
                
                # Newton for BE: y_new = y_old - h * f(t+h, y_new)
                t_new = t + h
                y_new = y.copy()
                
                for _ in range(50):
                    f_new = f(t_new, y_new)
                    calls += 1
                    if calls > max_calls:
                        return y, calls, False
                    
                    # Residual: y_new - y - h * f_new = 0
                    res = y_new - y - h * f_new
                    res_norm = np.linalg.norm(res)
                    
                    if res_norm < atol + rtol * np.linalg.norm(y_new):
                        break
                    
                    # Jacobian
                    J = jac(t_new, y_new)
                    # Solve (I + h*J) delta = res
                    A = np.eye(n) + h * J
                    try:
                        delta = np.linalg.solve(A, res)
                    except:
                        # Fallback to fixed point
                        y_new = y - h * f_new
                        break
                    
                    y_new = y_new + delta
                
                y = y_new
                t = t_new
                y_prev = y.copy()
                f_prev = f(t, y)
                calls += 1
                if calls > max_calls:
                    return y, calls, False
                continue # Next iteration will use BDF(2) if we have history? 
                         # Wait, I only have y (current) and y_prev (previous).
                         # For BDF(2), I need y_n and y_{n-1}.
                         # So after step 1, I have y_1 (current) and y_0 (prev).
                         # Next step is y_2 using y_1 and y_0.
            
            else:
                # BDF(2)
                t_new = t + h
                # Predictor: Linear extrapolation
                y_pred = y + (y - y_prev)
                y_new = y_pred
                
                for _ in range(50):
                    f_new = f(t_new, y_new)
                    calls += 1
                    if calls > max_calls:
                        return y, calls, False
                    
                    # BDF(2) formula: y_{n+1} = (4/3)y_n - (1/3)y_{n-1} - h f_{n+1}
                    # Residual: y_new - (4/3)y + (1/3)y_prev + h f_new = 0
                    res = y_new - (4.0/3.0)*y + (1.0/3.0)*y_prev + h * f_new
                    res_norm = np.linalg.norm(res)
                    
                    if res_norm < atol + rtol * np.linalg.norm(y_new):
                        break
                    
                    J = jac(t_new, y_new)
                    # Derivative of residual wrt y_new: I + h*J
                    A = np.eye(n) + h * J
                    try:
                        delta = np.linalg.solve(A, res)
                    except:
                        y_new = (4.0/3.0)*y - (1.0/3.0)*y_prev - h * f_new
                        break
                    
                    y_new = y_new + delta
                
                # Error estimation
                # We can estimate error by comparing BDF(2) step with a smaller step
                # Or just use the residual norm as a proxy? 
                # Standard BDF error estimation uses a lower order method.
                # For simplicity, let's use a fixed step size or very conservative adaptation.
                # Given the stiffness, let's try to keep h small initially.
                
                # Simple adaptation: if res_norm is high, reduce h.
                # This is crude but robust.
                if res_norm > 10 * (atol + rtol * np.linalg.norm(y_new)):
                    h *= 0.5
                    continue # Retry with smaller h
                elif res_norm < 0.1 * (atol + rtol * np.linalg.norm(y_new)):
                    h *= 1.5
                    if t + h > t1:
                        h = t1 - t
                
                y_prev = y.copy()
                y = y_new
                t = t_new
        
        return y, calls, True
    
    # Multi-pass strategy
    # Pass 1: rtol=1e-10, max_calls=10000
    # Pass 2: rtol=1e-12, max_calls=50000
    # Pass 3: rtol=1e-14, max_calls=200000
    
    best_y = y0
    best_score = -np.inf
    
    tolerances = [1e-10, 1e-12, 1e-14]
    max_calls_list = [10000, 50000, 200000]
    
    for i, rtol in enumerate(tolerances):
        if time.time() > deadline - 30:
            break
            
        max_calls = max_calls_list[i]
        
        try:
            y_sol, calls, success = solve_bdf(f, robertson_jac, t_span, y0, rtol, 1e-15, max_calls)
            
            if not success:
                continue
                
            # Calculate score estimate
            # We don't know the true error, but we assume tighter rtol -> smaller error.
            # We calculate the score based on the assumption that error ~ rtol.
            # score = -log10(err) - 0.5*log10(calls)
            # If we assume err is proportional to rtol, we can estimate.
            # But we want to return the best result.
            # The problem says "scored by ... against a SciPy reference".
            # So we just need to return the y with the highest potential score.
            
            # Let's just keep the last successful run with the tightest tolerance
            # provided it finished in time.
            best_y = y_sol
            
        except Exception:
            pass
            
    return best_y
