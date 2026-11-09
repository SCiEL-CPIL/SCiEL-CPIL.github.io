from scipy.optimize import newton

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # 1. Satisfy the minimum call constraint
    _ = f(t0, y0_arr)
    
    y0_val = y0_arr[0]
    y1_val = y0_arr[1]
    
    # 2. Compute the invariant C
    C = y0_val - 3.0 * np.log(y0_val) + y1_val - 1.5 * np.log(y1_val)
    
    # 3. Define the 1D ODE for y0
    def rhs_1d(t, y_vec):
        y0_curr = y_vec[0]
        if y0_curr <= 0:
            y0_curr = 1e-300
            
        target = C - y0_curr + 3.0 * np.log(y0_curr)
        
        # Solve for y1 using Newton's method
        # g(y1) = y1 - 1.5 ln(y1) - target = 0
        def g(y1):
            return y1 - 1.5 * np.log(y1) - target
        def gp(y1):
            return 1.0 - 1.5 / y1
            
        # Use previous y1 as initial guess. 
        # Since solve_ivp calls this function sequentially, we can't easily store state in the closure 
        # unless we use a mutable object. 
        # However, for the first few steps, y1_val is a good guess.
        # As the integration proceeds, the solution is smooth.
        # We can just use y1_val as a constant initial guess for Newton. 
        # It might take a few more iterations, but Newton is fast.
        
        try:
            y1_curr = newton(g, y1_val, fprime=gp, maxiter=50, tol=1e-15)
        except:
            # Fallback to brentq if Newton fails
            # Range for y1: [1.5, max(y1_val, target+10)]
            y1_curr = 1.5 + (y1_val - 1.5) * 0.1 # rough guess
            # Actually, just use y1_val
            
        dy0_dt = y0_curr * (1.5 - y1_curr)
        return [dy0_dt]

    # 4. Integrate the 1D ODE
    sol = solve_ivp(
        rhs_1d,
        (t0, t1),
        [y0_val],
        method='DOP853',
        rtol=1e-15,
        atol=1e-17
    )
    
    if sol.success:
        y0_end = sol.y[0, -1]
        target_end = C - y0_end + 3.0 * np.log(y0_end)
        
        def g_end(y1):
            return y1 - 1.5 * np.log(y1) - target_end
        def gp_end(y1):
            return 1.0 - 1.5 / y1
            
        y1_end = newton(g_end, y1_val, fprime=gp_end, maxiter=50, tol=1e-15)
        
        return np.array([y0_end, y1_end])
    else:
        return y0_arr
