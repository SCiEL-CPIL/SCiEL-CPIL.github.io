"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Solves the Robertson problem using a custom fixed-step TR-BDF2 scheme
    with analytic Jacobians and LU decomposition.
    
    The Robertson problem is:
    y' = f(t, y) where:
    y0' = -0.04*y0 + 1e4*y1*y2
    y1' =  0.04*y0 - 1e4*y1*y2 - 0.31*y1^2
    y2' =  0.31*y1^2
    
    y0 = [1, 0, 0], t in [0, 40]
    
    Strategy:
    1. Detect if the problem is Robertson (by checking y0 and f structure implicitly via constants).
       Actually, we can just assume it is Robertson if the signature matches, but to be safe,
       we will implement a generic stiff solver that *uses* analytic Jacobians if possible?
       No, the task says "fixed Robertson initial-value problem". 
       We can hardcode the Jacobian for Robertson to save calls and improve stability.
       
       However, the evaluator provides `f`. We must use `f` for the RHS.
       But we can compute the Jacobian J analytically for Robertson to speed up the implicit solve.
       
       Let's verify the Jacobian for Robertson:
       f0 = -0.04*y0 + 1e4*y1*y2
       f1 =  0.04*y0 - 1e4*y1*y2 - 0.31*y1^2
       f2 =  0.31*y1^2
       
       J = [[-0.04,  1e4*y2, 1e4*y1],
            [ 0.04, -1e4*y2 - 0.62*y1, -1e4*y1],
            [ 0,     0.62*y1, 0]]
            
    2. Use TR-BDF2 (Trapezoidal Rule + Backward Differentiation Formula of order 2).
       This is an L-stable 2nd order method.
       
       Step 1 (Trapezoidal):
       y_{n+1/2} = y_n + h/2 * (f(t_n, y_n) + f(t_{n+1}, y_{n+1/2}))
       Solve: (I - h/2 * J) y_{n+1/2} = y_n + h/2 * f_n + h/2 * J * y_{n+1/2}? 
       Wait, standard implicit step:
       y_{n+1/2} - h/2 * f(t_{n+1}, y_{n+1/2}) = y_n + h/2 * f_n
       
       Linearize: y_{n+1/2} - h/2 * (f_n + J (y_{n+1/2} - y_n)) = y_n + h/2 * f_n
       y_{n+1/2} - h/2 * J y_{n+1/2} + h/2 * J y_n - h/2 * f_n = y_n + h/2 * f_n
       (I - h/2 * J) y_{n+1/2} = y_n + h/2 * f_n - h/2 * J y_n + h/2 * f_n ? No.
       
       Let's stick to Newton iteration for the implicit step, reusing the Jacobian structure.
       Since the system is small (3x3), Newton is very cheap.
       
    3. Step size selection:
       Robertson is stiff. The solution changes rapidly at the beginning.
       t in [0, 40].
       y1 decays fast.
       A fixed step size might be risky if too large.
       However, TR-BDF2 is A-stable.
       Let's use a variable step size or a very fine fixed step?
       "Fixed Robertson problem" implies we know the dynamics.
       
       Let's try a hybrid:
       - Use a small step h for the first part (stiff transient), then larger steps.
       - Or simply use a fixed step size that is small enough to resolve the transient.
       The transient ends around t=10-20?
       y1 ~ exp(-0.31t) roughly? No, it's complex.
       
       Actually, the most efficient way for this specific problem is to use the known solution behavior.
       But we must be general enough to pass the interface.
       
       Let's implement a robust implicit solver with adaptive step size control based on error estimation?
       That requires storing previous steps.
       
       Alternative: Use `scipy.integrate.solve_ivp` with `method='Radau'` but provide the analytic Jacobian `jac`.
       The current program does NOT provide `jac`. This is likely the bottleneck!
       `solve_ivp` with Radau estimates the Jacobian via finite differences (3 * 3 = 9 extra function calls per Jacobian update, plus overhead).
       Providing the analytic Jacobian reduces function calls significantly and improves stability.
       
       Let's check the score: 5.6791.
       If we reduce calls by a factor of 2-3, the score term -0.5*log10(calls) increases by 0.5*log10(2) ~ 0.15 or 0.5*log10(3) ~ 0.25.
       Also, better stability might allow tighter tolerances or fewer steps.
       
       Strategy:
       1. Wrap `f` to count calls.
       2. Define the analytic Jacobian `jac(t, y)` for the Robertson problem.
          How to know it's Robertson?
          The problem statement says "fixed Robertson initial-value problem".
          We can assume `f` is the Robertson RHS.
          We can hardcode the Jacobian.
          
          To be safe, we can detect if the problem looks like Robertson by checking the initial state `y0`.
          If `y0` is close to [1, 0, 0], we use the Robertson Jacobian.
          Otherwise, we fall back to finite differences (or just don't pass jac).
          
       3. Use `solve_ivp` with `method='Radau'` and `jac=jac_robertson`.
       4. Set tight tolerances `rtol=1e-10, atol=1e-12`.
       
       This is a structural change: **Providing Analytic Jacobians** to the stiff solver.
    """
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Detect Robertson problem
    is_robertson = False
    if len(y0) == 3:
        # Check if y0 is approximately [1, 0, 0]
        if np.allclose(y0, [1.0, 0.0, 0.0], atol=1e-6):
            is_robertson = True
            
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
        
    # Define analytic Jacobian for Robertson
    def jac_robertson(t, y):
        y0_val, y1_val, y2_val = y[0], y[1], y[2]
        J = np.zeros((3, 3))
        J[0, 0] = -0.04
        J[0, 1] = 1e4 * y2_val
        J[0, 2] = 1e4 * y1_val
        
        J[1, 0] = 0.04
        J[1, 1] = -1e4 * y2_val - 0.62 * y1_val
        J[1, 2] = -1e4 * y1_val
        
        J[2, 0] = 0.0
        J[2, 1] = 0.62 * y1_val
        J[2, 2] = 0.0
        return J

    start_time = time.time()
    deadline = start_time + 280.0
    
    import scipy.integrate as integrate_scipy
    
    best_y = None
    
    # Try Radau with analytic Jacobian first
    if is_robertson:
        try:
            sol = integrate_scipy.solve_ivp(
                f_counted, 
                t_span, 
                y0, 
                method='Radau',
                jac=jac_robertson,
                rtol=1e-10, 
                atol=1e-12,
                max_step=1.0
            )
            if sol.success:
                best_y = sol.y[:, -1]
        except Exception:
            pass

    # If it failed or wasn't Robertson, try BDF with analytic Jacobian
    if best_y is None and is_robertson:
        try:
            sol = integrate_scipy.solve_ivp(
                f_counted, 
                t_span, 
                y0, 
                method='BDF',
                jac=jac_robertson,
                rtol=1e-10, 
                atol=1e-12,
                max_step=1.0
            )
            if sol.success:
                best_y = sol.y[:, -1]
        except Exception:
            pass

    # Fallback: Standard Radau without Jacobian (current behavior)
    if best_y is None:
        try:
            sol = integrate_scipy.solve_ivp(
                f_counted, 
                t_span, 
                y0, 
                method='Radau',
                rtol=1e-10, 
                atol=1e-12,
                max_step=1.0
            )
            if sol.success:
                best_y = sol.y[:, -1]
        except Exception:
            pass
            
    # Final Fallback: RK4
    if best_y is None:
        y = y0.copy()
        h = (t1 - t0) / 10000
        t = t0
        for _ in range(10000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
            t += h
            if time.time() > deadline:
                break
        best_y = y

    return best_y
# EVOLVE-BLOCK-END
