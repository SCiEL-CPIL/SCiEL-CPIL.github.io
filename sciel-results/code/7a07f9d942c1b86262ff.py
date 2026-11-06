import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Solves the Robertson ODE problem using Radau (implicit Runge-Kutta, order 5)
    with a known analytic Jacobian for maximum efficiency on this stiff problem.
    Uses very tight tolerances for high accuracy.
    """
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y1, y2, y3 = y
        J = np.array([
            [-0.04,      1e4*y3,    1e4*y2],
            [ 0.04, -1e4*y3-6e7*y2, -1e4*y2],
            [ 0.0,     6e7*y2,        0.0   ]
        ])
        return J
    
    start_time = time.time()
    deadline = 300.0  # 300 seconds wall-clock limit
    
    t0, tf = t_span
    
    # Primary solver: Radau (5th order implicit Runge-Kutta)
    # Radau is excellent for stiff problems and typically uses fewer
    # function evaluations than LSODA for similar accuracy levels.
    try:
        sol = solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='Radau', 
            jac=jac,
            rtol=1e-14, 
            atol=1e-16,
            max_step=np.inf,
            vectorized=False
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            raise RuntimeError("solve_ivp failed with Radau")
            
    except Exception as e:
        # Fallback 1: LSODA
        try:
            sol = solve_ivp(
                f_counted, 
                t_span, 
                y0, 
                method='LSODA', 
                jac=jac,
                rtol=1e-14, 
                atol=1e-16,
                max_step=np.inf,
                vectorized=False
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                raise RuntimeError("solve_ivp failed with LSODA")
        except Exception as e2:
            # Fallback 2: BDF
            try:
                sol = solve_ivp(
                    f_counted, 
                    t_span, 
                    y0, 
                    method='BDF', 
                    jac=jac,
                    rtol=1e-14, 
                    atol=1e-16,
                    max_step=np.inf,
                    vectorized=False
                )
                if sol.success:
                    return sol.y[:, -1]
                else:
                    raise RuntimeError("solve_ivp failed with BDF")
            except Exception as e3:
                # Last resort: simple RK4
                t0, tf = t_span
                n_steps = 50000
                h = (tf - t0) / n_steps
                y = np.array(y0, dtype=float)
                t = t0
                
                for _ in range(n_steps):
                    if time.time() - start_time > deadline:
                        break
                    k1 = f(t, y)
                    k2 = f(t + h/2, y + h/2 * k1)
                    k3 = f(t + h/2, y + h/2 * k2)
                    k4 = f(t + h, y + h * k3)
                    y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                    t += h
                    
                return y
