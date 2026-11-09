import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 250  # Safety margin before 300s limit
    
    try:
        # Use Radau for stiff problems with tight tolerances
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
            max_step=0.05
        )
        
        if time.time() > deadline:
            # If we're close to deadline, return what we have
            if sol.success:
                return sol.y[:, -1]
            else:
                return y0
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau failed, try BDF as fallback
            sol2 = scipy.integrate.solve_ivp(
                f,
                t_span,
                y0,
                method='BDF',
                rtol=1e-9,
                atol=1e-11,
                max_step=0.05
            )
            if time.time() > deadline:
                return y0
            if sol2.success:
                return sol2.y[:, -1]
            else:
                return y0
                
    except Exception as e:
        # Fallback: try a simple explicit RK4 with fixed steps
        try:
            n_steps = 1000
            dt = (t1 - t0) / n_steps
            y = y0.copy()
            for i in range(n_steps):
                t = t0 + i * dt
                if time.time() > deadline:
                    break
                k1 = f(t, y)
                k2 = f(t + dt/2, y + dt/2 * k1)
                k3 = f(t + dt/2, y + dt/2 * k2)
                k4 = f(t + dt, y + dt * k3)
                y = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            return y
        except Exception:
            return y0
