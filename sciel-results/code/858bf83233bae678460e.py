import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Initial step size
    h = (t1 - t0) / 100.0
    h_min = (t1 - t0) * 1e-12
    h_max = (t1 - t0)
    
    # Tolerances matching reference
    rtol = 1e-11
    atol = 1e-13
    
    # Safety factor
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0
    
    def rk45_step(t, y, h):
        # k1
        k1 = f(t, y)
        # k2
        k2 = f(t + h/5.0, y + h/5.0 * k1)
        # k3
        k3 = f(t + 3.0*h/10.0, y + h * (3.0/40.0 * k1 + 9.0/40.0 * k2))
        # k4
        k4 = f(t + 4.0*h/5.0, y + h * (-44.0/45.0 * k1 + 56.0/15.0 * k2 - 32.0/9.0 * k3))
        # k5
        k5 = f(t + 8.0*h/9.0, y + h * (4.0/5.0 * k1 - 16.0/15.0 * k2 + 7.0/12.0 * k3))
        # k6
        k6 = f(t + h, y + h * (-8.0/5.0 * k1 + 1.0 * k2 - 12.0/13.0 * k3 + 1.0 * k4))
        # k7
        k7 = f(t + h, y + h * (361.0/540.0 * k1 - 3.0/5.0 * k2 + 16.0/15.0 * k3 - 5.0 * k4 + 0.25 * k5))
        
        # 5th order solution
        y5 = y + h * (361.0/540.0 * k1 - 3.0/5.0 * k2 + 16.0/15.0 * k3 - 5.0 * k4 + 0.25 * k5 + 1.0/42.0 * k6)
        
        # 4th order solution
        y4 = y + h * (35.0/384.0 * k1 + 0.0 * k2 + 500.0/1113.0 * k3 + 125.0/192.0 * k4 - 2187.0/6784.0 * k5 + 11.0/84.0 * k6 - 1.0/40.0 * k7)
        
        err = y5 - y4
        return y5, err

    start_time = time.time()
    deadline = start_time + 290.0
    max_calls = 2000000
    calls = 0
    
    # Ensure at least one call if t_span is tiny or zero? 
    # If t0 == t1, return y0. But f must be called at least once?
    # "f must be called at least once"
    # If t0 == t1, we should call f(t0, y0) once and return y0?
    # The task says t in [0, 2], so t0 < t1.
    
    while t < t1:
        if time.time() > deadline:
            break
            
        if t + h > t1:
            h = t1 - t
            
        y_new, err = rk45_step(t, y, h)
        calls += 6
        
        if calls > max_calls:
            break
            
        err_norm = np.linalg.norm(err)
        scale = np.linalg.norm(rtol * np.abs(y) + atol)
        
        if err_norm <= scale:
            y = y_new
            t += h
            if err_norm < 1e-15:
                factor = max_factor
            else:
                factor = safety * (err_norm / scale) ** (-0.2)
            h = h * np.clip(factor, min_factor, max_factor)
            h = min(h, h_max)
        else:
            factor = safety * (err_norm / scale) ** (-0.2)
            h = h * np.clip(factor, min_factor, max_factor)
            h = max(h, h_min)
            if h < h_min:
                break

    return y
