import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n_points = len(y0)
    
    # Choose N steps
    # We want to use as many steps as possible to minimize error
    # while staying within time and call limits.
    # Call limit: 2,000,000
    # Time limit: 300s
    # Let's start with N = 1,000,000
    N = 1_000_000
    
    h = (t1 - t0) / N
    
    # We need y0, y1, y2, y3 and f0, f1, f2, f3
    # Use RK4 for the first 3 steps to generate these
    
    def rk4_step(t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # Step 1: y0 -> y1
    t = t0
    y = y0
    f_prev3 = f(t, y) # f(t0, y0)
    
    y = rk4_step(t, y, h)
    t = t + h
    f_prev2 = f(t, y) # f(t1, y1)
    
    # Step 2: y1 -> y2
    y = rk4_step(t, y, h)
    t = t + h
    f_prev1 = f(t, y) # f(t2, y2)
    
    # Step 3: y2 -> y3
    y = rk4_step(t, y, h)
    t = t + h
    f_curr = f(t, y) # f(t3, y3)
    
    # Now we are at t3, y3.
    # We have f(t0), f(t1), f(t2), f(t3)
    # History: f_prev3, f_prev2, f_prev1, f_curr
    # Next step is to t4 using AB4
    
    # Loop from i=4 to N-1
    # We need to perform N-4 more steps
    for _ in range(N - 4):
        # AB4 formula:
        # y_{n+1} = y_n + h/24 * (55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})
        y = y + h/24.0 * (55.0 * f_curr - 59.0 * f_prev1 + 37.0 * f_prev2 - 9.0 * f_prev3)
        t = t + h
        f_new = f(t, y)
        
        # Shift history
        f_prev3 = f_prev2
        f_prev2 = f_prev1
        f_prev1 = f_curr
        f_curr = f_new
        
    return y
