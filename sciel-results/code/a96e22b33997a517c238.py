import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Bulirsch-Stoer sequence of N values
    # Start with small N, increase until convergence
    Ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    # Table for extrapolation
    # Y[i][j] is the extrapolated value using N_i and previous
    # Standard BS uses a 2D table.
    
    results = []
    N_prev = 0
    y_prev = None
    
    # We need at least 2 results to start extrapolating?
    # Actually, we can just store y_h for each h.
    
    y_vals = []
    N_vals = []
    
    for N in Ns:
        h = (t1 - t0) / N
        t = t0
        y_curr = y.copy()
        
        # Perform N steps of RK4
        for _ in range(N):
            k1 = f(t, y_curr)
            k2 = f(t + h/2, y_curr + h/2 * k1)
            k3 = f(t + h/2, y_curr + h/2 * k2)
            k4 = f(t + h, y_curr + h * k3)
            y_curr = y_curr + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            
        y_vals.append(y_curr)
        N_vals.append(N)
        
        # Check convergence if we have at least 3 points
        if len(y_vals) >= 3:
            # Simple Richardson Extrapolation on the last 3 points
            # y(h), y(h/2), y(h/4)
            # e ~ h^4
            # y_ext = y(h/4) + (y(h/4) - y(h/2)) / ( (h/2)/(h/4) )^4 - 1 )
            # Ratio is 2. 2^4 - 1 = 15.
            # y_ext = y_last + (y_last - y_prev) / 15
            y_ext = y_vals[-1] + (y_vals[-1] - y_vals[-2]) / 15.0
            
            # Check difference between this extrapolated value and the previous extrapolated value?
            # Or just check if y_ext is stable?
            # Better: Extrapolate using the last 2 RK4 results to get a 4th order estimate,
            # and compare with the previous 4th order estimate?
            # Actually, standard BS builds a table.
            # Let's just check if the change in y_ext is small compared to the change in y_raw.
            
            # Let's compute the previous extrapolated value (using y_vals[-2] and y_vals[-3])
            y_ext_prev = y_vals[-2] + (y_vals[-2] - y_vals[-3]) / 15.0
            
            diff = np.linalg.norm(y_ext - y_ext_prev)
            tol = 1e-14 * np.linalg.norm(y_ext)
            
            if diff < tol:
                return y_ext

    # If we didn't converge, return the last extrapolated value
    if len(y_vals) >= 2:
        return y_vals[-1] + (y_vals[-1] - y_vals[-2]) / 15.0
    else:
        return y_vals[-1]
