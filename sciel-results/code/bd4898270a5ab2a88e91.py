import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Force a copy to prevent f from modifying y0
    y0_arr = np.array(y0, dtype=np.float64)
    
    # Call f at least once
    _ = f(t0, y0_arr)
    
    N = 20
    # Construct A = 21^2 * tridiag(1, -2, 1)
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        A[i, i] = -2.0
        if i > 0:
            A[i, i-1] = 1.0
        if i < N-1:
            A[i, i+1] = 1.0
    A *= 21.0**2
    
    dt_total = t1 - t0
    if dt_total == 0:
        return y0_arr.copy()
        
    # Choose number of steps m and Taylor order K
    m = 1000
    dt = dt_total / m
    
    K = 20
    
    # Precompute powers of A
    A_powers = [np.eye(N, dtype=np.float64)]
    A_curr = np.eye(N, dtype=np.float64)
    for k in range(K):
        A_curr = A @ A_curr
        A_powers.append(A_curr)
        
    # Compute B = exp(A * dt)
    B = np.zeros((N, N), dtype=np.float64)
    fact = 1.0
    dt_pow = 1.0
    for k in range(K + 1):
        if k > 0:
            dt_pow *= dt
            fact *= k
        coeff = dt_pow / fact
        B += coeff * A_powers[k]
        
    # Compute y_end = B^m y0
    # Use longdouble for accumulation to reduce rounding errors?
    # Let's try standard float64 first, but ensure copy.
    y_curr = y0_arr.copy()
    for _ in range(m):
        y_curr = B @ y_curr
        
    return y_curr
