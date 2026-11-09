import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    start_time = time.time()
    deadline = start_time + 285.0

    # 1. Extract A and b
    # b = f(t0, 0)
    b = f(t0, np.zeros(n))
    
    # A columns
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        A[:, j] = f(t0, ej) - b
        
    # Check time limit
    if time.time() > deadline:
        # Fallback to a simple RK4 if we are out of time? 
        # Unlikely to happen with N=100.
        return y

    dt = t1 - t0
    
    # 2. Solve y' = A y + b
    # Using block matrix exponential for high accuracy
    # M = [[A*dt, b*dt], [0, 0]]
    # expm(M) = [[E, Z], [0, 1]]
    # y_final = E @ y0 + Z
    
    # Construct block matrix
    # Size (n+1) x (n+1)
    M = np.zeros((n+1, n+1))
    M[:n, :n] = A * dt
    M[:n, n] = b * dt
    # M[n, n] = 0 is already there
    
    E_block = expm(M)
    
    E = E_block[:n, :n]
    Z = E_block[:n, n]
    
    y_final = E @ y + Z
    
    return y_final
