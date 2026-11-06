import numpy as np
import scipy.linalg

def integrate(f, t_span, y0):
       t0, t1 = t_span
       dt_total = t1 - t0
       y0 = np.array(y0, dtype=float)
       n = len(y0)
       
       # Extract matrix A assuming f(t, y) = A y
       A = np.zeros((n, n), dtype=float)
       for j in range(n):
           e_j = np.zeros(n, dtype=float)
           e_j[j] = 1.0
           A[:, j] = f(0.0, e_j)
           
       # Use Padé [1/1] approximation with sub-stepping
       # e^{At} approx (I - At/2)^{-1} (I + At/2)
       # Step size dt
       N = 100
       dt = dt_total / N
       
       # Precompute matrices for one step
       # X_{k+1} = (I - dt*A/2)^{-1} (I + dt*A/2) X_k
       B = np.eye(n) - (dt / 2.0) * A
       C = np.eye(n) + (dt / 2.0) * A
       
       # Factorize B
       lu, piv = scipy.linalg.lu_factor(B)
       
       y_curr = y0.copy()
       for _ in range(N):
           # y_next = B^{-1} C y_curr
           # First compute C @ y_curr
           y_temp = C @ y_curr
           # Then solve B @ y_next = y_temp
           y_curr = scipy.linalg.lu_solve((lu, piv), y_temp)
           
       return y_curr
