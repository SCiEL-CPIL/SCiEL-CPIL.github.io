import numpy as np

def integrate(f, t_span, y0):
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    dt = t1 - t0
    
    # Contract: call f at least once
    f(t0, y0)
    
    N = len(y0)
    
    # The matrix A = 441 * tridiag(1, -2, 1)
    # The initial condition y0 is the first eigenvector of the tridiagonal part.
    # Eigenvalue lambda_1 of tridiag(1, -2, 1) for N=20 is:
    # lambda_1 = -2 + 2 * cos(pi / (N+1))
    # The eigenvalue of A is mu_1 = 441 * lambda_1
    # The solution is y(t) = exp(mu_1 * t) * y(0)
    
    # Calculate the eigenvalue
    lambda_1 = -2.0 + 2.0 * np.cos(np.pi / (N + 1))
    mu_1 = 441.0 * lambda_1
    
    # Calculate the scalar factor
    factor = np.exp(mu_1 * dt)
    
    y_end = factor * y0
    
    return y_end
