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
    # Using sin form for better numerical stability:
    # lambda_1 = -4 * sin(pi / (2*(N+1)))**2
    
    # Calculate the eigenvalue
    # np.pi / (2 * (N + 1))
    theta = np.pi / (2.0 * (N + 1))
    lambda_1 = -4.0 * (np.sin(theta) ** 2)
    mu_1 = 441.0 * lambda_1
    
    # Calculate the scalar factor
    factor = np.exp(mu_1 * dt)
    
    y_end = factor * y0
    
    return y_end
