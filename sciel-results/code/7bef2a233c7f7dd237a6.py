import numpy as np
from scipy.linalg import lu, solve

def integrate(f, t_span, y0):
    t0, t1 = t_span
    T = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    b = np.array(f(0.0, np.zeros(N)), dtype=float)
    
    A = np.zeros((N, N), dtype=float)
    for j in range(N):
        ej = np.zeros(N)
        ej[j] = 1.0
        col = np.array(f(t0, ej), dtype=float) - b
        A[:, j] = col
        
    X = A * T
    
    # (13,13) Pade coefficients
    a = np.array([1.0, 0.5, 0.04166666666666666, 0.002380952380952381, 0.00011111111111111112, 4.761904761904762e-06, 1.7857142857142858e-07, 5.952380952380952e-09, 1.7857142857142858e-10, 4.96031746031746e-12, 1.240079365079365e-13, 2.800180509275481e-15, 5.600361018550962e-17, 1.000064003372884e-18])
    b_coeff = np.array([1.0, 1.0769230769230769, 0.05952380952380952, 0.0022222222222222222, 6.349206349206349e-05, 1.4313431343134314e-06, 2.7027027027027025e-08, 4.504504504504504e-10, 6.435006435006435e-12, 8.164958164958164e-14, 9.552889552889552e-16, 1.005567316146316e-17, 9.57683158282158e-20, 8.211584358132785e-22])
    
    # Compute powers of X
    X_powers = [np.eye(N)]
    X_k = X
    for k in range(1, 14):
        X_powers.append(X_k)
        X_k = X_k @ X
        
    # Compute Q(X)
    Q = np.zeros((N, N))
    for k in range(14):
        Q += b_coeff[k] * X_powers[k]
        
    # Factorize Q
    P, L, U = lu(Q)
    
    # Compute P(X) y0
    P_y0 = np.zeros(N)
    for k in range(14):
        P_y0 += a[k] * (X_powers[k] @ y0)
        
    # Solve Q y_temp = P_y0
    # scipy.linalg.solve with lu factorization
    y_temp = solve((L, U), P_y0, trans=1) # Check trans
    # Actually, scipy.linalg.solve(A, b) doesn't take (L,U) directly in the same way.
    # Use scipy.linalg.solve_triangular
    from scipy.linalg import solve_triangular
    y_temp = solve_triangular(U, solve_triangular(L, P_y0, lower=True, trans=1), lower=False, trans=1)
    # Note: L is unit lower triangular.
    
    if np.allclose(b, 0.0, atol=1e-15):
        y_end = y_temp
    else:
        try:
            x = np.linalg.solve(A, b)
            P_x = np.zeros(N)
            for k in range(14):
                P_x += a[k] * (X_powers[k] @ x)
            y_ex_b = solve_triangular(U, solve_triangular(L, P_x, lower=True, trans=1), lower=False, trans=1)
            y_end = y_temp + y_ex_b - x
        except np.linalg.LinAlgError:
            y_end = y_temp + T * b
            
    return y_end
