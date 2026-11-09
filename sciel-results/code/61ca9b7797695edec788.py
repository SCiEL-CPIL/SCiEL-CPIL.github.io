import numpy as np

def _compute_pade_coeffs(m):
    """
    Computes coefficients for the (m,m) Pade approximant of exp(z).
    Returns arrays a (for P) and b (for Q), where P(z) = sum a_k z^k, Q(z) = sum b_k z^k.
    b[0] is 1.
    """
    # Unknowns: a[0]...a[m], b[1]...b[m]
    # b[0] = 1
    n_unknowns = 2 * m + 1
    A_mat = np.zeros((n_unknowns, n_unknowns))
    B_vec = np.zeros(n_unknowns)
    
    # Indices for unknowns:
    # 0 .. m : a_0 .. a_m
    # m+1 .. 2m : b_1 .. b_m
    
    # Equations 1: k = 0 .. m
    # a_k - sum_{i=0}^k b_i / (k-i)! = 0
    # a_k = sum_{i=0}^k b_i / (k-i)!
    # a_k - sum_{i=1}^k b_i / (k-i)! = b_0 / k! = 1/k!
    for k in range(m + 1):
        row = k
        A_mat[row, k] = 1.0 # a_k
        for i in range(1, k + 1):
            # b_i is at index m + i
            # term: - b_i / (k-i)!
            A_mat[row, m + i] = -1.0 / np.math.factorial(k - i)
        B_vec[row] = 1.0 / np.math.factorial(k)
        
    # Equations 2: k = m+1 .. 2m
    # sum_{i=0}^m b_i / (k-i)! = 0
    # sum_{i=1}^m b_i / (k-i)! = - b_0 / k! = -1/k!
    for k in range(m + 1, 2 * m + 1):
        row = m + (k - (m + 1)) # row index m .. 2m
        # Wait, rows are 0..2m.
        # Equations 1 occupy rows 0..m.
        # Equations 2 occupy rows m+1 .. 2m.
        row = k # This is simpler if we map k directly?
        # Let's stick to 0-indexed rows.
        # Eq 1: k=0..m -> rows 0..m
        # Eq 2: k=m+1..2m -> rows m+1..2m
        row = k
        
        for i in range(1, m + 1):
            A_mat[row, m + i] = 1.0 / np.math.factorial(k - i)
        B_vec[row] = -1.0 / np.math.factorial(k)
        
    coeffs = np.linalg.solve(A_mat, B_vec)
    a = coeffs[:m + 1]
    b = np.zeros(m + 1)
    b[0] = 1.0
    b[1:] = coeffs[m + 1:]
    
    return a, b

def _pade_13_13(Z, a, b):
    m = 13
    I = np.eye(Z.shape[0])
    P = np.zeros_like(Z)
    Q = np.zeros_like(Z)
    
    Z_power = I.copy()
    for k in range(m + 1):
        P += a[k] * Z_power
        Q += b[k] * Z_power
        if k < m:
            Z_power = Z_power @ Z
            
    # Solve Q X = P
    R = np.linalg.solve(Q, P)
    return R

# Precompute coefficients
_a_13, _b_13 = _compute_pade_coeffs(13)

def integrate(f, t_span, y0):
    t0, t1 = t_span
    T = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    b_vec = np.array(f(0.0, np.zeros(N)), dtype=float)
    
    A = np.zeros((N, N), dtype=float)
    for j in range(N):
        ej = np.zeros(N)
        ej[j] = 1.0
        col = np.array(f(t0, ej), dtype=float) - b_vec
        A[:, j] = col
        
    Z = A * T
    
    M = _pade_13_13(Z, _a_13, _b_13)
    
    if np.allclose(b_vec, 0.0, atol=1e-15):
        y_end = M @ y0
    else:
        try:
            x = np.linalg.solve(A, b_vec)
            y_end = M @ y0 + (M - np.eye(N)) @ x
        except np.linalg.LinAlgError:
            y_end = M @ y0 + T * b_vec
            
    return y_end
