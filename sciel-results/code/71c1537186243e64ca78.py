<<<<<<< SEARCH
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The decomposition satisfies: for all i,j,k,
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check each (i, j, k) triple
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
=======
def _is_valid_half_integer_solution(U, V, W, n, m, p, tol=1e-6):
    """
    Check if U, V, W can be rounded to half-integers and still satisfy the equation.
    Returns (True, U_int, V_int, W_int) if valid, else (False, None, None, None).
    """
    # Round to nearest half-integer: round(2*x)/2
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Verify with tight tolerance since entries are now discrete
    rank = U_r.shape[1]
    U_r = np.array(U_r, dtype=float)
    V_r = np.array(V_r, dtype=float)
    W_r = np.array(W_r, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False, None, None, None
    
    # Convert to lists of floats (which are 0, 0.5, 1, 1.5, etc.)
    return True, U_r.tolist(), V_r.tolist(), W_r.tolist()
>>>>>>> REPLACE

