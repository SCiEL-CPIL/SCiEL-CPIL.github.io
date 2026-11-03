def _solve_W_linear(U, V, n, m, p, R):
    """Given U and V, solve for W such that the decomposition is exact.
    For each (i,k), solve A_{ik} @ W_{ik} = 1 where A_{ik}[l,r] = U[i*m+l, r] * V[l*p+k, r]
    Returns W or None if no solution exists.
    """
    W = np.zeros((p * n, R), dtype=float)
    for i in range(n):
        for k in range(p):
            # Build A_{ik}: m x R matrix
            A = np.zeros((m, R))
            for l in range(m):
                A[l, :] = U[i * m + l, :] * V[l * p + k, :]
            # Solve A @ W_{ik} = 1
            # W_{ik} is a vector of length R
            try:
                # Use least squares
                W_ik, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                # Check if solution is exact
                if np.max(np.abs(A @ W_ik - 1.0)) > 1e-9:
                    return None
                W[k * n + i, :] = W_ik
            except np.linalg.LinAlgError:
                return None
    return W
