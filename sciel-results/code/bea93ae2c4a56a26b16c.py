def _als_step_U(T_nz, V, W, rank):
    """Solve for U given V, W. T_nz is list of (a,b,c) for non-zero entries."""
    n_m = V.shape[0] // V.shape[1] * 0  # not needed
    # U shape: (n*m, rank)
    # For each non-zero (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 1
    # Group by a
    from collections import defaultdict
    groups = defaultdict(list)
    for a, b, c in T_nz:
        groups[a].append((b, c))
    
    U = np.zeros((max(a for a,b,c in T_nz)+1, rank))
    for a, eqs in groups.items():
        # For this a, we have len(eqs) equations
        # Each eq: U[a,:] @ (V[b,:]*W[c,:]) = 1
        # Matrix A: (len(eqs), rank), A[i, r] = V[b_i, r] * W[c_i, r]
        # Solve A @ U[a,:]^T = 1
        A = np.array([V[b, :] * W[c, :] for b, c in eqs])
        rhs = np.ones(len(eqs))
        # Minimum norm solution
        sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
        U[a, :] = sol
    return U
