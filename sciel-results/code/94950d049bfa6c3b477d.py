def _is_valid_strict(U, V, W, tol=1e-9):
    """Check if U, V, W (numpy arrays) form a valid decomposition."""
    rank = U.shape[1]
    # Efficient check using einsum or loops
    # T[i,j,k] = sum_r U[i*M+j, r] V[j*P+k, r] W[k*N+i, r]
    # We need T[i,j,k] == 1 for all i,j,k
    
    # Use broadcasting/einsum for speed
    # U: (N*M, R)
    # V: (M*P, R)
    # W: (P*N, R)
    
    # Reshape U to (N, M, R)
    U_reshaped = U.reshape(N, M, rank)
    # Reshape V to (M, P, R)
    V_reshaped = V.reshape(M, P, rank)
    # Reshape W to (P, N, R)
    W_reshaped = W.reshape(P, N, rank)
    
    # We want T[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
    # This is a bit tricky with einsum because of the index permutations.
    # T[i, j, k] = np.einsum('ijr,jkr,kr->ijk', U_reshaped, V_reshaped, W_reshaped) 
    # Wait, W index is (k, i, r). So W_reshaped[k, i, r].
    # einsum: 'ijr,jkr,kir->ijk'
    
    try:
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
        
    # Check if T is all 1s
    return np.allclose(T, 1.0, atol=tol)
