def _compute_error_and_grad(U, V, W, n, m, p, rank):
    """Compute error tensor and gradients efficiently using vectorization."""
    # Reshape to 3D: (n, m, r), (m, p, r), (p, n, r)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Compute sum over rank c: T[i, j, jp, k] = sum_c U[i, j, c] V[jp, k, c] W[k, i, c]
    # Indices: U: i,j,c ; V: jp,k,c ; W: k,i,c
    # Output: i, j, jp, k
    T = np.einsum('ijc,jkc,kic->ijjk', U_3d, V_3d, W_3d, optimize=True)
    
    # Construct identity tensor I[i, j, jp, k] = 1 if j==jp else 0
    # Create a mask for j==jp
    # j ranges 0..m-1, jp ranges 0..m-1
    identity_mask = np.eye(m, dtype=float) # shape (m, m)
    # Broadcast to (n, m, m, p)
    # I[i, j, jp, k] depends only on j, jp
    I = np.broadcast_to(identity_mask[None, :, :, None], (n, m, m, p)).copy()
    
    err = T - I
    
    # Gradients
    # dU[i, j, c] = sum_{jp, k} err[i, j, jp, k] V[jp, k, c] W[k, i, c]
    # dU = einsum('ijjk,jkc,kic->ijc', err, V_3d, W_3d)
    dU = np.einsum('ijjk,jkc,kic->ijc', err, V_3d, W_3d, optimize=True)
    
    # dV[jp, k, c] = sum_{i, j} err[i, j, jp, k] U[i, j, c] W[k, i, c]
    # dV = einsum('ijjk,ijc,kic->jkc', err, U_3d, W_3d)
    dV = np.einsum('ijjk,ijc,kic->jkc', err, U_3d, W_3d, optimize=True)
    
    # dW[k, i, c] = sum_{i, j, jp} err[i, j, jp, k] U[i, j, c] V[jp, k, c]
    # Note: sum over i, j, jp. But err has i, j, jp, k.
    # dW[k, i, c] = sum_{j, jp} err[i, j, jp, k] U[i, j, c] V[jp, k, c]
    # dW = einsum('ijjk,ijc,jkc->kic', err, U_3d, V_3d)
    dW = np.einsum('ijjk,ijc,jkc->kic', err, U_3d, V_3d, optimize=True)
    
    return err, dU.reshape(n*m, rank), dV.reshape(m*p, rank), dW.reshape(p*n, rank)
