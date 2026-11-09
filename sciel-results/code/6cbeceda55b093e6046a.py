import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Try ranks from low to high
    for rank in [20, 21, 22, 23, 24]:
        if time.time() - start_time > deadline - 5:
            break
        result = _find_decomposition(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _find_decomposition(n, m, p, rank, deadline):
    """Try to find a rank-r decomposition."""
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Multiple random restarts
    for restart in range(50):
        if time.time() > deadline - 2:
            break
        
        np.random.seed(restart)
        U = np.random.randn(n*m, rank) * 0.3
        V = np.random.randn(m*p, rank) * 0.3
        W = np.random.randn(p*n, rank) * 0.3
        
        # ALS iterations
        for it in range(200):
            if time.time() > deadline - 1:
                break
            
            # Update U: for each (i,j), solve linear system
            # Σ_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = target[i*m+j, j'*p+k, k'*n+i']
            # For fixed (i,j): Σ_r U[i*m+j, r] * A[r, :] = b where A[r, (j',k,i',k')] = V[j'*p+k, r]*W[k'*n+i', r]
            
            # This is a least squares problem for each (i,j)
            # A is rank × (m*p*p*n)
            # b is (m*p*p*n,)
            
            A = np.einsum('jr,kr->rjk', V, W)  # shape (rank, m*p, p*n)
            A_flat = A.reshape(rank, -1)  # shape (rank, m*p*p*n)
            
            for ij in range(n*m):
                b = target[ij].reshape(-1)
                # Solve A_flat.T @ x = b
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_flat.T, b, rcond=None)
                    U[ij] = sol
                except:
                    pass
            
            # Update V similarly
            A = np.einsum('ir,kr->rik', U, W)  # shape (n*m, rank, p*n)
            A_flat = A.reshape(-1, rank, p*n).transpose(1, 0, 2).reshape(rank, -1)
            # Hmm, this is getting messy. Let me use a different indexing.
            
            # For V: Σ_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = target[...]
            # For fixed (j',k): Σ_r V[j'*p+k, r] * (U[i*m+j, r] * W[k'*n+i', r]) = target[i*m+j, j'*p+k, k'*n+i']
            # A[r, (i,j,i',k')] = U[i*m+j, r] * W[k'*n+i', r]
            # b[(i,j,i',k')] = target[i*m+j, j'*p+k, k'*n+i']
            
            A = np.einsum('ir,kr->rik', U, W)  # (n*m, rank, p*n)
            # Reshape to (rank, n*m, p*n)
            A = A.transpose(1, 0, 2).reshape(rank, -1)
            
            for jk in range(m*p):
                b = target[:, jk, :].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    V[jk] = sol
                except:
                    pass
            
            # Update W similarly
            A = np.einsum('ir,jr->rij', U, V)  # (n*m, rank, m*p)
            A = A.transpose(1, 0, 2).reshape(rank, -1)
            
            for kn in range(p*n):
                b = target[:, :, kn].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    W[kn] = sol
                except:
                    pass
            
            # Check convergence
            current = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.sum((current - target)**2)
            if err < 1e-10:
                break
        
        # Final check
        current = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.sum((current - target)**2)
        if err < 1e-8:
            # Round
            def round_mat(M):
                M_r = M.copy()
                for i in range(M_r.shape[0]):
                    for j in range(M_r.shape[1]):
                        r = round(M_r[i,j])
                        if abs(M_r[i,j] - r) < 1e-3:
                            M_r[i,j] = float(r)
                        else:
                            r2 = round(M_r[i,j]*2)/2
                            if abs(M_r[i,j] - r2) < 1e-3:
                                M_r[i,j] = r2
                return M_r
            
            U_r = round_mat(U)
            V_r = round_mat(V)
            W_r = round_mat(W)
            
            check = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.sum((check - target)**2) < 1e-6:
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
    
    return None
