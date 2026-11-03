import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Tensor T[i, j, k] = 1 if indices match, else 0
    # Dimensions: U (n*m, R), V (m*p, R), W (p*n, R)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Create the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    best_result = None
    best_rank = n * m * p # 36
    
    # Try ranks from low to high
    for rank in range(27, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this rank
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS iterations
            for it in range(500):
                if time.time() > deadline:
                    break
                
                # Solve for U
                # T[i, :, :] = U[i, :] @ (V @ W) ... no, T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # For fixed V, W, T[i, j, k] is linear in U[i, r].
                # T[i, :, :] (m*p, p*n) = U[i, :] (1, rank) * (V (m*p, rank) * W.T (rank, p*n))?
                # Actually, T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # Let M = V * W.T (m*p, p*n). M[j, k] = sum_r V[j, r] W[k, r]
                # Then T[i, j, k] = sum_r U[i, r] M[j, k] is NOT correct.
                # It is T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
                # This is a bilinear form.
                # For fixed V, W, the equation for U[i, :] is:
                # sum_r U[i, r] (V[:, r] * W[:, r]) = T[i, :, :]
                # Let A[i, j, k] = V[j, r] * W[k, r] for a specific r? No.
                # We need to solve for U[i, r].
                # Reshape T[i] to vector of length (m*p)*(p*n).
                # The coefficient for U[i, r] is the outer product of V[:, r] and W[:, r].
                # Let K_r = V[:, r] * W[:, r] (element-wise, shape m*p, p*n).
                # Then T[i, j, k] = sum_r U[i, r] K_r[j, k].
                # This is a linear system for U[i, r].
                
                # Construct design matrix for U
                # For each i, we have (m*p)*(p*n) equations.
                # Variables are U[i, 0..rank-1].
                # Eq: sum_r U[i, r] * (V[j, r] * W[k, r]) = T[i, j, k]
                
                # This is expensive to form explicitly for every i if we do it naively.
                # But dimensions are small: 9, 12, 12.
                
                # Solve for U
                for i in range(n*m):
                    # Form matrix A_u of shape (m*p*p*n, rank)
                    # A_u[j*p+k, r] = V[j, r] * W[k, r]
                    A_u = np.zeros((m*p*p*n, rank))
                    for j in range(m):
                        for k in range(p):
                            for r in range(rank):
                                A_u[j*p+k, r] = V[j, r] * W[k, r]
                    
                    b_u = T[i].flatten()
                    # Solve least squares
                    U[i], _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                
                # Solve for V
                for j in range(m*p):
                    A_v = np.zeros((n*m*p*n, rank))
                    for i in range(n):
                        for k in range(p):
                            for r in range(rank):
                                A_v[i*p*n+k, r] = U[i*m + (j//p), r] * W[k*n + (i%m), r] # Wait, indices are tricky
                    # Index mapping:
                    # T index: i, j, k
                    # U index: i*m + j' where j' is the j from A?
                    # In T[i, j, k], j is the column of A (which is row of B).
                    # U[i*m + j], V[j*p + k], W[k*n + i].
                    # So for fixed j (in V), we vary i and k.
                    # U index: i*m + j
                    # W index: k*n + i
                    # T index: i, j, k
                    for i in range(n):
                        for k in range(p):
                            for r in range(rank):
                                A_v[i*p*n+k, r] = U[i*m + j, r] * W[k*n + i, r]
                    
                    b_v = T[:, j, :].flatten()
                    V[j], _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                
                # Solve for W
                for k in range(p*n):
                    A_w = np.zeros((n*m*m*p, rank))
                    # Fixed k. Vary i, j.
                    # V index: j*p + k
                    # U index: i*m + j
                    # T index: i, j, k
                    for i in range(n):
                        for j in range(m):
                            for r in range(rank):
                                A_w[i*m*p+j, r] = U[i*m + j, r] * V[j*p + k, r]
                    
                    b_w = T[:, :, k].flatten()
                    W[k], _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                
                # Check convergence / validity
                # Reconstruct and check error
                if it % 10 == 0:
                    # Check if integer/half-integer
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify
                    T_recon = np.zeros((n*m, m*p, p*n))
                    for r in range(rank):
                        T_recon += np.einsum('i,j,k->ijk', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                    
                    if np.allclose(T_recon, T, atol=1e-9):
                        # Found valid decomposition
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
            
            # If not found in this attempt, continue to next attempt/rank

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
