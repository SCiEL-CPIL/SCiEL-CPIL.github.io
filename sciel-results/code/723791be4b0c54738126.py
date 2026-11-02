import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    # Target Tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    time_limit = 145.0 # Safety margin

    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    best_rank = rank_schoolbook

    # Pre-compute permutations/reshapes for fast updates
    # T is (n*m, m*p, p*n)
    
    for r in range(rank_schoolbook - 1, 20, -1):
        if time.time() - start_time > time_limit:
            break
            
        for restart in range(30):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            err = float('inf')
            converged = False
            
            for it in range(200):
                if time.time() - start_time > time_limit:
                    break
                    
                # 1. Update U
                # Solve U @ (V*W).T = T
                # A_U is (m*p, r) where A_U[j, c] = V[j, c] * W[? , c]
                # Wait, the term is V[j, c] * W[k, c].
                # For fixed i, we sum over j, k.
                # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # Let M_U be (m*p, p*n) such that M_U[j, k] = V[j, c] W[k, c]? No, c is summed.
                # We need to form a matrix A_U of shape (m*p, r) ?
                # No, the basis for U[i, c] is the vector v_c = V[:, c] * W[:, c] ?
                # No, V is (m*p, r), W is (p*n, r).
                # The product V[j, c] * W[k, c] is an element of a matrix of size (m*p, p*n) for each c.
                # Let B_c = np.outer(V[:, c], W[:, c]) -> shape (m*p, p*n).
                # We want sum_c U[i, c] B_c = T[i, :, :]
                # Vectorize B_c to length m*p * p*n.
                # Stack all B_c to form A_U of shape (m*p * p*n, r).
                # Then we want U[i, :] @ A_U.T = T[i, :, :].flatten()
                # So we solve A_U @ U.T = T_flat.T
                
                # Construct A_U
                # A_U[:, c] = flatten(outer(V[:, c], W[:, c]))
                A_U = np.zeros((m*p * p*n, r))
                for c in range(r):
                    A_U[:, c] = np.outer(V[:, c], W[:, c]).flatten()
                
                T_flat = T.reshape(n*m, m*p * p*n)
                # Solve A_U @ U.T = T_flat.T
                try:
                    U_new, residuals, rank, sv = np.linalg.lstsq(A_U, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass

                # 2. Update V
                # Term: V[j, c] * U[i, c] * W[k, c]
                # Basis for V[j, c] is outer(U[:, c], W[:, c])?
                # For fixed j, we vary i, k.
                # T[:, j, :] = sum_c V[j, c] outer(U[:, c], W[:, c])
                # Vectorize outer to length n*m * p*n.
                A_V = np.zeros((n*m * p*n, r))
                for c in range(r):
                    A_V[:, c] = np.outer(U[:, c], W[:, c]).flatten()
                
                # T needs to be permuted to (m*p, n*m, p*n) then flattened
                T_perm_V = np.transpose(T, (1, 0, 2))
                T_flat_V = T_perm_V.reshape(m*p, n*m * p*n)
                
                try:
                    V_new, residuals, rank, sv = np.linalg.lstsq(A_V, T_flat_V.T, rcond=None)
                    V = V_new.T
                except:
                    pass

                # 3. Update W
                # Term: W[k, c] * U[i, c] * V[j, c]
                # Basis for W[k, c] is outer(U[:, c], V[:, c])
                A_W = np.zeros((n*m * m*p, r))
                for c in range(r):
                    A_W[:, c] = np.outer(U[:, c], V[:, c]).flatten()
                    
                T_perm_W = np.transpose(T, (2, 0, 1))
                T_flat_W = T_perm_W.reshape(p*n, n*m * m*p)
                
                try:
                    W_new, residuals, rank, sv = np.linalg.lstsq(A_W, T_flat_W.T, rcond=None)
                    W = W_new.T
                except:
                    pass

                # Check convergence
                T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
                err = np.linalg.norm(T - T_rec)
                if err < 1e-9:
                    converged = True
                    break
            
            if converged:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                check_sum = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(check_sum, T, atol=1e-9):
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    best_rank = r
                    break # Found a solution for this rank, try lower
        
    return {"U": best_U, "V": best_V, "W": best_W}
