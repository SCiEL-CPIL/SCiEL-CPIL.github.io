import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    start_time = time.time()
    max_time = 110.0 # Safety margin
    
    # Try ranks from 25 up to 30
    # Start with 25 as it is the known minimum
    for rank in range(25, n*m*p + 1):
        if time.time() - start_time > max_time:
            break
            
        # Try a few random initializations for this rank
        found = False
        for attempt in range(20):
            if time.time() - start_time > max_time:
                break
                
            # Initialize U, V, W
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # ALS iterations
            for it in range(100):
                # Solve for U
                # B[j, k, c] = V[j, c] * W[k, c]
                # We want sum_c U[i, c] * V[j, c] * W[k, c] = 1
                # For each i, solve U[i, :] * (V * W)^T = ones(m*p)
                # Let M_U be matrix of size (m*p, rank) where M_U[j*p+k, c] = V[j, c] * W[k, c]
                M_U = np.zeros((m*p, rank))
                for j in range(m):
                    for k in range(p):
                        M_U[j*p+k, :] = V[j, :] * W[k, :]
                
                # For each i, solve U[i, :] @ M_U.T = ones(m*p)
                # This is equivalent to M_U @ U[i, :].T = ones(m*p)
                # Let's solve M_U.T @ U[i, :].T = ones(m*p) ? No.
                # U[i, c] * M_U[j*p+k, c] summed over c is 1.
                # This is (M_U.T @ U[i].T) ? No.
                # Let u_i be vector U[i, :].
                # sum_c u_i[c] * M_U[idx, c] = 1 for all idx.
                # This is u_i @ M_U.T = 1 (vector of ones).
                # So M_U.T @ u_i.T = 1.T? No.
                # M_U is (m*p, rank).
                # u_i is (1, rank).
                # u_i @ M_U.T is (1, m*p).
                # We want this to be ones(1, m*p).
                # So we solve M_U.T @ u_i.T = ones(m*p).
                # Wait, M_U.T is (rank, m*p). u_i.T is (rank, 1).
                # Product is (rank, 1). That's not right.
                
                # Let's re-verify indices.
                # C[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # Fix V, W.
                # For a fixed i, we have variables U[i, 0] ... U[i, rank-1].
                # Equations: for all j, k:
                # sum_c U[i, c] (V[j, c] W[k, c]) = 1
                # Let A_{j,k, c} = V[j, c] W[k, c].
                # sum_c U[i, c] A_{j,k, c} = 1.
                # This is a linear system: A_{j,k, :} dot U[i, :] = 1.
                # Stack all (j,k) into a matrix A_mat of shape (m*p, rank).
                # A_mat[idx, c] = A_{j,k, c}.
                # Then A_mat @ U[i, :].T = ones(m*p).
                # We want to find U[i, :].
                # This is a least squares problem: min || A_mat @ u - 1 ||^2.
                # u is the solution.
                # u, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m*p), rcond=None)
                
                for i in range(n):
                    u, _, _, _ = np.linalg.lstsq(M_U, np.ones(m*p), rcond=None)
                    U[i, :] = u
                    
                # Solve for V
                # Fix U, W.
                # C[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # For fixed j, variables V[j, c].
                # Equations for all i, k:
                # sum_c V[j, c] (U[i, c] W[k, c]) = 1
                # Let B_{i,k, c} = U[i, c] W[k, c].
                # Stack into B_mat of shape (n*p, rank).
                B_mat = np.zeros((n*p, rank))
                for i in range(n):
                    for k in range(p):
                        B_mat[i*p+k, :] = U[i, :] * W[k, :]
                for j in range(m):
                    v, _, _, _ = np.linalg.lstsq(B_mat, np.ones(n*p), rcond=None)
                    V[j, :] = v
                    
                # Solve for W
                # Fix U, V.
                # For fixed k, variables W[k, c].
                # Equations for all i, j:
                # sum_c W[k, c] (U[i, c] V[j, c]) = 1
                # Let C_{i,j, c} = U[i, c] V[j, c].
                # Stack into C_mat of shape (n*m, rank).
                C_mat = np.zeros((n*m, rank))
                for i in range(n):
                    for j in range(m):
                        C_mat[i*m+j, :] = U[i, :] * V[j, :]
                for k in range(p):
                    w, _, _, _ = np.linalg.lstsq(C_mat, np.ones(n*m), rcond=None)
                    W[k, :] = w
                    
                # Check error
                # Reconstruct T
                T_rec = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_rec[i, j, k] = np.sum(U[i, :] * V[j, :] * W[k, :])
                err = np.linalg.norm(T_rec - T)
                if err < 1e-8:
                    break
                    
            if err < 1e-8:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness
                T_rec_snap = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_rec_snap[i, j, k] = np.sum(U_snap[i, :] * V_snap[j, :] * W_snap[k, :])
                
                if np.allclose(T_rec_snap, T, atol=1e-9):
                    # Convert to lists
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    
    # Fallback: Schoolbook
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
