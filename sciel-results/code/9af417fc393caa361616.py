import numpy as np
import time
import random

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T of shape (dim_u, dim_v, dim_w)
    # T[i, j, k] = 1 if this term appears in schoolbook multiplication
    # Schoolbook: C[i, k] = sum_j A[i, j] B[j, k]
    # Index mapping:
    # A index: i * m + j
    # B index: j * p + k
    # C index: k * n + i  (Wait, the prompt says W[k*n+i, r]. So C index is k*n + i)
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    def check_decomposition(U, V, W, T):
        # U, V, W are lists of lists or arrays
        # Check if sum_r U[:,r] V[:,r] W[:,r] == T
        # This is expensive if done naively.
        # Better: for each r, compute outer product and sum.
        # Or flatten T and check linear combination.
        rank = U.shape[1]
        T_flat = T.flatten()
        # Build the linear system for the coefficients? No, we have the matrices.
        # Reconstruct T_recon
        T_recon = np.zeros_like(T)
        for r in range(rank):
            T_recon += np.einsum('ir,jr,kr->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_recon, T, atol=1e-9)

    def search_rank(rank, max_iters=1000, restarts=10):
        best_error = float('inf')
        best_solution = None
        
        for _ in range(restarts):
            # Initialize with small random integers/half-integers
            # Or random reals, then snap later?
            # Let's initialize with small random reals
            U = np.random.randn(dim_u, rank) * 0.5
            V = np.random.randn(dim_v, rank) * 0.5
            W = np.random.randn(dim_w, rank) * 0.5
            
            for it in range(max_iters):
                # Update U
                # T = sum_r U_r \otimes V_r \otimes W_r
                # Fix V, W. For each i, T[i, :, :] = sum_r U[i, r] V_r \otimes W_r
                # This is a linear system for U[i, :]
                # We can solve for all i at once?
                # Let's solve for one i at a time for simplicity, or batch.
                
                # Construct RHS for U update
                # For each i, we have T[i, a, b] = sum_r U[i, r] * (V[r, a] * W[r, b])
                # Let M_r = V[:, r] \otimes W[:, r]
                # Then T[i, :, :] (flattened) = sum_r U[i, r] M_r (flattened)
                # This is a linear system: X U[i, :]^T = T[i, :, :]^T
                # where X is (dim_v * dim_w) x rank
                
                X = np.zeros((dim_v * dim_w, rank))
                for r in range(rank):
                    X[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                
                # Solve for U
                U_new = np.zeros((dim_u, rank))
                for i in range(dim_u):
                    rhs = T[i, :, :].flatten()
                    # Least squares
                    sol, _, _, _ = np.linalg.lstsq(X, rhs, rcond=None)
                    U_new[i, :] = sol
                
                # Update V
                Y = np.zeros((dim_u * dim_w, rank))
                for r in range(rank):
                    Y[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                V_new = np.zeros((dim_v, rank))
                for j in range(dim_v):
                    rhs = T[:, j, :].flatten() # Note: T is (i, j, k). We fix j.
                    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                    # For fixed j, T[:, j, :] (i, k) = sum_r V[j, r] (U[:, r] \otimes W[:, r])
                    # So RHS is T[:, j, :] flattened.
                    sol, _, _, _ = np.linalg.lstsq(Y, rhs, rcond=None)
                    V_new[j, :] = sol

                # Update W
                Z = np.zeros((dim_u * dim_v, rank))
                for r in range(rank):
                    Z[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                W_new = np.zeros((dim_w, rank))
                for k in range(dim_w):
                    rhs = T[:, :, k].flatten() # Fix k
                    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                    # For fixed k, T[:, :, k] (i, j) = sum_r W[k, r] (U[:, r] \otimes V[:, r])
                    sol, _, _, _ = np.linalg.lstsq(Z, rhs, rcond=None)
                    W_new[k, :] = sol

                U, V, W = U_new, V_new, W_new
                
                # Check error
                T_recon = np.zeros_like(T)
                for r in range(rank):
                    T_recon += np.einsum('ir,jr,kr->ijk', U[:, r], V[:, r], W[:, r])
                error = np.linalg.norm(T_recon - T)
                
                if error < best_error:
                    best_error = error
                    best_solution = (U.copy(), V.copy(), W.copy())
                
                if error < 1e-6:
                    break
            
            # If we found a good numerical solution, try to rationalize
            if best_error < 1e-5:
                U_r, V_r, W_r = best_solution
                # Snap to nearest half-integer
                U_snap = np.round(U_r * 2) / 2.0
                V_snap = np.round(V_r * 2) / 2.0
                W_snap = np.round(W_r * 2) / 2.0
                
                if check_decomposition(U_snap, V_snap, W_snap, T):
                    return U_snap, V_snap, W_snap

        return None

    start_time = time.time()
    
    # Try ranks from low to high
    for rank in [17, 18, 19, 20]:
        if time.time() - start_time > 100:
            break
            
        solution = search_rank(rank, max_iters=500, restarts=5)
        if solution:
            U, V, W = solution
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
