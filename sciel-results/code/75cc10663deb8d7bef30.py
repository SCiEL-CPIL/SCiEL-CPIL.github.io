"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W form a valid tensor decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Check all nonzero entries: T[(i,j), (j,k), (i,k)] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check zero entries: T[(i,j), (j2,k2), (i2,k2)] = 0 unless i2=i, j2=j
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        if j2 != j:
                            val = np.sum(U[i*m+j, :] * V[j2*p+k, :] * W[k*n+i, :])
                            if abs(val) > tol:
                                return False
        for i in range(n):
            for i2 in range(n):
                for j in range(m):
                    for k in range(p):
                        if i2 != i:
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i2, :])
                            if abs(val) > tol:
                                return False
        for j in range(m):
            for k in range(p):
                for k2 in range(p):
                    for i in range(n):
                        if k2 != k:
                            val = np.sum(U[i*m+j, :] * V[j*p+k2, :] * W[k*n+i, :])
                            if abs(val) > tol:
                                return False
        return True

    def try_rank_fast(r, max_iters=30):
        """Fast ALS for rank-r decomposition."""
        if time.time() > deadline - 15:
            return None
        
        np.random.seed(42 + r)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Build target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Update U
            KR_VW = np.einsum('ar,br->abr', V, W).reshape(m*p*p*n, r)
            for row in range(n*m):
                T_row = T[row].reshape(-1)
                U[row, :] = np.linalg.lstsq(KR_VW, T_row, rcond=None)[0]
            
            # Update V
            KR_UW = np.einsum('ar,br->abr', U, W).reshape(n*m*p*n, r)
            for row in range(m*p):
                T_col = T[:, row, :].reshape(-1)
                V[row, :] = np.linalg.lstsq(KR_UW, T_col, rcond=None)[0]
            
            # Update W
            KR_UV = np.einsum('ar,br->abr', U, V).reshape(n*m*m*p, r)
            for row in range(p*n):
                T_col = T[:, :, row].reshape(-1)
                W[row, :] = np.linalg.lstsq(KR_UV, T_col, rcond=None)[0]
            
            # Periodic check
            if it % 5 == 4:
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    return [U.tolist(), V.tolist(), W.tolist()]
        
        if verify(U.tolist(), V.tolist(), W.tolist()):
            return [U.tolist(), V.tolist(), W.tolist()]
        return None

    # Try ranks from 46 down
    for r in [46, 44, 42, 40, 38]:
        if time.time() > deadline - 30:
            break
        result = try_rank_fast(r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
