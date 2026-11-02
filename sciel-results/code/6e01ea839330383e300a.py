import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before 150s kill
    
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    # Target: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # Build the target tensor in flattened form
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    def verify(U, V, W, rank):
        """Verify exact decomposition. Returns True if valid."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank] * V[j*p+k, :rank], W[k*n+i, :rank])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    def try_rank(rank, max_iters=200, n_restarts=5, seed=42):
        """Try to find a valid rank-r decomposition using ALS."""
        if rank < 1:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            rng = np.random.RandomState(seed + restart * 1000)
            
            # Initialize
            U = rng.randn(nm, rank)
            V = rng.randn(mp, rank)
            W = rng.randn(pn, rank)
            
            # Normalize to avoid numerical issues
            U /= np.linalg.norm(U) + 1e-10
            V /= np.linalg.norm(V) + 1e-10
            W /= np.linalg.norm(W) + 1e-10
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j), solve U[i*m+j, :] to minimize error
                for i in range(n):
                    for j in range(m):
                        # U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c] summed over k)
                        # For fixed (i,j), the target is 1 for all k
                        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                        # Let X[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        # Then sum_c U[i*m+j, c] * X[k, c] = 1 for all k
                        # This is: X @ U[i*m+j, :] = ones(p)
                        X = V[j*p:(j+1)*p, :] * W.reshape(p, n, rank)[:, i, :]
                        # X is (p, rank), U_row is (rank,), want X @ U_row = ones(p)
                        # Least squares: U_row = pinv(X) @ ones(p)
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(X, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k)
                for j in range(m):
                    for k in range(p):
                        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                        # Let Y[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        # Y @ V[j*p+k, :] = ones(n)
                        Y = U.reshape(n, m, rank)[:, j, :] * W.reshape(p, n, rank)[k, :, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(Y, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i)
                for k in range(p):
                    for i in range(n):
                        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                        # Let Z[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        # Z @ W[k*n+i, :] = ones(m)
                        Z = U.reshape(n, m, rank)[i, :, :] * V.reshape(m, p, rank)[:, k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(Z, np.ones(m), rcond=None)[0]
                        except:
                            pass
            
            # Check if we got a good solution
            # Round to integers and verify
            U_int = np.round(U).astype(int)
            V_int = np.round(V).astype(int)
            W_int = np.round(W).astype(int)
            
            if verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
            # Also try rounding to half-integers
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            
            if verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), rank):
                return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
        return None
    
    # Try ranks from low to high
    best_result = None
    for rank in range(12, 31):
        if time.time() > deadline - 10:
            break
        result = try_rank(rank)
        if result is not None:
            best_result = (rank, result)
            break  # Found the lowest rank, stop
    
    if best_result is not None:
        rank, (U, V, W) = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 30
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
