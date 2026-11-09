import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor: T[i,j,k] = 1 for all valid indices
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Start with schoolbook decomposition as fallback
    rank_sb = nm * mp * pn // m  # n*m*p = 24
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # Lower bound: max(nm, mp, pn) = max(6, 12, 8) = 12
    # Try ranks from 12 to 23
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Reshape to 3D tensors
        U3 = U.reshape(n, m, -1)  # (n, m, rank)
        V3 = V.reshape(m, p, -1)  # (m, p, rank)
        W3 = W.reshape(p, n, -1)  # (p, n, rank)
        
        # Compute tensor product sum
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        rank = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(rank):
            T += U3[:,:,c][:, np.newaxis, :] * V3[:,:,c][np.newaxis, :, :] * W3[:,:,c][np.newaxis, np.newaxis, :]
            # More precisely:
            # T[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        # Recompute properly
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
        
        # Check if all entries are 1
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)
    
    def try_rank(r):
        """Try to find a rank-r decomposition using optimization."""
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        
        # Try multiple initializations
        for init in range(5):
            if time.time() > deadline:
                break
            
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Use gradient descent to minimize the error
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Compute current tensor
                T = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i,j,k] = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                
                # Error
                err = T - 1.0
                if np.max(np.abs(err)) < 1e-10:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = err[i,j,k]
                            for c in range(r):
                                dU[i*m+j, c] += 2 * e * V[j*p+k, c] * W[k*n+i, c]
                                dV[j*p+k, c] += 2 * e * U[i*m+j, c] * W[k*n+i, c]
                                dW[k*n+i, c] += 2 * e * U[i*m+j, c] * V[j*p+k, c]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Periodically try to round and verify
                if step % 200 == 0:
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    if verify(U_rounded, V_rounded, W_rounded):
                        return (U_rounded, V_rounded, W_rounded)
            
            # Final check
            if np.max(np.abs(T - 1.0)) < 1e-10:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if verify(U_rounded, V_rounded, W_rounded):
                    return (U_rounded, V_rounded, W_rounded)
        
        return None
    
    # Try ranks from 12 upward
    for r in range(12, 24):
        if time.time() > deadline:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
